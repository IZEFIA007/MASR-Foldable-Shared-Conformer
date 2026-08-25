from typing import Optional, Tuple, List
import torch
from torch import nn
from typeguard import typechecked

# 这些 import 全部照搬你可折叠版本用到的模块，确保一致
from masr.model_utils.conformer.attention import MultiHeadedAttention, RelPositionMultiHeadedAttention
from masr.model_utils.conformer.convolution import ConvolutionModule
from masr.model_utils.conformer.embedding import NoPositionalEncoding, PositionalEncoding, RelPositionalEncoding
from masr.model_utils.conformer.subsampling import Conv2dSubsampling4, Conv2dSubsampling6, Conv2dSubsampling8, LinearNoSubsampling
# 注意：这里导入的 PositionwiseFeedForward 必须是你可折叠版本里用的那个（无条件接收共享权重的版本）
from masr.model_utils.conformershare.positionwise import PositionwiseFeedForward   # 或者 conformershare.positionwise，只要和可折叠版本的一样就行
from masr.model_utils.utils.common import get_activation
from masr.model_utils.utils.mask import add_optional_chunk_mask, make_pad_mask

__all__ = ['ConformerEncoder']


class ConformerEncoderLayer(nn.Module):
    """完全复用可折叠版本中的 ConformerEncoderLayer，一模一样"""
    def __init__(
            self,
            size: int,
            self_attn: nn.Module,
            feed_forward: nn.Module,
            feed_forward_macaron: Optional[nn.Module] = None,
            conv_module: Optional[nn.Module] = None,
            dropout_rate: float = 0.1,
            normalize_before: bool = True,
            concat_after: bool = False):
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-5)
        self.norm_mha = nn.LayerNorm(size, eps=1e-5)
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-5)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-5)
            self.norm_final = nn.LayerNorm(size, eps=1e-5)
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before
        self.concat_after = concat_after
        if self.concat_after:
            self.concat_linear = nn.Linear(size + size, size)
        else:
            self.concat_linear = nn.Identity()

    def forward(
            self,
            x: torch.Tensor,
            mask: torch.Tensor,
            pos_emb: torch.Tensor,
            mask_pad: torch.Tensor = torch.ones([0, 0, 0], dtype=torch.bool),
            att_cache: torch.Tensor = torch.zeros([0, 0, 0, 0]),
            cnn_cache: torch.Tensor = torch.zeros([0, 0, 0, 0]),
            shared_ffn_w1=None, shared_ffn_b1=None, shared_ffn_w2=None, shared_ffn_b2=None):
        # Macaron FFN
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x, shared_ffn_w1, shared_ffn_b1, shared_ffn_w2, shared_ffn_b2))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # MHA
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb, cache=att_cache)
        if self.concat_after:
            x_concat = torch.concat((x, x_att), dim=-1)
            x = residual + self.concat_linear(x_concat)
        else:
            x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # Conv
        new_cnn_cache = torch.zeros([0, 0, 0], dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)
            if not self.normalize_before:
                x = self.norm_conv(x)

        # Main FFN
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(
            self.feed_forward(x, shared_ffn_w1, shared_ffn_b1, shared_ffn_w2, shared_ffn_b2))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache


class ConformerEncoder(nn.Module):
    """从 FoldableConformerEncoder 改造而来：去掉展开逻辑，保留全部共享代码"""
    @typechecked
    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_blocks: int = 12,             # 相当于原 num_physical_layers
            dropout_rate: float = 0.1,
            positional_dropout_rate: float = 0.1,
            attention_dropout_rate: float = 0.0,
            input_layer: str = "conv2d",
            pos_enc_layer_type: str = "rel_pos",
            normalize_before: bool = True,
            concat_after: bool = False,
            static_chunk_size: int = 0,
            use_dynamic_chunk: bool = False,
            global_cmvn: torch.nn.Module = None,
            use_dynamic_left_chunk: bool = False,
            macaron_style: bool = True,
            activation_type: str = "swish",
            use_cnn_module: bool = True,
            cnn_module_kernel: int = 15,
            causal: bool = False,
            cnn_module_norm: str = "layer_norm",
            max_len: int = 5000,
            group_size: int = 3,               # 分组大小
            ffn_rank: int = 0                  # 保留参数，但在纯共享时可设为0
    ):
        super().__init__()
        self._output_size = output_size
        self.group_size = group_size
        num_groups = num_blocks // group_size
        if num_blocks % group_size != 0:
            raise ValueError(f"num_blocks ({num_blocks}) must be divisible by group_size ({group_size})")

        # 位置编码
        if pos_enc_layer_type == "abs_pos":
            pos_enc_class = PositionalEncoding
        elif pos_enc_layer_type == "rel_pos":
            pos_enc_class = RelPositionalEncoding
        elif pos_enc_layer_type == "no_pos":
            pos_enc_class = NoPositionalEncoding
        else:
            raise ValueError("unknown pos_enc_layer: " + pos_enc_layer_type)

        # 输入层
        if input_layer == "linear":
            subsampling_class = LinearNoSubsampling
        elif input_layer == "conv2d":
            subsampling_class = Conv2dSubsampling4
        elif input_layer == "conv2d6":
            subsampling_class = Conv2dSubsampling6
        elif input_layer == "conv2d8":
            subsampling_class = Conv2dSubsampling8
        else:
            raise ValueError("unknown input_layer: " + input_layer)

        self.global_cmvn = global_cmvn
        self.embed = subsampling_class(
            idim=input_size,
            odim=output_size,
            dropout_rate=dropout_rate,
            pos_enc_class=pos_enc_class(
                d_model=output_size,
                dropout_rate=positional_dropout_rate,
                max_len=max_len)
        )

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

        activation = get_activation(activation_type)

        # ========== FFN 分组共享权重（直接搬自 FoldableConformerEncoder）==========
        d_model = output_size
        d_ff = linear_units
        self.shared_ffn_w1 = nn.ParameterList()
        self.shared_ffn_b1 = nn.ParameterList()
        self.shared_ffn_w2 = nn.ParameterList()
        self.shared_ffn_b2 = nn.ParameterList()

        for g in range(num_groups):
            w1 = nn.Parameter(torch.randn(d_ff, d_model))
            b1 = nn.Parameter(torch.zeros(d_ff))
            w2 = nn.Parameter(torch.randn(d_model, d_ff))
            b2 = nn.Parameter(torch.zeros(d_model))
            nn.init.xavier_uniform_(w1)
            nn.init.xavier_uniform_(w2)
            nn.init.zeros_(b1)
            nn.init.zeros_(b2)
            self.shared_ffn_w1.append(w1)
            self.shared_ffn_b1.append(b1)
            self.shared_ffn_w2.append(w2)
            self.shared_ffn_b2.append(b2)
        # ========================================================

        # 注意力
        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size, attention_dropout_rate)

        # 前馈：如果你可折叠版本用的 PositionwiseFeedForward 有 ffn_rank 参数，就保留；如果没有，就去掉 ffn_rank
        positionwise_layer = PositionwiseFeedForward
        # 根据你可折叠版本的实际构造函数签名调整参数
        # 假设你可折叠版本是 (output_size, linear_units, dropout_rate, activation, ffn_rank)
        # 如果纯共享，ffn_rank 传 0 表示不用低秩残差
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation, ffn_rank)

        # 卷积
        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm, causal)

        # 构建层（原 physical_layers，这里改成 encoders）
        self.encoders = nn.ModuleList([
            ConformerEncoderLayer(
                size=output_size,
                self_attn=encoder_selfattn_layer(*encoder_selfattn_layer_args),
                feed_forward=positionwise_layer(*positionwise_layer_args),
                feed_forward_macaron=positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                conv_module=convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after
            ) for _ in range(num_blocks)
        ])

    def output_size(self) -> int:
        return self._output_size

    def forward(
            self,
            xs: torch.Tensor,
            xs_lens: torch.Tensor,
            decoding_chunk_size: int = 0,
            num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks
        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)
        # 顺序执行每一层，不展开。共享权重的传递方式与可折叠版本完全一致
        for i, layer in enumerate(self.encoders):
            g = i // self.group_size
            xs, chunk_masks, _, _ = layer(
                xs, chunk_masks, pos_emb, mask_pad,
                shared_ffn_w1=self.shared_ffn_w1[g],
                shared_ffn_b1=self.shared_ffn_b1[g],
                shared_ffn_w2=self.shared_ffn_w2[g],
                shared_ffn_b2=self.shared_ffn_b2[g]
            )

        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, masks

    def forward_chunk(
            self,
            xs: torch.Tensor,
            offset: int,
            required_cache_size: int,
            att_cache: torch.Tensor = torch.zeros([0, 0, 0, 0]),
            cnn_cache: torch.Tensor = torch.zeros([0, 0, 0, 0]),
            att_mask: torch.Tensor = torch.ones([0, 0, 0], dtype=torch.bool)
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        assert xs.size(0) == 1
        tmp_masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)

        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        pos_emb = self.embed.position_encoding(offset=offset - cache_t1, size=attention_key_size)

        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            g = i // self.group_size
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs, att_mask, pos_emb,
                att_cache=att_cache[i:i + 1] if elayers > 0 else att_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache,
                shared_ffn_w1=self.shared_ffn_w1[g],
                shared_ffn_b1=self.shared_ffn_b1[g],
                shared_ffn_w2=self.shared_ffn_w2[g],
                shared_ffn_b2=self.shared_ffn_b2[g]
            )
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache)

        if self.normalize_before:
            xs = self.after_norm(xs)

        r_att_cache = torch.concat(r_att_cache, dim=0)
        r_cnn_cache = torch.stack(r_cnn_cache, dim=0)
        return xs, r_att_cache, r_cnn_cache