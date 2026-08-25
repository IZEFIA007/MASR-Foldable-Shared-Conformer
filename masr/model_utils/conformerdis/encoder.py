from typing import Optional, Tuple, List

import torch
from torch import nn
from typeguard import typechecked

from masr.model_utils.conformer.attention import MultiHeadedAttention
from masr.model_utils.conformer.attention import RelPositionMultiHeadedAttention
from masr.model_utils.conformer.convolution import ConvolutionModule
from masr.model_utils.conformer.embedding import NoPositionalEncoding
from masr.model_utils.conformer.embedding import PositionalEncoding
from masr.model_utils.conformer.embedding import RelPositionalEncoding
from masr.model_utils.conformer.positionwise import PositionwiseFeedForward
from masr.model_utils.conformer.subsampling import Conv2dSubsampling4
from masr.model_utils.conformer.subsampling import Conv2dSubsampling6
from masr.model_utils.conformer.subsampling import Conv2dSubsampling8
from masr.model_utils.conformer.subsampling import LinearNoSubsampling
from masr.model_utils.utils.common import get_activation
from masr.model_utils.utils.mask import add_optional_chunk_mask, make_pad_mask

__all__ = ['FoldableConformerEncoder']


class FoldableConformerEncoder(nn.Module):
    """Foldable Conformer encoder module with structure unfolding."""

    @typechecked
    def __init__(
            self,
            input_size: int,
            output_size: int = 256,
            attention_heads: int = 4,
            linear_units: int = 2048,
            num_physical_layers: int = 6,  # 物理层数（种子层数）
            max_logical_depth: int = 12,    # 最大逻辑深度（可展开到的最大层数）
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
            max_len: int = 5000
    ):
        super().__init__()
        self._output_size = output_size
        self.num_physical_layers = num_physical_layers
        # 修正：使用传入的 max_logical_depth，而不是硬编码
        self.max_logical_depth = max_logical_depth

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
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk

        activation = get_activation(activation_type)

        if pos_enc_layer_type != "rel_pos":
            encoder_selfattn_layer = MultiHeadedAttention
        else:
            encoder_selfattn_layer = RelPositionMultiHeadedAttention
        encoder_selfattn_layer_args = (attention_heads, output_size, attention_dropout_rate)

        positionwise_layer = PositionwiseFeedForward
        positionwise_layer_args = (output_size, linear_units, dropout_rate, activation)

        convolution_layer = ConvolutionModule
        convolution_layer_args = (output_size, cnn_module_kernel, activation, cnn_module_norm, causal)

        # 初始化物理层（种子层）
        self.physical_layers = nn.ModuleList([
            ConformerEncoderLayer(
                size=output_size,
                self_attn=encoder_selfattn_layer(*encoder_selfattn_layer_args),
                feed_forward=positionwise_layer(*positionwise_layer_args),
                feed_forward_macaron=positionwise_layer(*positionwise_layer_args) if macaron_style else None,
                conv_module=convolution_layer(*convolution_layer_args) if use_cnn_module else None,
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after
            ) for _ in range(num_physical_layers)
        ])

    def output_size(self) -> int:
        return self._output_size

    def _get_unfolding_pattern(self, logical_depth: int) -> List[int]:
        """根据逻辑深度计算每个物理层需要重复的次数（返回长度 == num_physical_layers 的列表）"""
        if logical_depth <= self.num_physical_layers:
            # 只执行前 logical_depth 个物理层（每个一次），其余为 0 次
            return [1] * logical_depth + [0] * (self.num_physical_layers - logical_depth)

        # logical_depth > num_physical_layers: 均匀分配重复次数，前 extra 层多重复一次
        base_repeat = logical_depth // self.num_physical_layers
        extra = logical_depth % self.num_physical_layers
        repeats = [base_repeat + 1 if i >= self.num_physical_layers - extra else base_repeat
                   for i in range(self.num_physical_layers)]
        return repeats

    def forward(
            self,
            xs: torch.Tensor,
            xs_lens: torch.Tensor,
            decoding_chunk_size: int = 0,
            num_decoding_left_chunks: int = -1,
            logical_depth: Optional[int] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果 logical_depth 为 None，则使用默认最大深度
        if logical_depth is None:
            logical_depth = self.max_logical_depth
        # 如果 logical_depth 超过最大值，则截断为最大值
        elif logical_depth > self.max_logical_depth:
            logical_depth = self.max_logical_depth

        # 计算原始时间长度（在 subsampling 之前）
        T = xs.size(1)
        masks = ~make_pad_mask(xs_lens, T).unsqueeze(1)

        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)

        # embed 处理（会返回 subsampled xs、pos_emb、以及更新后的 masks）
        xs, pos_emb, masks = self.embed(xs, masks)
        mask_pad = masks

        chunk_masks = add_optional_chunk_mask(xs, masks,
                                              self.use_dynamic_chunk,
                                              self.use_dynamic_left_chunk,
                                              decoding_chunk_size,
                                              self.static_chunk_size,
                                              num_decoding_left_chunks)

        # 计算每个物理层需要重复的次数（长度等于 num_physical_layers）
        repeats = self._get_unfolding_pattern(logical_depth)


        # 训练调试断言，TorchScript 导出不允许 try/except，因此只保留 assert
        # 如果不希望导出时报错，也可以直接注释掉这一行
        assert sum(repeats) == logical_depth, \
            f"Unfolding pattern sum ({sum(repeats)}) != logical_depth ({logical_depth})"

        # 对每个物理层按 repeats 执行（重复使用物理层以实现参数共享）
        for layer_idx, layer in enumerate(self.physical_layers):
            repeat_count = repeats[layer_idx]
            for _ in range(repeat_count):
                xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)

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

        # 流式路径不做 unfolding（和论文/实现一致）
        for i, layer in enumerate(self.physical_layers):
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs, att_mask, pos_emb,
                att_cache=att_cache[i:i + 1] if elayers > 0 else att_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache)
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache)

        if self.normalize_before:
            xs = self.after_norm(xs)

        r_att_cache = torch.concat(r_att_cache, dim=0)
        r_cnn_cache = torch.stack(r_cnn_cache, dim=0)
        return xs, r_att_cache, r_cnn_cache


class ConformerEncoderLayer(nn.Module):
    """Encoder layer module."""

    def __init__(
            self,
            size: int,
            self_attn: nn.Module,
            feed_forward: Optional[nn.Module] = None,
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
            cnn_cache: torch.Tensor = torch.zeros([0, 0, 0, 0])
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

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

        new_cnn_cache = torch.zeros([0, 0, 0], dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)

            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)
        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache
