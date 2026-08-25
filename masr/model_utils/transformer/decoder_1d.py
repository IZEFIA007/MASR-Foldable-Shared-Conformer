from typing import List, Optional, Tuple

import torch
from torch import nn
from typeguard import typechecked

from masr.model_utils.conformer.attention import MultiHeadedAttention
from masr.model_utils.conformer.embedding import PositionalEncoding
from masr.model_utils.conformer.positionwise import PositionwiseFeedForward
from masr.model_utils.utils.mask import (subsequent_mask, make_pad_mask)


class TransformerDecoder(nn.Module):
    """单向 Transformer 解码器
    结构说明：
    - 仅保留左向（从前到后）自回归注意力；
    - 取消 BiTransformerDecoder 的右向部分；
    - 仅保留标准的 encoder-decoder 注意机制；
    """

    @typechecked
    def __init__(self,
                 vocab_size: int,
                 encoder_output_size: int,
                 attention_heads: int = 4,
                 linear_units: int = 2048,
                 num_blocks: int = 6,
                 dropout_rate: float = 0.1,
                 positional_dropout_rate: float = 0.1,
                 self_attention_dropout_rate: float = 0.0,
                 src_attention_dropout_rate: float = 0.0,
                 input_layer: str = "embed",
                 use_output_layer: bool = True,
                 normalize_before: bool = True,
                 concat_after: bool = False,
                 max_len: int = 5000):
        super().__init__()

        attention_dim = encoder_output_size

        # === 输入层 ===
        if input_layer == "embed":
            self.embed = nn.Sequential(
                nn.Embedding(vocab_size, attention_dim),
                PositionalEncoding(attention_dim, positional_dropout_rate, max_len=max_len),
            )
        else:
            raise ValueError(f"only 'embed' is supported: {input_layer}")

        self.normalize_before = normalize_before
        self.after_norm = nn.LayerNorm(attention_dim, eps=1e-12)
        self.use_output_layer = use_output_layer
        self.output_layer = nn.Linear(attention_dim, vocab_size)

        # === 解码层堆叠 ===
        self.decoders = nn.ModuleList([
            DecoderLayer(
                size=attention_dim,
                self_attn=MultiHeadedAttention(attention_heads, attention_dim, self_attention_dropout_rate),
                src_attn=MultiHeadedAttention(attention_heads, attention_dim, src_attention_dropout_rate),
                feed_forward=PositionwiseFeedForward(attention_dim, linear_units, dropout_rate),
                dropout_rate=dropout_rate,
                normalize_before=normalize_before,
                concat_after=concat_after,
            ) for _ in range(num_blocks)
        ])

    # === 训练或批量推理阶段：输入整个目标序列 ===
    def forward(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            ys_in_pad: torch.Tensor,
            ys_in_lens: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            memory: 编码器输出 (B, Tmax, D)
            memory_mask: 编码器掩码 (B, 1, Tmax)
            ys_in_pad: 目标序列 (B, L)
            ys_in_lens: 序列长度 (B,)
        Returns:
            x: 解码输出 (B, L, vocab_size)
            olens: 有效长度 (B,)
        """
        tgt = ys_in_pad
        maxlen = tgt.size(1)

        # 生成目标 mask（单向 + padding）
        tgt_mask = ~make_pad_mask(ys_in_lens, maxlen).unsqueeze(1).to(tgt.device)
        subsequent = subsequent_mask(maxlen, device=tgt.device).unsqueeze(0)
        tgt_mask = tgt_mask & subsequent  # 单向掩码

        x, _ = self.embed(tgt)

        for layer in self.decoders:
            x, tgt_mask, memory, memory_mask = layer(x, tgt_mask, memory, memory_mask)

        if self.normalize_before:
            x = self.after_norm(x)
        if self.use_output_layer:
            x = self.output_layer(x)

        olens = tgt_mask.sum(1)
        return x, olens

    # === 推理解码阶段（逐步生成） ===
    def forward_one_step(
            self,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            cache: Optional[List[torch.Tensor]] = None,
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Args:
            memory: 编码器输出 (B, Tmax, D)
            memory_mask: 编码器掩码 (B, 1, Tmax)
            tgt: 当前目标序列 (B, L)
            tgt_mask: 目标掩码 (B, L, L)
            cache: 每层缓存的历史输出
        Returns:
            y: 当前步的输出 logits (B, vocab)
            new_cache: 更新后的缓存
        """
        x, _ = self.embed(tgt)
        new_cache = []

        for i, decoder in enumerate(self.decoders):
            c = None if cache is None else cache[i]
            x, tgt_mask, memory, memory_mask = decoder(
                x, tgt_mask, memory, memory_mask, cache=c)
            new_cache.append(x)

        if self.normalize_before:
            y = self.after_norm(x[:, -1])
        else:
            y = x[:, -1]

        if self.use_output_layer:
            y = torch.nn.functional.log_softmax(self.output_layer(y), dim=-1)

        return y, new_cache


class DecoderLayer(nn.Module):
    """Transformer 单层解码结构"""

    def __init__(self,
                 size: int,
                 self_attn: nn.Module,
                 src_attn: nn.Module,
                 feed_forward: nn.Module,
                 dropout_rate: float,
                 normalize_before: bool = True,
                 concat_after: bool = False):
        super().__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.norm1 = nn.LayerNorm(size, eps=1e-12)
        self.norm2 = nn.LayerNorm(size, eps=1e-12)
        self.norm3 = nn.LayerNorm(size, eps=1e-12)
        self.dropout = nn.Dropout(dropout_rate)
        self.normalize_before = normalize_before
        self.concat_after = concat_after

        if self.concat_after:
            self.concat_linear1 = nn.Linear(size + size, size)
            self.concat_linear2 = nn.Linear(size + size, size)
        else:
            self.concat_linear1 = nn.Identity()
            self.concat_linear2 = nn.Identity()

    def forward(
            self,
            tgt: torch.Tensor,
            tgt_mask: torch.Tensor,
            memory: torch.Tensor,
            memory_mask: torch.Tensor,
            cache: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """单层前向传播"""
        residual = tgt
        if self.normalize_before:
            tgt = self.norm1(tgt)

        if cache is None:
            tgt_q = tgt
            tgt_q_mask = tgt_mask
        else:
            tgt_q = tgt[:, -1:, :]
            residual = residual[:, -1:, :]
            tgt_q_mask = tgt_mask[:, -1:, :]

        # === 自注意力 ===
        if self.concat_after:
            tgt_concat = torch.concat(
                (tgt_q, self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0]), dim=-1)
            x = residual + self.concat_linear1(tgt_concat)
        else:
            x = residual + self.dropout(self.self_attn(tgt_q, tgt, tgt, tgt_q_mask)[0])

        if not self.normalize_before:
            x = self.norm1(x)

        # === 编码器-解码器注意力 ===
        residual = x
        if self.normalize_before:
            x = self.norm2(x)
        if self.concat_after:
            x_concat = torch.concat(
                (x, self.src_attn(x, memory, memory, memory_mask)[0]), dim=-1)
            x = residual + self.concat_linear2(x_concat)
        else:
            x = residual + self.dropout(self.src_attn(x, memory, memory, memory_mask)[0])
        if not self.normalize_before:
            x = self.norm2(x)

        # === 前馈网络 ===
        residual = x
        if self.normalize_before:
            x = self.norm3(x)
        x = residual + self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm3(x)

        if cache is not None:
            x = torch.concat([cache, x], dim=1)

        return x, tgt_mask, memory, memory_mask
