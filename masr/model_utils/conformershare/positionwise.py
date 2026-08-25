import torch
from torch import nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int, dropout_rate: float,
                 activation: nn.Module = nn.ReLU(), rank: int = 8):
        """
        Args:
            idim: 输入/输出维度 (d_model)
            hidden_units: FFN 中间层维度 (d_ff)
            dropout_rate: dropout 比例
            activation: 激活函数
            rank: 低秩残差的秩 (用于 A @ B 分解)
        """
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)
        self.rank = rank

        # ---- 第一层 FFN 的低秩残差：A1 (d_model × rank), B1 (rank × d_ff) ----
        self.A1 = nn.Parameter(torch.randn(idim, rank) * 0.02)
        self.B1 = nn.Parameter(torch.randn(rank, hidden_units) * 0.02)

        # ---- 第二层 FFN 的低秩残差：A2 (d_ff × rank), B2 (rank × d_model) ----
        self.A2 = nn.Parameter(torch.randn(hidden_units, rank) * 0.02)
        self.B2 = nn.Parameter(torch.randn(rank, idim) * 0.02)

        # 可选：对角矩阵，如需完全对齐 ResidualTransformer 可取消注释
        # self.D1 = nn.Parameter(torch.zeros(idim, hidden_units))
        # self.D2 = nn.Parameter(torch.zeros(hidden_units, idim))

    def forward(self, xs: torch.Tensor,
                shared_w1: torch.Tensor, shared_b1: torch.Tensor,
                shared_w2: torch.Tensor, shared_b2: torch.Tensor) -> torch.Tensor:
        # ---- 第一层 ----
        # 共享部分
        out_shared1 = F.linear(xs, shared_w1, shared_b1)
        # 低秩残差部分
        residual1 = xs @ self.A1 @ self.B1   # (batch, seq, d_model) -> (..., rank) -> (..., d_ff)
        out1 = out_shared1 + residual1        # 若添加了对角阵 D1，此处再 + self.D1

        out1 = self.activation(out1)
        out1 = self.dropout(out1)

        # ---- 第二层 ----
        out_shared2 = F.linear(out1, shared_w2, shared_b2)
        residual2 = out1 @ self.A2 @ self.B2   # (..., d_ff) -> (..., rank) -> (..., d_model)
        out2 = out_shared2 + residual2         # 若添加了对角阵 D2，此处再 + self.D2

        return out2