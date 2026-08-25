import torch
from torch import nn
import torch.nn.functional as F

class PositionwiseFeedForward(nn.Module):
    def __init__(self, idim: int, hidden_units: int, dropout_rate: float, activation: nn.Module = nn.ReLU()):
        super().__init__()
        self.activation = activation
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, xs: torch.Tensor, shared_w1: torch.Tensor, shared_b1: torch.Tensor, shared_w2: torch.Tensor, shared_b2: torch.Tensor) -> torch.Tensor:
        out = F.linear(xs, shared_w1, shared_b1)
        out = self.activation(out)
        out = self.dropout(out)
        out = F.linear(out, shared_w2, shared_b2)
        return out