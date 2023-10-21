import torch
from torch import nn


class ResidualConnection(nn.Module):
    def __init__(
        self,
        layer_norm_dimension: int,
        module: nn.Module,
    ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(layer_norm_dimension)
        self.module = module

    def forward(self, x: torch.Tensor, *args, **kwargs):
        _x = x
        x = self.module(x, *args, **kwargs)
        return self.layer_norm(x + _x)
