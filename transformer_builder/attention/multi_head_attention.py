import torch
from torch import nn

from .self_attention import SelfAttention


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        layer_before: nn.Module = nn.Identity(),
        self_attention_heads: list[SelfAttention] = None,
        layer_after: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.layer_before = layer_before

        self.self_attention_heads = self_attention_heads or [SelfAttention()]

        if not self.self_attention_heads:
            raise ValueError("self_attention_heads must not be empty")

        self.layer_after = layer_after

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.layer_before(x)
        heads_output = [
            self_attention_head(x, mask)
            for self_attention_head in self.self_attention_heads
        ]

        x = torch.concat(heads_output, dim=-1)

        return self.layer_after(x)
