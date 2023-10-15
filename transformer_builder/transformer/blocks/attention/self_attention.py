from typing import Optional, Callable

import torch
from torch import nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        layer_before: Optional[nn.Module] = None,
        k_architecture: Optional[nn.Module] = None,
        q_architecture: Optional[nn.Module] = None,
        v_architecture: Optional[nn.Module] = None,
        scale: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        layer_after: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layer_before = layer_before

        self.k_architecture = k_architecture
        self.q_architecture = q_architecture
        self.v_architecture = v_architecture

        self.scale = scale or (lambda x: x)  # TODO: add real default scale function

        self.layer_after = layer_after

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer_before(x)

        k = self.k_architecture(x)
        q = self.q_architecture(x)
        self.v_architecture(x)

        attention_score = (q @ k) / torch.sqrt(
            torch.tensor(q.size(-1), dtype=torch.float32)
        )

        attention_score = self.scale(attention_score)
        attention_score = torch.softmax(attention_score, dim=-1)

        attention = self.layer_after(attention_score)
        return attention
