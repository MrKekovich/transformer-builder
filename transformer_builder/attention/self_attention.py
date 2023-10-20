import torch
from torch import nn
from torch.nn.functional import scaled_dot_product_attention


class SelfAttention(nn.Module):
    def __init__(
        self,
        dropout: float = 0.0,
        casual_masking: bool = False,
        layer_before: nn.Module = nn.Identity(),
        q_architecture: nn.Module = nn.Identity(),
        k_architecture: nn.Module = nn.Identity(),
        v_architecture: nn.Module = nn.Identity(),
        layer_after: nn.Module = nn.Identity(),
        scale: float = None,
    ):
        super().__init__()
        self.dropout = dropout
        self.casual_masking = casual_masking

        self.layer_before = layer_before
        self.q_architecture = q_architecture
        self.k_architecture = k_architecture
        self.v_architecture = v_architecture
        self.layer_after = layer_after

        self.scale = scale

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor = None,
    ) -> torch.Tensor:
        x = self.layer_before(x)

        query = self.q_architecture(x)
        key = self.k_architecture(x)
        value = self.v_architecture(x)

        attention = scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=mask,
            dropout_p=self.dropout,
            scale=self.scale,
            is_causal=self.casual_masking,
        )

        attention = self.layer_after(attention)

        return attention
