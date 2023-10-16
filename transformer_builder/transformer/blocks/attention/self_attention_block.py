import torch
import torch.nn.functional as F
from torch import nn


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        layer_before: nn.Module = nn.Identity(),
        # TODO: remove nn.Identity() for qkv and add default implementation
        k_architecture: nn.Module = nn.Identity(),
        q_architecture: nn.Module = nn.Identity(),
        v_architecture: nn.Module = nn.Identity(),
        custom_attention_mask: torch.Tensor = None,
        dropout: float = 0.1,
        is_causal: bool = False,
        attention_scale_factor: float = None,
        layer_after: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.layer_before = layer_before

        self.k_architecture = k_architecture
        self.q_architecture = q_architecture
        self.v_architecture = v_architecture

        self.custom_attention_mask = custom_attention_mask
        self.dropout = dropout
        self.is_causal = is_causal
        self.attention_scale_factor = attention_scale_factor

        self.layer_after = layer_after

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        x = self.layer_before(x)

        query = self.q_architecture(x)
        key = self.k_architecture(x)
        value = self.v_architecture(x)

        attention = F.scaled_dot_product_attention(
            query=query,
            key=key,
            value=value,
            attn_mask=self.custom_attention_mask,
            dropout_p=self.dropout,
            is_causal=self.is_causal,
            scale=self.attention_scale_factor,
        )

        attention = self.layer_after(attention)
        return attention
