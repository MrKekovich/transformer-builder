import torch
import torch.nn.functional as F
from torch import nn


class SelfAttentionBlock(nn.Module):
    def __init__(
        self,
        layer_before: nn.Module = nn.Identity(),
        k_architecture: nn.Module = nn.Identity(),
        q_architecture: nn.Module = nn.Identity(),
        v_architecture: nn.Module = nn.Identity(),
        custom_attention_mask: torch.Tensor = None,
        dropout: float = 0.0,
        is_causal: bool = False,
        attention_scale_factor: float = None,
        layer_after: nn.Module = nn.Identity(),
    ) -> None:
        """
        Args:
            layer_before (nn.Module): Feed forward layer before self-attention.

            k_architecture (nn.Module): The architecture of key computation.

            q_architecture (nn.Module): The architecture of query computation.

            v_architecture (nn.Module): The architecture of value computation.

            custom_attention_mask (torch.Tensor): Used to mask the attention.
            Must be None when is_causal is True

            dropout (float): The attention dropout.

            is_causal (bool): Casual masking.
            When true - will apply triangle mask with diagonal = 1.

            attention_scale_factor (float): The attention scale factor.
            Applied to the q @ k product and is used to normalize the attention.

            layer_after (nn.Module): Feed forward layer after self-attention.
        """
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
        """
        Args:
            x (torch.Tensor): The input tensor.
            Size must be compatible with embedding dimension to perform matmul.

        Returns:
            (torch.Tensor): The Scaled dot product of attention.
        """
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
