import torch
from hypothesis import given, strategies as st, settings
from torch import nn

from transformer_builder.layers.residual_connection import ResidualConnection


@given(st.integers(min_value=1, max_value=10))
def test_creation(input_dim: int) -> None:
    ResidualConnection(
        module=nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, input_dim),
        ),
        normalization=nn.LayerNorm(input_dim),
    )


@given(st.integers(min_value=1, max_value=10))
@settings(deadline=None)
def test_forward(input_dim: int) -> None:
    input_tensor = torch.randn(input_dim, input_dim)
    layer = ResidualConnection(
        module=nn.Sequential(
            nn.Linear(input_dim, input_dim),
            nn.Linear(input_dim, input_dim),
        ),
        normalization=nn.LayerNorm(input_dim),
    )

    layer(input_tensor)
