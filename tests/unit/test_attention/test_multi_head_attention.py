import torch
from hypothesis import given, settings, strategies as st

from transformer_builder.attention import MultiHeadAttention


def test_creation():
    MultiHeadAttention()


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    sequence_length=st.integers(min_value=1, max_value=8),
    embedding_dim=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_forward(
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
):
    input_tensor = torch.rand(batch_size, sequence_length, embedding_dim)
    mult_head_attention = MultiHeadAttention()
    output = mult_head_attention(input_tensor)
    assert output.shape == input_tensor.shape
