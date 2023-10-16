import torch
from hypothesis import given, strategies as st

from transformer_builder.transformer import MultiHeadAttention, SelfAttentionBlock


@given(
    num_heads=st.integers(min_value=1, max_value=10),
    batch_size=st.integers(min_value=1, max_value=10),
    seq_len=st.integers(min_value=1, max_value=10),
    embedding_dim=st.integers(min_value=1, max_value=10),
)
def test_multi_head_attention(
    num_heads: int,
    batch_size: int,
    seq_len: int,
    embedding_dim: int,
) -> None:
    embedding_dim_for_one_head = embedding_dim
    embedding_dim *= num_heads

    input_tensor = torch.randn(batch_size, seq_len, embedding_dim)

    attention = MultiHeadAttention(
        embedding_dimension=embedding_dim,
        attention_blocks=[
            SelfAttentionBlock(embedding_dimension=embedding_dim_for_one_head)
            for _ in range(num_heads)
        ],
    )

    attention_output = attention(input_tensor)

    assert attention_output.shape == input_tensor.shape
