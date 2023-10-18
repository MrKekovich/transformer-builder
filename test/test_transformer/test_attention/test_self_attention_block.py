import torch
from hypothesis import strategies as st, given
from torch import nn

from transformer_builder.transformer import SelfAttentionBlock


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    seq_len=st.integers(min_value=1, max_value=10),
    embedding_dim=st.integers(min_value=1, max_value=10),
    dropout=st.floats(min_value=0.1, max_value=1.0),
    is_causal=st.booleans(),
    diagonal=st.integers(min_value=1, max_value=10),
    use_mask=st.booleans(),
)
def test_creation_and_forward(
    batch_size: int,
    seq_len: int,
    embedding_dim: int,
    dropout: float,
    is_causal: bool,
    diagonal: int,
    use_mask: bool,
) -> None:
    tensor = torch.randn(batch_size, seq_len, embedding_dim)
    mask = torch.ones(seq_len, seq_len).tril(diagonal=diagonal)

    attention_default = SelfAttentionBlock()
    assert attention_default(tensor).shape == tensor.shape

    attention_custom = SelfAttentionBlock(
        layer_before=nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
        q_architecture=nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
        k_architecture=nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
        v_architecture=nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
        custom_attention_mask=mask if not is_causal and use_mask else None,
        dropout=dropout,
        is_causal=is_causal if use_mask else False,
        attention_scale_factor=None,
        layer_after=nn.Linear(in_features=embedding_dim, out_features=embedding_dim),
    )
    assert attention_custom(tensor).shape == tensor.shape
