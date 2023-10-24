import torch
from hypothesis import given, settings, strategies as st

from tests.cases.layers import (
    residual_connection_get_cases_with_self_attention,
    residual_connection_get_cases_with_multi_head_attention,
)
from tests.conftest import backpropagation


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    sequence_len=st.integers(min_value=1, max_value=10),
    embedding_dim=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_with_self_attention(
    batch_size: int,
    sequence_len: int,
    embedding_dim: int,
) -> None:
    input_tensor = torch.randn(batch_size, sequence_len, embedding_dim)
    for residual_connection in residual_connection_get_cases_with_self_attention(
        embedding_dim=embedding_dim,
    ):
        residual_connection(input_tensor)


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    sequence_len=st.integers(min_value=1, max_value=10),
    embedding_dim=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_with_multi_head_attention(
    batch_size: int,
    sequence_len: int,
    embedding_dim: int,
) -> None:
    input_tensor = torch.randn(batch_size, sequence_len, embedding_dim)
    for residual_connection in residual_connection_get_cases_with_multi_head_attention(
        embedding_dim=embedding_dim
    ):
        residual_connection(input_tensor)


@given(
    batch_size=st.integers(min_value=1, max_value=10),
    sequence_len=st.integers(min_value=1, max_value=10),
    embedding_dim=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_backpropagation(
    batch_size: int,
    sequence_len: int,
    embedding_dim: int,
) -> None:
    input_tensor = torch.randn(batch_size, sequence_len, embedding_dim)
    for (
        residual_connection_self_attention
    ) in residual_connection_get_cases_with_self_attention(
        embedding_dim=embedding_dim,
    ):
        backpropagation(
            model=residual_connection_self_attention,
            input_tensor=input_tensor,
        )

    for (
        residual_connection_multi_head_attention
    ) in residual_connection_get_cases_with_multi_head_attention(
        embedding_dim=embedding_dim,
    ):
        backpropagation(
            input_tensor=input_tensor,
            model=residual_connection_multi_head_attention,
        )
