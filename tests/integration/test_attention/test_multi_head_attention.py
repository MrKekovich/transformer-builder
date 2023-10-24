import pytest
import torch
from hypothesis import given, strategies as st, settings

from tests.cases.attention import multi_head_attention_get_all_cases
from tests.conftest import backpropagation


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    sequence_length=st.integers(min_value=1, max_value=8),
    embedding_dim=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
@pytest.mark.usefixtures("manual_seed")
def test_backpropagation(
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
) -> None:
    input_tensor = torch.rand(batch_size, sequence_length, embedding_dim)

    for multi_head_attention in multi_head_attention_get_all_cases(
        embedding_dim=embedding_dim
    ):
        backpropagation(
            model=multi_head_attention,
            input_tensor=input_tensor,
        )
