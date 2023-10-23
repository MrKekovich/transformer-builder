import pytest
import torch
from hypothesis import given, settings, strategies as st
from torch import nn

from tests.helpers import self_attention_get_all_cases


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
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)
    target_tensor = torch.randn(batch_size, sequence_length, embedding_dim)

    loss_fn = nn.MSELoss()

    for attention in self_attention_get_all_cases(embedding_dim=embedding_dim):
        try:
            optimizer = torch.optim.Adam(attention.parameters())
        except ValueError as e:
            print(f"Empty list of parameters, skipping...\n{e}")
            continue

        output = attention(x=input_tensor)
        loss = loss_fn(output, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
