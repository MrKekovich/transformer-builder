import pytest
import torch
from hypothesis import given, strategies as st, settings

from tests.helpers import self_attention_get_all_cases


@given(
    embedding_dim=st.integers(min_value=1, max_value=10),
)
@pytest.mark.usefixtures("manual_seed")
def test_creation(embedding_dim: int):
    self_attention_get_all_cases(embedding_dim=embedding_dim)


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
    input_tensor_without_batch = torch.rand(sequence_length, embedding_dim)

    for attention in self_attention_get_all_cases(embedding_dim=embedding_dim):
        attention(x=input_tensor)
        attention(x=input_tensor_without_batch)


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    sequence_length=st.integers(min_value=1, max_value=8),
    embedding_dim=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_forward_with_mask(
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
):
    input_tensor = torch.rand(batch_size, sequence_length, embedding_dim)
    input_tensor_without_batch = torch.rand(sequence_length, embedding_dim)

    mask = torch.rand(batch_size, sequence_length, sequence_length)
    mask_without_batch = torch.rand(sequence_length, sequence_length)

    for attention in self_attention_get_all_cases(embedding_dim=embedding_dim):
        attention(x=input_tensor, mask=mask)
        attention(x=input_tensor, mask=mask_without_batch)
        attention(x=input_tensor_without_batch, mask=mask_without_batch)


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    sequence_length=st.integers(min_value=1, max_value=8),
    embedding_dim=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_forward_with_casual_mask(
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
):
    input_tensor = torch.rand(batch_size, sequence_length, embedding_dim)
    input_tensor_without_batch = torch.rand(sequence_length, embedding_dim)

    for attention in self_attention_get_all_cases(embedding_dim=embedding_dim):
        attention.casual_masking = True
        attention(x=input_tensor)
        attention(x=input_tensor_without_batch)


@given(
    batch_size=st.integers(min_value=1, max_value=8),
    sequence_length=st.integers(min_value=1, max_value=8),
    embedding_dim=st.integers(min_value=1, max_value=10),
)
@settings(deadline=None)
def test_forward_with_mask_error(
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
):
    input_tensor = torch.rand(batch_size, sequence_length, embedding_dim)
    input_tensor_without_batch = torch.rand(sequence_length, embedding_dim)

    mask = torch.rand(batch_size, sequence_length, sequence_length)
    mask_without_batch = torch.rand(sequence_length, sequence_length)

    for attention in self_attention_get_all_cases(embedding_dim=embedding_dim):
        attention.casual_masking = True
        with pytest.raises(RuntimeError):
            attention(x=input_tensor, mask=mask)
        with pytest.raises(RuntimeError):
            attention(x=input_tensor, mask=mask_without_batch)
        with pytest.raises(RuntimeError):
            attention(x=input_tensor_without_batch, mask=mask_without_batch)
