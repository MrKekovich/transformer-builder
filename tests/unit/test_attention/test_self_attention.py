import pytest
import torch
from hypothesis import given, strategies as st
from torch import nn

from transformer_builder.attention import SelfAttention


@given(
    embedding_dim=st.integers(min_value=1, max_value=10),
    num_heads=st.integers(min_value=1, max_value=10),
)
@pytest.mark.usefixtures("manual_seed")
def test_creation(embedding_dim: int, num_heads: int):
    embedding_dim = embedding_dim * num_heads
    self_attention_get_all_cases(embedding_dim=embedding_dim, num_heads=num_heads)


def test_mask_error():
    attention = SelfAttention(casual_masking=True)
    input_tensor = torch.rand(1, 1, 1)
    mask = torch.rand(1, 1, 1)
    with pytest.raises(RuntimeError):
        attention(x=input_tensor, mask=mask)


def self_attention_get_all_cases(embedding_dim: int = 3, num_heads: int = 1):
    return [
        self_attention_case_default(),
        self_attention_case_simple(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        ),
        self_attention_case_division_in_last_layer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        ),
        self_attention_case_concatenation_in_last_layer(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        ),
        self_attention_case_nested_self_attention(
            embedding_dim=embedding_dim,
            num_heads=num_heads,
        ),
    ]


def self_attention_case_default():
    return SelfAttention()


def self_attention_case_simple(
    embedding_dim: int,
    num_heads: int = 3,
):
    return SelfAttention(
        q_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
        k_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
        v_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
    )


def self_attention_case_division_in_last_layer(
    embedding_dim: int,
    num_heads: int = 3,
):
    assert (
        embedding_dim % num_heads == 0
    ), "embedding_dim must be divisible by num_heads"
    return SelfAttention(
        layer_before=nn.Linear(embedding_dim, embedding_dim),
        q_architecture=nn.Linear(embedding_dim, embedding_dim),
        k_architecture=nn.Linear(embedding_dim, embedding_dim),
        v_architecture=nn.Linear(embedding_dim, embedding_dim),
        layer_after=nn.Linear(embedding_dim, embedding_dim // num_heads),
    )


def self_attention_case_concatenation_in_last_layer(
    embedding_dim: int,
    num_heads: int = 3,
):
    assert (
        embedding_dim % num_heads == 0
    ), "embedding_dim must be divisible by num_heads"
    return SelfAttention(
        layer_before=nn.Linear(embedding_dim, embedding_dim),
        q_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
        k_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
        v_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
        layer_after=nn.Linear(embedding_dim // num_heads, embedding_dim),
    )


def self_attention_case_nested_self_attention(
    embedding_dim: int,
    num_heads: int = 3,
):
    assert (
        embedding_dim % num_heads == 0
    ), "embedding_dim must be divisible by num_heads"
    return SelfAttention(
        layer_before=SelfAttention(),
        q_architecture=SelfAttention(
            layer_before=nn.Linear(embedding_dim, embedding_dim),
            q_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
            k_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
            v_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
            layer_after=nn.Linear(
                embedding_dim // num_heads, embedding_dim // num_heads
            ),
        ),
        k_architecture=SelfAttention(
            layer_before=SelfAttention(
                layer_before=SelfAttention(),
                layer_after=SelfAttention(
                    layer_before=nn.Linear(embedding_dim, embedding_dim),
                    layer_after=nn.Linear(embedding_dim, embedding_dim // 3),
                ),
            ),
        ),
        v_architecture=SelfAttention(
            layer_after=SelfAttention(
                layer_after=nn.Linear(embedding_dim, embedding_dim // 3),
            )
        ),
        layer_after=nn.Linear(embedding_dim // 3, embedding_dim // 3),
    )
