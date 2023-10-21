from torch import nn

from transformer_builder.attention import SelfAttention


def self_attention_get_all_cases(embedding_dim: int = 3):
    return [
        self_attention_case_default(),
        self_attention_case_simple(
            embedding_dim=embedding_dim,
        ),
        self_attention_case_nested_self_attention(
            embedding_dim=embedding_dim,
        ),
    ]


def self_attention_case_default():
    return SelfAttention()


def self_attention_case_simple(
    embedding_dim: int,
    num_heads: int = 1,
):
    return SelfAttention(
        q_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
        k_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
        v_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
    )


def self_attention_layers_before_and_after(
    embedding_dim: int,
    num_heads: int = 1,
):
    return SelfAttention(
        layer_before=nn.Linear(embedding_dim, embedding_dim),
        q_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
        k_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
        v_architecture=nn.Linear(embedding_dim, embedding_dim // num_heads),
        layer_after=nn.Linear(embedding_dim // num_heads, embedding_dim // num_heads),
    )


def self_attention_case_nested_self_attention(
    embedding_dim: int,
    num_heads: int = 1,
):
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
                    layer_after=nn.Linear(embedding_dim, embedding_dim // num_heads),
                ),
            ),
        ),
        v_architecture=SelfAttention(
            layer_after=SelfAttention(
                layer_after=nn.Linear(embedding_dim, embedding_dim // num_heads),
            )
        ),
        layer_after=nn.Linear(embedding_dim // num_heads, embedding_dim // num_heads),
    )