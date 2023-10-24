from typing import List

from torch import nn

from transformer_builder.attention import MultiHeadAttention
from .self_attention_cases import (
    self_attention_case_simple,
    self_attention_case_simple_layers_before_and_after,
    self_attention_case_nested,
)


def multi_head_attention_get_all_cases(
    embedding_dim: int = 3,
) -> List[MultiHeadAttention]:
    return [
        multi_head_attention_case_default(),
        multi_head_attention_case_simple(
            embedding_dim=embedding_dim,
        ),
        multi_head_attention_case_simple_layers_before_and_after(
            embedding_dim=embedding_dim,
        ),
        multi_head_attention_case_nested(
            embedding_dim=embedding_dim,
        ),
    ]


def multi_head_attention_case_default() -> MultiHeadAttention:
    return MultiHeadAttention()


def multi_head_attention_case_simple(embedding_dim: int) -> MultiHeadAttention:
    return MultiHeadAttention(
        self_attention_heads=[self_attention_case_simple(embedding_dim=embedding_dim)],
    )


def multi_head_attention_case_simple_layers_before_and_after(
    embedding_dim: int,
) -> MultiHeadAttention:
    # 3 heads require embedding_dim to be divisible by 3.
    # We use embedding_dim * 3 to make it divisible by 3,
    # so we can use any embedding_dim, and it'll be fine.
    d_head = embedding_dim * 3
    return MultiHeadAttention(
        layer_before=nn.Linear(embedding_dim, d_head),
        self_attention_heads=[
            self_attention_case_simple_layers_before_and_after(
                embedding_dim=d_head,
                num_heads=3,
            ),
            self_attention_case_simple_layers_before_and_after(
                embedding_dim=d_head,
                num_heads=3,
            ),
            self_attention_case_simple_layers_before_and_after(
                embedding_dim=d_head,
                num_heads=3,
            ),
        ],
        layer_after=nn.Linear(d_head, embedding_dim),
    )


def multi_head_attention_case_nested(
    embedding_dim: int,
) -> MultiHeadAttention:
    return MultiHeadAttention(
        layer_before=MultiHeadAttention(),
        self_attention_heads=[
            self_attention_case_nested(
                embedding_dim=embedding_dim,
                num_heads=1,
            )
        ],
        layer_after=MultiHeadAttention(
            self_attention_heads=[
                self_attention_case_nested(
                    embedding_dim=embedding_dim,
                    num_heads=1,
                )
            ]
        ),
    )
    pass
