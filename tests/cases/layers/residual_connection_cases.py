from typing import Generator

from torch import nn

from tests.cases.attention import multi_head_attention_get_all_cases
from tests.cases.attention import self_attention_get_all_cases
from transformer_builder.layers.residual_connection import ResidualConnection


def residual_connection_get_cases_with_self_attention(
    embedding_dim: int,
) -> Generator[ResidualConnection, None, None]:
    for self_attention in self_attention_get_all_cases(
        embedding_dim=embedding_dim,
    ):
        yield ResidualConnection(
            module=self_attention,
            normalization=nn.LayerNorm(embedding_dim),
        )


def residual_connection_get_cases_with_multi_head_attention(
    embedding_dim: int,
) -> Generator[ResidualConnection, None, None]:
    for multi_head_attention in multi_head_attention_get_all_cases(
        embedding_dim=embedding_dim,
    ):
        yield ResidualConnection(
            module=multi_head_attention,
            normalization=nn.LayerNorm(embedding_dim),
        )
