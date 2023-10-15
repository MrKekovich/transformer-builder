from typing import Optional

import torch
from torch import nn

from transformer_builder.transformer import DecoderBlock


class Decoder(nn.Module):
    def __init__(
        self,
        input_dimension: int = 512,
        embedding_dimension: int = 512,
        embedding_layer: nn.Module = None,
        decoder_blocks: list[DecoderBlock] = None,
        normalization: Optional[nn.Module] = None,
        output_dimension: int = 512,
    ):
        super().__init__()
        self.input_dimension = input_dimension

        self.embedding_dimension = embedding_dimension
        self.embedding_layer = embedding_layer or nn.Embedding(
            input_dimension,
            embedding_dimension,
        )

        self.decoder_blocks = decoder_blocks
        self.normalization = normalization or nn.LayerNorm(embedding_dimension)

        self.output_dimension = output_dimension

    def forward(
        self,
        x: torch.Tensor,
        encoder_output: torch.Tensor,
    ) -> torch.Tensor:
        x = self.embedding_layer(x)

        for decoder_block in self.decoder_blocks:
            x = decoder_block(x, encoder_output, self.normalization)

        return x
