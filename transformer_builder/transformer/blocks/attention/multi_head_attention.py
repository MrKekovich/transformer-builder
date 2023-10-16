import torch
from torch import nn


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        embedding_dimension: int,
        attention_blocks: list[nn.Module],
        layer_before: nn.Module = nn.Identity(),
        layer_after: nn.Module = nn.Identity(),
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.attention_blocks = attention_blocks

        self._number_of_heads = len(attention_blocks)

        if self.embedding_dimension % self._number_of_heads != 0:
            raise ValueError(
                f"embedding dimension ({self.embedding_dimension}) "
                f"must be divisible by number of heads ({self._number_of_heads})"
            )

        self.layer_before: nn.Module = layer_before
        self.layer_after: nn.Module = layer_after

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x.shape = [batch_size, seq_len, embedding_dim]
        x = self.layer_before(x)

        embedding_chunks = torch.chunk(x, self._number_of_heads, dim=-1)
        heads_output = [
            attention_head(x)
            for attention_head, x in zip(self.attention_blocks, embedding_chunks)
        ]

        x = torch.cat(heads_output, dim=-1)

        x = self.layer_after(x)

        return x
