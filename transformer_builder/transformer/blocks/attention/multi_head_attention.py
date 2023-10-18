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
        """
        Args:
            embedding_dimension: The dimension of the embedding.

            attention_blocks: A list of attention blocks.

            layer_before: Feed forward layer before multi-head attention.

            layer_after: Feed forward layer after multi-head attention.
        """
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.attention_blocks = attention_blocks

        self._number_of_heads = len(attention_blocks)

        self.validate()

        self.layer_before: nn.Module = layer_before
        self.layer_after: nn.Module = layer_after

    def validate(self) -> None:
        if self._number_of_heads < 1:
            raise ValueError(f"Number of heads ({self._number_of_heads}) must be >= 1")
        if self.embedding_dimension % self._number_of_heads != 0:
            raise ValueError(
                f"Embedding dimension ({self.embedding_dimension}) "
                f"must be divisible by number of heads ({self._number_of_heads})"
            )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:
        # x.shape = [batch_size, seq_len, embedding_dim]
        x = self.layer_before(x)
        heads_output = [attention_head(x) for attention_head in self.attention_blocks]

        x = torch.cat(heads_output, dim=-1)

        x = self.layer_after(x)

        return x
