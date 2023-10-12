from typing import Optional, Callable


class TransformerBuilder:
    """
    Transformer Builder class
    """

    def __init__(
            self,
            max_sequence_length: int,
            vocabulary_size: int = None,
            embedding_dimension: int = None,
            default_positional_encoding: Optional[Callable] = None,
            default_decoder_block: Optional[Callable] = None,
            default_encoder_block: Optional[Callable] = None,
    ):
        """
        Args:
            max_sequence_length:
            vocabulary_size:
            embedding_dimension:
            default_positional_encoding:
            default_decoder_block:
            default_encoder_block:
        """


    def decoder_block(self, *layers):
        return self

    def add_embedding(self, param):
        return self

    def add_positional_encoding(self, param):
        return self
