from transformer_builder.builders.transformer_builder import TransformerBuilder


def test_init():
    TransformerBuilder(max_sequence_length=1)
