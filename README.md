# This project is incomplete!

---

# Transformer Builder - Create Custom Transformer Models with Ease

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)

Transformers have become a popular choice for a wide range of Natural Language Processing (NLP) and deep learning tasks.
The Transformer Builder package allows you to create custom transformer models with ease, providing flexibility and
modularity for your deep learning projects.

## Features

- Build custom transformer models with a user-friendly and flexible interface.
- Configurable encoder and decoder blocks with support for custom self-attention mechanisms.
- Encapsulated self-attention blocks that adapt to your specific use case.
- Create encoder and decoder blocks for a wide range of NLP tasks.
- Open-source and customizable to fit your project's requirements.

## Installation

You can install Transformer Builder using pip:

```bash
pip install transformer-builder
```

## Usage

Here's an example of how to use Transformer Builder to create a custom model:

```python
import torch
from torch import nn
from transformer_builder import TransformerBuilder, SelfAttentionBlock

vocabulary_size = 64_000
max_sequence_length = 1024
embedding_dimension = 100

builder = TransformerBuilder(
    vocabulary_size=vocabulary_size,
    max_sequence_length=max_sequence_length,
    embedding_dimension=embedding_dimension,
    default_decoder_block=None,
    default_encoder_block=None,
    default_positional_encoding=PositionalEncoding
)
gpt_model = (
    builder
    .add_embedding(nn.Embedding(vocabulary_size, embedding_dimension))
    .add_positional_encoding(nn.Parameter(torch.zeros(1, max_sequence_length, embedding_dimension)))
    .add_decoder_block(
        nn.Linear(embedding_dimension, embedding_dimension),
        # You can use specific implementation of self-attention block with classes:
        # EncoderSelfAttentionBlock and DecoderSelfAttentionBlock.
        SelfAttentionBlock(  # Uses polymorphism to create different implementations of self-attention block.
            before=nn.Sequential(
                nn.Linear(embedding_dimension, embedding_dimension * 4),
                nn.Linear(embedding_dimension * 4, embedding_dimension),
            ),
            k=nn.Sequential(
                nn.Linear(embedding_dimension, embedding_dimension * 2),
                nn.Linear(embedding_dimension * 2, embedding_dimension),
            ),
            q=builder.decoder_block(),  # You can even pass whole decoder block into self-attention block!
            # The standard decoder_block implementation is copied from GPT-1. 
            # You can change that to your needs by monkey-patching TransformerBuilder or subclassing it.

            # Default values for kqv are nn.Linear(embedding_dimension, embedding_dimension),
            # Default values for `before` and `after` is None which means it won't affect architecture.
            count=10  # Notice that number of heads should be a divisor for embedding dimension.
        ),
        count=10
    ).build()
)

```

## Customization

With Transformer Builder, you can customize each aspect of your blocks individually,
allowing for fine-grained control over your model's architecture.
The example above demonstrates how to configure the self-attention layer,
layer normalization, and linear layers.

## Contributing

If you would like to contribute to this project, please follow our
[contribution guidelines](https://github.com/MrKekovich/transformer-builder/blob/master/CONTRIBUTING.md).

## Support and Feedback

If you have questions, encounter issues, or have feedback, please open an issue on our
[GitHub repository](https://github.com/MrKekovich/transformer-builder).

## Acknowledgments

This project was inspired by the need for a flexible and customizable API for creating
decoder blocks in deep learning models.

## Author

[MrKekovich](https://github.com/MrKekovich)

## License

This project is licensed under the MIT License.
See the [LICENSE](https://github.com/MrKekovich/transformer-builder/blob/master/LICENSE) 
file for details.
