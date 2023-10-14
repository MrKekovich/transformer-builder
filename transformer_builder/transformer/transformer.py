from typing import Optional, Any

from torch import nn


class Transformer(nn.Transformer):
    def __init__(
        self,
        model_dimension: int = 512,
        custom_encoder: Optional[Any] = None,
        custom_decoder: Optional[Any] = None,
        batch_first: bool = False,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            d_model=model_dimension,
            custom_encoder=custom_encoder,
            custom_decoder=custom_decoder,
            batch_first=batch_first,
            device=device,
            dtype=dtype,
        )
        ...
