from torch import nn


class TransformerBuilder:
    def __init__(
        self,
    ):
        self._fields = {
            "model": nn.Sequential(),
        }

    def add_something(self, name: str, module: nn.Module) -> "TransformerBuilder":
        return self

    def build(self) -> None:
        return
