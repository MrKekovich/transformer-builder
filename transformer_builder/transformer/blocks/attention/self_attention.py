from typing import Optional

from torch import nn


class SelfAttention(nn.Module):
    def __init__(
        self,
        layer_before: Optional[nn.Module] = None,
        k_architecture: Optional[nn.Module] = None,
        q_architecture: Optional[nn.Module] = None,
        v_architecture: Optional[nn.Module] = None,
        layer_after: Optional[nn.Module] = None,
    ):
        super().__init__()
        self.layer_before = layer_before
        self.k_architecture = k_architecture
        self.q_architecture = q_architecture
        self.v_architecture = v_architecture
        self.layer_after = layer_after
