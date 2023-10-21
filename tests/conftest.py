import pytest
import torch


@pytest.fixture(scope="function")
def manual_seed(seed: int = 42):
    torch.manual_seed(seed)
