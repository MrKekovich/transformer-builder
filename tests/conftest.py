import pytest
import torch
from torch import nn


@pytest.fixture(scope="function")
def manual_seed(seed: int = 42):
    torch.manual_seed(seed)


def backpropagation(
    model: nn.Module,
    input_tensor: torch.Tensor,
    loss_fn: nn.Module = None,
    optimizer: nn.Module = None,
) -> None:
    loss_fn = loss_fn or nn.MSELoss()
    parameters = list(model.parameters())

    if not optimizer and not parameters:
        return

    if not optimizer:
        optimizer = torch.optim.Adam(parameters)

    optimizer.zero_grad()
    loss = loss_fn(model(input_tensor), input_tensor)
    loss.backward()
    optimizer.step()
