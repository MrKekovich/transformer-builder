import pytest
import torch
from torch import nn

from tests.unit.test_attention.test_self_attention import self_attention_get_all_cases


@pytest.mark.usefixtures("manual_seed")
def test_backpropagation():
    input_tensor = torch.tensor([[1, 2, 3]], dtype=torch.float)
    target_tensor = torch.tensor([[3, 2, 1]], dtype=torch.float)

    loss_fn = nn.MSELoss()

    for attention in self_attention_get_all_cases(embedding_dim=3, num_heads=3):
        try:
            optimizer = torch.optim.Adam(attention.parameters())
        except ValueError as e:
            print(f"Empty list of parameters, skipping...\n{e}")
            continue

        output = attention(x=input_tensor)
        loss = loss_fn(output, target_tensor)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
