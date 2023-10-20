import pytest
import torch
from hypothesis import given, strategies as st, settings
from torch import nn

from transformer_builder.attention import SelfAttention


@given(
    batch_size=st.integers(min_value=1, max_value=100),
    sequence_length=st.integers(min_value=1, max_value=100),
    embedding_dim=st.integers(min_value=1, max_value=100),
)
@settings(deadline=None)
@pytest.mark.usefixtures("manual_seed")
def test_creation(
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
):
    input_tensor = torch.randn(batch_size, sequence_length, embedding_dim)
    mask = torch.randn(sequence_length, sequence_length)

    normal_attention = SelfAttention()
    casual_masking_attention = SelfAttention(
        casual_masking=True,
    )

    test_forward(attention=normal_attention, input_tensor=input_tensor)
    test_forward(attention=normal_attention, mask=mask, input_tensor=input_tensor)
    test_forward(attention=casual_masking_attention, input_tensor=input_tensor)

    with pytest.raises(RuntimeError):
        test_forward(
            attention=casual_masking_attention, mask=mask, input_tensor=input_tensor
        )


@pytest.mark.usefixtures("manual_seed")
def test_forward(
    input_tensor: torch.Tensor = None,
    attention: SelfAttention = None,
    mask: torch.Tensor = None,
):
    input_tensor = input_tensor if input_tensor is not None else torch.randn(1, 1, 1)
    attention = attention or SelfAttention()
    output = attention(x=input_tensor, mask=mask)
    assert input_tensor.dtype == output.dtype


@given(
    batch_size=st.integers(min_value=1, max_value=100),
    sequence_length=st.integers(min_value=1, max_value=100),
    embedding_dim=st.integers(min_value=1, max_value=100),
)
@settings(deadline=None)
@pytest.mark.usefixtures("manual_seed")
def test_exotic_architectures(
    batch_size: int,
    sequence_length: int,
    embedding_dim: int,
):
    layers_before = [
        nn.Linear(embedding_dim, embedding_dim * 4),
        nn.Sequential(*[nn.Linear(embedding_dim, embedding_dim) for _ in range(4)]),
        nn.Sequential(
            SelfAttention(
                layer_before=nn.Linear(embedding_dim, embedding_dim * 4),
                q_architecture=nn.Linear(embedding_dim * 4, embedding_dim * 8),
                k_architecture=nn.Linear(embedding_dim * 4, embedding_dim * 8),
                v_architecture=nn.Linear(embedding_dim * 4, embedding_dim * 8),
                layer_after=nn.Linear(embedding_dim * 8, embedding_dim),
            )
        ),
    ]
    q_architectures = [
        nn.Linear(embedding_dim * 4, embedding_dim * 8),
        nn.Linear(embedding_dim, embedding_dim),
        nn.Linear(embedding_dim, embedding_dim),
    ]
    k_architectures = [
        nn.Linear(embedding_dim * 4, embedding_dim * 8),
        nn.Linear(embedding_dim, embedding_dim),
        nn.Linear(embedding_dim, embedding_dim),
    ]
    v_architectures = [
        nn.Linear(embedding_dim * 4, embedding_dim * 8),
        nn.Linear(embedding_dim, embedding_dim),
        nn.Linear(embedding_dim, embedding_dim),
    ]
    layers_after = [
        nn.Linear(embedding_dim * 8, embedding_dim),
        nn.Linear(embedding_dim, embedding_dim),
        nn.Linear(embedding_dim, embedding_dim),
    ]
    input_tensors = [
        torch.randn(batch_size, sequence_length, embedding_dim),
        torch.randn(sequence_length, embedding_dim),
    ]

    for (
        layer_before,
        q_architecture,
        k_architecture,
        v_architecture,
        layer_after,
    ) in zip(
        layers_before,
        q_architectures,
        k_architectures,
        v_architectures,
        layers_after,
    ):
        attention = SelfAttention(
            layer_before=layer_before,
            q_architecture=q_architecture,
            k_architecture=k_architecture,
            v_architecture=v_architecture,
            layer_after=layer_after,
        )
        for input_tensor in input_tensors:
            test_forward(input_tensor=input_tensor, attention=attention)


@pytest.mark.usefixtures("manual_seed")
def test_backpropagation_optimizer_step():
    input_tensor = torch.tensor([1, 2, 3], dtype=torch.float).unsqueeze(0)
    target_tensor = torch.tensor([3, 2, 1], dtype=torch.float).unsqueeze(0)

    attention = SelfAttention(layer_before=nn.Linear(3, 3))
    optimizer = torch.optim.SGD(
        attention.parameters(),
        lr=1,
    )

    output_before = attention(input_tensor)
    print(f"before {output_before}")
    # MSE requires least computation powers (?)
    loss = torch.nn.functional.mse_loss(output_before, target_tensor)
    loss.backward()
    optimizer.step()
    output_after = attention(input_tensor)
    print(f"after {output_after}")
