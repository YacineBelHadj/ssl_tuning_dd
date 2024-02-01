import pytest
from src.model.component.simple_dense_net import SimpleDenseNet
from torch import nn
import torch

def test_simple_dense_net_initialization():
    model = SimpleDenseNet(
        input_dim=10,
        hidden_layers=[5, 5],
        embedding_dim=3,
        output_dim=2,
        dropout=0.1,
        activation='ReLU',
        batch_norm=True,
        bias=True,
        temperature=1.0,
        l1_regularization=0.01,
        l2_regularization=0.01,
    )

    assert isinstance(model, nn.Module), "SimpleDenseNet should be an instance of nn.Module."

def test_simple_dense_net_forward_pass():
    model = SimpleDenseNet(
        input_dim=10,
        hidden_layers=[5, 5],
        embedding_dim=3,
        output_dim=2,
        dropout=0.1,
        activation='ReLU',
        batch_norm=True,
        bias=True,
        temperature=1.0,
        l1_regularization=0.01,
        l2_regularization=0.01,)

    x = torch.randn(1, 10)  # Sample input tensor
    output = model(x)

    assert output.shape == (1, 2), "Output shape should match the defined output dimension."
