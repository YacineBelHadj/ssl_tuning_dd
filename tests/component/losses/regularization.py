import pytest
import torch
from src.component.losses.regularization import l1_regularization, l2_regularization  # Adjust import path as necessary
from torch import nn

class TestModel(nn.Module):
    def __init__(self):
        super(TestModel, self).__init__()
        self.linear = nn.Linear(2, 2, bias=True)
        self.linear.weight.data = torch.tensor([[1.0, -2.0], [3.0, -4.0]])
        self.linear2 = nn.Linear(2, 2, bias=False)
        self.linear2.weight.data = torch.tensor([[0.0, -2.0], [3.0, -0.0]])
        self.network = nn.Sequential(self.linear, self.linear2)

    def forward(self, x):
        return self.network(x)
    
def test_l1_regularization():
    model = TestModel()

    # Manually calculate expected L1 regularization value
    expected_l1 = torch.abs(model.linear.weight.data).sum() + torch.abs(model.linear2.weight.data).sum()

    # Compute L1 regularization using the function
    computed_l1 = l1_regularization(model)

    assert torch.isclose(expected_l1, computed_l1), "Computed L1 regularization should match the expected value."

def test_l2_regularization():
    model = TestModel()

    # Manually calculate expected L2 regularization value
    expected_l2 = torch.sqrt(torch.sum(torch.pow(model.linear.weight.data, 2)) + torch.sum(torch.pow(model.linear2.weight.data, 2)))

    # Compute L2 regularization using the function
    computed_l2 = l2_regularization(model)

    assert torch.isclose(expected_l2, computed_l2), "Computed L2 regularization should match the expected value."
