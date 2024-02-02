import torch
from src.component.losses.regularization import l2_regularization

def test_l2_regularization():
    # Create a dummy model
    model = torch.nn.Linear(10, 5)
    
    # Compute the L2 regularization loss
    reg_loss = l2_regularization(model)
    
    # Assert that the regularization loss is a torch.Tensor
    assert isinstance(reg_loss, torch.Tensor)
    
    # Assert that the regularization loss is non-negative
    assert reg_loss >= 0.0
    
    # Assert that the regularization loss is zero when all weights are zero
    model.weight.data.zero_()
    model.bias.data.zero_()
    reg_loss = l2_regularization(model)
    assert reg_loss == 0.0

# Run the test
test_l2_regularization()