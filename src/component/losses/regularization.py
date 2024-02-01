import torch
#let's define a function that computes the L2 regularization loss for a torch.nn.Module

def l2_regularization(model:torch.nn.Module)->torch.Tensor:
    """
    Computes L2 regularization loss for a torch.nn.Module
    """
    reg_loss = 0.
    for param in model.parameters():
        #only weight using hasattr
        if hasattr(param,"weight"):
            reg_loss += torch.norm(param)
    return reg_loss

def l1_regularization(model:torch.nn.Module)->torch.Tensor:
    """
    Computes L1 regularization loss for a torch.nn.Module
    """
    reg_loss = 0.
    for param in model.parameters():
        #only weight using hasattr
        if hasattr(param,"weight"):
            reg_loss += torch.norm(param,p=1)
    return reg_loss
