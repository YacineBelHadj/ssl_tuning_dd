import pytest

from src.datamodules.componenet.collate import custom_collate_fn
import torch

def test_custom_collate_fn():
    # create smaple 1 label as dict one hot encoded for each task
    label_dict = {"task1": 2, "task2": 3, 'task3': 3}
    row_1_label = {'task1': torch.tensor([1, 0]), 'task2': torch.tensor([0, 1, 0]), 'task3': torch.tensor([0, 0, 1])}
    row_1_data = torch.tensor([1, 2, 3])
    row_2 = {'task1': torch.tensor([0, 1]), 'task2': torch.tensor([1, 0, 0]), 'task3': torch.tensor([0, 0, 1])}
    row_2_data = torch.tensor([4, 5, 6])
    # expted output 
    collated = (torch.stack((row_1_data,row_2_data)),
                {'task1': torch.tensor([[1, 0],[0, 1]]), 'task2': torch.tensor([[0, 1, 0],[1, 0, 0]]), 'task3': torch.tensor([[0, 0, 1],[0, 0, 1]])})
    collated_res = custom_collate_fn([(row_1_data,row_1_label),(row_2_data,row_2)])
    # check if everything is ok 

    assert torch.equal(collated_res[0], collated[0]), 'data is not well collated'
    assert torch.equal(collated_res[1]['task1'], collated[1]['task1']) ,'task1 label is not well collated'
    assert torch.equal(collated_res[1]['task2'], collated[1]['task2']) ,'task2 label is not well collated'
    assert torch.equal(collated_res[1]['task3'], collated[1]['task3']) ,'task3 label is not well collated'


