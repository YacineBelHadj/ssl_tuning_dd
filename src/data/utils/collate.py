import torch
from torch.utils.data import DataLoader
def collate_fn_first_column(batch):

    first_column = [item[0] for item in batch]
    return torch.stack(first_column)

# take a dataloader and make a dataloader with the first column of the batch
# as the only element  

def get_first_column_dataloader(original_dataloader):
    # Create a new DataLoader with the original dataset and settings, but with a custom collate function
    new_dataloader = DataLoader(
        dataset=original_dataloader.dataset, 
        batch_size=original_dataloader.batch_size,
        shuffle=original_dataloader.shuffle,
        sampler=original_dataloader.sampler,
        batch_sampler=original_dataloader.batch_sampler,
        num_workers=original_dataloader.num_workers,
        collate_fn=collate_fn_first_column,
        pin_memory=original_dataloader.pin_memory,
        drop_last=original_dataloader.drop_last,
        timeout=original_dataloader.timeout,
        worker_init_fn=original_dataloader.worker_init_fn,
        prefetch_factor=getattr(original_dataloader, 'prefetch_factor', 2),  # Defaulting to 2 if not present
        persistent_workers=getattr(original_dataloader, 'persistent_workers', False)  # Defaulting to False if not present
    )
    
    return new_dataloader
