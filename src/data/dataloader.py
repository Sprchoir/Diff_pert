from torch.utils.data import DataLoader
from .dataset import DiffDataset

def get_dataloader(configs, split):
    dataset = DiffDataset(configs, split)
    # DataLoader
    batch_size = configs["training"].get("batch_size", 64)
    num_workers = configs["training"].get("num_workers", 4)
    shuffle = True if split== "train" else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)

    return dataloader