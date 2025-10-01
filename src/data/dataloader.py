from torch.utils.data import DataLoader
from .dataset import DiffDataset
from .dataset_d import DecoderDataset
from ..utils.utils import get_data_decoder

def get_dataloader(configs, split):
    dataset = DiffDataset(configs, split)
    # DataLoader
    batch_size = configs["training"].get("batch_size", 64)
    num_workers = configs["training"].get("num_workers", 4)
    shuffle = True if split== "train" else False
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, pin_memory=True, drop_last=True)

    return dataloader

def get_dataloader_decoder(configs, split):
    batch_size = configs["training"].get("batch_size", 64)
    num_workers = configs["training"].get("num_workers", 4)
    if split == "train":
        # Get data from diffusion model
        gen_times = configs["decoder"]["gen_times"]
        Y_low_gen, Y_full, Y_low_real = get_data_decoder(gen_times, configs, split="train")
        train_dataset = DecoderDataset(configs, Y_low_gen, Y_full, Y_low_real, split)
        val_dataset = DecoderDataset(configs, Y_low_gen, Y_full, Y_low_real, "val")
        # DataLoader
        train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True, drop_last=True)
        val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=True)
        return train_dataloader, val_dataloader
    else:
        Y_low_gen, Y_full = get_data_decoder(configs=configs, gen_times=1, split="test")
        test_dataset = DecoderDataset(configs, Y_low_gen, Y_full, split="test")
        test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, drop_last=False)
        return test_dataloader