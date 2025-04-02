from torch.utils.data import Dataset
import numpy as np
import torch
from glob import glob
import lightning as L
import torch.utils.data as data


class FragmentDataset(Dataset):
    def __init__(self, dir_read):
        files_input = sorted(glob(f"{dir_read}/inputs/*.npy"))
        files_label = sorted(glob(f"{dir_read}/labels/*.npy"))
        inputs, labels = [], []
        for file_i, file_l in zip(files_input, files_label):
            inputs.append(np.load(file_i).astype(np.float32))
            labels.append(np.load(file_l).astype(np.float32))
        self.inputs = np.concatenate(inputs, axis=0)
        self.labels = np.concatenate(labels, axis=0)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.labels[idx]


class FragmentDataModule(L.LightningDataModule):
    def setup(self, dir_read):
        dataset = FragmentDataset(dir_read)
        n = len(dataset)
        t = int(n * 0.9)
        self.train, self.val = data.random_split(
            dataset, [0.9,0.1], generator=torch.Generator().manual_seed(42)
        )

    def train_dataloader(self, batch_size=1024):
        return data.DataLoader(self.train, batch_size=batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self, batch_size=1024):
        return data.DataLoader(self.val, batch_size=batch_size, shuffle=False, num_workers=7)