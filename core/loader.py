from torch.utils.data import Dataset
import numpy as np
from glob import glob
import lightning as L
import torch.utils.data as data
from random import shuffle


class FragmentDataset(Dataset):
    def __init__(self, files_input, files_label):
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
        files_input = sorted(glob(f"{dir_read}/inputs/*.npy"))
        files_label = sorted(glob(f"{dir_read}/labels/*.npy"))
        ind = list(range(len(files_input)))
        shuffle(ind)
        val_ind = ind[:int(len(ind)*0.1)]
        train_ind = ind[int(len(ind)*0.1):]
        self.train = FragmentDataset(
            [files_input[i] for i in train_ind],
            [files_label[i] for i in train_ind]
        )
        self.val = FragmentDataset(
            [files_input[i] for i in val_ind],
            [files_label[i] for i in val_ind]
        )

    def train_dataloader(self, batch_size=1024):
        return data.DataLoader(self.train, batch_size=batch_size, shuffle=True, num_workers=7)

    def val_dataloader(self, batch_size=1024):
        return data.DataLoader(self.val, batch_size=batch_size, shuffle=False, num_workers=7)