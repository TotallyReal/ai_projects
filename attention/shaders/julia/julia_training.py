from torch.utils.data import Dataset
import numpy as np
import torch

class MyImageDataset(Dataset):
    def __init__(self, data_path):
        data = torch.load(data_path)
        self.images = data["images"].unsqueeze(1)
        self.labels = data["labels"]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]


MyImageDataset('data/dataset.pt')