from torch.utils.data import Dataset
import torch
import numpy as np

class MiniPileDataset(Dataset):
    def __init__(self, file_path, device = "cuda:0"):
        self.embeddings = np.load(file_path, mmap_mode="r")
        self.device = device

    def __len__(self):
        return self.embeddings.shape[1]

    def __getitem__(self, idx):
        return torch.tensor(self.embeddings[:, idx]).to(self.device).squeeze()
