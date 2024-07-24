from torch.utils.data import Dataset
import numpy as np


class MiniPileDataset(Dataset):
    def __init__(self, sentences, embeddings):
        if sentences.endswith(".npy"):
            self.sentences = np.load(sentences, mmap_mode="r")
        else:
            self.sentences = sentences

        if embeddings.endswith(".npy"):
            self.embeddings = np.load(embeddings, mmap_mode="r")
        else:
            self.embeddings = embeddings

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.embeddings[idx]
