from torch.utils.data import Dataset
import numpy as np
import pandas as pd


class MiniPileDataset(Dataset):
    def __init__(self, sentences, embeddings):
        if sentences.endswith(".csv"):
            self.sentences = pd.read_csv(sentences)["sentence"].tolist()
        else:
            self.sentences = sentences

        # embeddings is the source of truth!
        if embeddings.endswith(".npy"):
            self.embeddings = np.load(embeddings, mmap_mode="r")
        else:
            self.embeddings = embeddings

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return self.sentences[idx], self.embeddings[idx]
