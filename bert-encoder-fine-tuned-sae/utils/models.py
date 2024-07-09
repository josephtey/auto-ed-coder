from torch.utils.data import Dataset


class MiniPileDataset(Dataset):
    def __init__(self, sentences, embeddings):
        self.sentences = sentences
        self.embeddings = embeddings

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx], self.embeddings[idx]
