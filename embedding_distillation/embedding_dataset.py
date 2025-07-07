from torch.utils.data import Dataset

class EmbeddingDataset(Dataset):
    def __init__(self, texts: list, embedder):
        """
        Initializes the dataset with texts and an embedder.

        :param texts: List of texts to be embedded.
        :param embedder: An instance of an embedding model.
        """
        self.texts = texts
        self.embedder = embedder

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        embedding = self.embedder.embed_query(text)
        return text, embedding