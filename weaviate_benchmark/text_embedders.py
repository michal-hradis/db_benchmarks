import torch
import numpy as np
from transformers import AutoModel, AutoTokenizer
from sentence_transformers import SentenceTransformer


class EmbeddingSeznam:
    def __init__(self, model_name: str = "Seznam/retromae-small-cs"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)

    def embed_documents(self, texts: [str]) -> np.ndarray:
        with torch.no_grad():
            inputs = self.tokenizer(
                texts,
                max_length=128,
                truncation=True,
                padding="max_length",
                return_tensors="pt"
            ).to(self.device)
            outputs = self.model(**inputs)
            return outputs.last_hidden_state[:, 0].cpu().detach().numpy()

    def embed_query(self, query: str) -> list[float]:
        return self.embed_documents([query])[0]


class EmbeddingGemma:
    def __init__(self, model_name: str = "BAAI/bge-multilingual-gemma2"):
        print(f"Loading Gemma model {model_name}...")
        self.model = SentenceTransformer(model_name, model_kwargs={"torch_dtype": torch.float16})
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = self.model.to(self.device)
        self.instruction = 'Given a web search query, retrieve relevant passages that answer the query.'
        self.prompt = f'<instruct>{self.instruction}\n<query>'

    def embed_documents(self, texts: [str]) -> list[list[float]]:
        # Compute the query and document embeddings
        document_embeddings = self.model.encode(texts)
        return document_embeddings

    def embed_query(self, query) -> list[float]:
        query_embeddings = self.model.encode([query], prompt=self.prompt)
        return query_embeddings[0]
