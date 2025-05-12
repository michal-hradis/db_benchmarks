import sys

from scipy.stats import trim1
from transformers import AutoTokenizer, AutoModel
import torch
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from time import time

def main():
    client = WeaviateClient(
        connection_params=ConnectionParams.from_params(
            http_host="localhost", http_port=8080, http_secure=False,
            grpc_host="localhost", grpc_port=50051, grpc_secure=False,
        ))
    client.connect()
    # Check if Weaviate is ready
    if not client.is_ready():
        print("Weaviate is not ready.")
        return
    chunk_col = client.collections.get("TextChunk")

    # 1) Load embedding model
    model_name = "Seznam/retromae-small-cs"
    tokenizer  = AutoTokenizer.from_pretrained(model_name)
    model      = AutoModel.from_pretrained(model_name)
    device     = "cuda" if torch.cuda.is_available() else "cpu"
    model      = model.to(device)

    while True:
        # 2) Read query from terminal
        query = input("Enter your search query: ").strip()
        if not query:
            print("No query given, exiting.")
            return

        # 3) Embed the query
        inputs   = tokenizer(query,
                             max_length=128,
                             truncation=True,
                             padding="max_length",
                             return_tensors="pt").to(device)
        outputs  = model(**inputs)
        q_vector = outputs.last_hidden_state[:,0].cpu().detach().tolist()[0]


        # 5) Run the vector search
        t1 = time()
        result    = chunk_col.query.near_vector(
            near_vector = q_vector,
            limit       = 1024
        )

        # 6) Print the top hits
        print(f"\nTop {len(result.objects)} results for “{query}”. Retrieved in {time() - t1:.2f} seconds:")
        for idx, obj in enumerate(result.objects, start=1):
            text = obj.properties.get("text", "<no text>")
            print(f"  {idx}) {text}")
        print("")

if __name__ == "__main__":
    main()