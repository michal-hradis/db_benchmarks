import sys

from scipy.stats import trim1
from transformers import AutoTokenizer, AutoModel
import torch
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from time import time
import argparse
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description="Query Weaviate for text chunks. Can specify a filter.")
    parser.add_argument("--min-year", type=int, help="Minimum year")
    parser.add_argument("--max-year", type=int, help="Maximum year")
    parser.add_argument("--min-date", type=datetime, help="Minimum date")
    parser.add_argument("--max-date", type=datetime, help="Maximum date")
    parser.add_argument("--genre", type=str, help="Genre to filter by")
    parser.add_argument("--author", type=str, help="Author to filter by")
    parser.add_argument("--title-words", nargs="+", type=str, help="Words to filter by in the title")
    parser.add_argument("--url", type=str)
    parser.add_argument("--limit", type=int, default=80, help="Number of results to return")
    return parser.parse_args()


def main():
    args = parse_args()
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

        filters = []
        if args.min_year:
            filters.append({
                "path": ["document", "yearIssued"],
                "operator": "GreaterThan",
                "valueInt": args.min_year
            })
        if args.max_year:
            filters.append({
                "path": ["document", "yearIssued"],
                "operator": "LessThan",
                "valueInt": args.max_year
            })
        if args.genre:
            filters.append({
                "path": ["document", "genre"],
                "operator": "Equal",
                "valueText": args.genre
            })
        if args.author:
            filters.append({
                "path": ["document", "author"],
                "operator": "Equal",
                "valueText": args.author
            })
        if args.title_words:
            filters.append({
                "path": ["document", "title"],
                "operator": "Contains",
                "valueText": " ".join(args.title_words)
            })

        # 5) Run the vector search
        t1 = time()
        result = client.query.get("TextChunk", ["text"]).with_near_vector({"vector": q_vector}).with_where(
            {"operator": "And", "operands": filters}).with_limit(args.limit).do()

        # 6) Print the top hits
        print(f"\nTop {len(result.objects)} results for “{query}”. Retrieved in {time() - t1:.2f} seconds:")
        for idx, obj in enumerate(result.objects, start=1):
            text = obj.properties.get("text", "<no text>")
            print(f"  {idx}) {text}")
        print("")

if __name__ == "__main__":
    main()