import sys

from scipy.stats import trim1
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
from time import time
import argparse
from datetime import datetime
from weaviate.classes.query import Filter, QueryReference
from insert_documents import parse_filename
from text_embedders import EmbeddingSeznam, EmbeddingGemma



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
    chunk_col = client.collections.get("Chunks")
    doc_col = client.collections.get("Books")
    print("Document collection size:", doc_col.aggregate.over_all(total_count=True).total_count)
    print("TextChunk collection size:", chunk_col.aggregate.over_all(total_count=True).total_count)

    # 1) Load embedding model
    embedder = EmbeddingGemma()

    while True:
        # 2) Read query from terminal
        query = input("Enter your search query: ").strip()
        if not query:
            print("No query given, exiting.")
            return

        # 3) Embed the query
        q_vector = embedder.embed_query(query)

        # Build filters based on input arguments
        filters = []

        if args.min_year:
            filters.append(Filter.by_ref(link_on="document").by_property("yearIssued").greater_than(args.min_year))
        if args.max_year:
            filters.append(Filter.by_ref(link_on="document").by_property("yearIssued").less_than(args.max_year))
        if args.genre:
            filters.append(Filter.by_ref(link_on="document").by_property("genre").equal(args.genre))
        if args.author:
            filters.append(Filter.by_ref(link_on="document").by_property("author").equal(args.author))
        if args.title_words:
            filters.append(Filter.by_ref(link_on="document").by_property("title").contains(" ".join(args.title_words)))

        # Combine filters with AND logic
        if filters:
            combined_filter = filters[0]
            for f in filters[1:]:
                combined_filter = combined_filter & f
        else:
            combined_filter = None

        # Perform the vector search with filtering
        t1 = time()
        result = chunk_col.query.hybrid(
            query=query,
            alpha=0.6,
            vector=q_vector.tolist(),
            limit=args.limit,
            filters=combined_filter
        )

        # 6) Print the top hits
        print(f"\nTop {len(result.objects)} results for “{query}”. Retrieved in {time() - t1:.2f} seconds:")
        for idx, obj in enumerate(result.objects, start=1):
            text = obj.properties.get("text", "<no text>").replace("\n", " ")
            print(f"============= {idx}) ==================================")
            print(text)
            print("========================================================")
            print()

if __name__ == "__main__":
    main()