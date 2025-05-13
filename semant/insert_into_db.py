import json
import os
import argparse
import glob
from tqdm import tqdm
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
import weaviate.classes.config as wvc

def parse_args():
    parser = argparse.ArgumentParser(description="Create DB schema, red document json and push documents into db, read text cunks from jsonl files and push to DB.")
    parser.add_argument("--source-dir", type=str, required=True, help="Where to read source .jsonl and .json files from.")
    parser.add_argument("--delete-old", action="store_true", help="Delete old collections before creating new ones.")
    args = parser.parse_args()
    return args


def create_schema(client: WeaviateClient, delete_old: bool) -> None:
    if delete_old:
        # Optional: clean slate
        for cls in ("Semant Books", "Chunks"):
            try:
                client.collections.delete(cls)
            except Exception as e:
                pass

    # 4) Create Document (with a reverse reference slot for collections)
    client.collections.create(
        name="Semant Books",
        vector_index_config=wvc.Configure.VectorIndex.hnsw(),
        properties=[
            wvc.Property(name="title", data_type=wvc.DataType.TEXT),
            wvc.Property(name="subTitle", data_type=wvc.DataType.TEXT),
            wvc.Property(name="partNumber", data_type=wvc.DataType.INT),
            wvc.Property(name="partName", data_type=wvc.DataType.TEXT),
            wvc.Property(name="yearIssued", data_type=wvc.DataType.INT),
            wvc.Property(name="dateIssued", data_type=wvc.DataType.DATE),
            wvc.Property(name="author", data_type=wvc.DataType.TEXT),
            wvc.Property(name="publisher", data_type=wvc.DataType.TEXT),
            wvc.Property(name="language", data_type=wvc.DataType.TEXT_ARRAY),
            wvc.Property(name="description", data_type=wvc.DataType.TEXT),
            wvc.Property(name="url", data_type=wvc.DataType.TEXT),
            wvc.Property(name="public", data_type=wvc.DataType.BOOL),
            wvc.Property(name="documentType", data_type=wvc.DataType.TEXT),
            wvc.Property(name="keywords", data_type=wvc.DataType.TEXT_ARRAY),
            wvc.Property(name="genre", data_type=wvc.DataType.TEXT),
            wvc.Property(name="placeTerm", data_type=wvc.DataType.TEXT),
        ]
    )

    # 5) Create TextChunk (with reference back to its Document)
    client.collections.create(
        name="Chunks",
        vector_index_config=wvc.Configure.VectorIndex.hnsw(),
        properties=[
            wvc.Property(name="text", data_type=wvc.DataType.TEXT),
            wvc.Property(name="start_page_id", data_type=wvc.DataType.TEXT),
            wvc.Property(name="from_page", data_type=wvc.DataType.TEXT),
            wvc.Property(name="to_page", data_type=wvc.DataType.TEXT),
        ],
        references=[
            wvc.ReferenceProperty(
                name="document",
                target_collection="Semant Books",
                cardinality="*"
            )
        ]
    )

def insert_documents(client: WeaviateClient, source_dir: str) -> None:
    doc_col = client.collections.get("Semant Books")
    json_files = glob.glob(os.path.join(source_dir, "*.doc.json"))
    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            uuid = data["id"]
            del data["id"]
            doc_col.data.insert(
                properties=data,
                uuid=uuid
            )

def insert_chunks(client: WeaviateClient, source_dir: str) -> None:
    chunk_col = client.collections.get("Chunks")
    jsonl_files = glob.glob(os.path.join(source_dir, "*.chunk.jsonl"))
    with client.batch.fixed_size(batch_size=32, concurrent_requests=2) as batch:
        for jsonl_file in tqdm(jsonl_files):
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    data = json.loads(line)
                    uuid = data["id"]
                    del data["id"]
                    document_uuid = data["document"]
                    del data["document"]
                    vector = data["vector"]

                    batch.add_object(
                        collection="Chunks",
                        uuid=uuid,
                        properties=data,
                        vector=vector,
                        references={
                            "document": document_uuid
                        }
                    )


def main():
    args = parse_args()

    # Connect to Weaviate
    client = WeaviateClient(
        connection_params= ConnectionParams.from_params(
            http_host="localhost", http_port=8080, http_secure=False,
            grpc_host="localhost", grpc_port=50051, grpc_secure=False,
        ))
    client.connect()
    if not client.is_ready():
        print("Weaviate is not ready.")
        return

    # Create schema
    create_schema(client, args.delete_old)

    # Insert documents and chunks
    insert_documents(client, args.source_dir)
    insert_chunks(client, args.source_dir)