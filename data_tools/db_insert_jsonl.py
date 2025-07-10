import json
import os
import argparse
import glob
import random
from tqdm import tqdm
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
import weaviate.classes.config as wvc

def parse_args():
    parser = argparse.ArgumentParser(description="Create DB schema, red document json and push documents into db, read text cunks from jsonl files and push to DB.")
    parser.add_argument("--source-dir", type=str, required=True, help="Where to read source .jsonl and .json files from.")
    parser.add_argument("--delete-old", action="store_true", help="Delete old collections before creating new ones.")
    parser.add_argument("--document-collection", type=str, default="Documents", help="Name of the document collection.")
    parser.add_argument("--chunk-collection", type=str, default="Chunks", help="Name of the text chunk collection.")
    parser.add_argument("--owner-name", type=str, required=False, help="Collection name.")
    args = parser.parse_args()
    return args

def create_document_collection(client: WeaviateClient, keys: list[str], collection_name: str) -> None:
    client.collections.create(
        name=collection_name,
        properties=[
            wvc.Property(name="library", data_type=wvc.DataType.TEXT),
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

def create_schema(client: WeaviateClient, delete_old: bool) -> None:
    if delete_old:
        # Optional: clean slate
        for cls in ("Books", "Chunks"):
            try:
                client.collections.delete(cls)
            except Exception as e:
                pass
    else:
        return

    # 4) Create Document (with a reverse reference slot for collections)


    # 5) Create TextChunk (with reference back to its Document)
    client.collections.create(
        name="Chunks",
        vector_index_config=wvc.Configure.VectorIndex.hnsw(),
        properties=[
            wvc.Property(name="title", data_type=wvc.DataType.TEXT),
            wvc.Property(name="text", data_type=wvc.DataType.TEXT),
            wvc.Property(name="start_page_id", data_type=wvc.DataType.TEXT),
            wvc.Property(name="from_page", data_type=wvc.DataType.INT),
            wvc.Property(name="to_page", data_type=wvc.DataType.INT),
            wvc.Property(name="end_paragraph", data_type=wvc.DataType.BOOL),
            wvc.Property(name="language", data_type=wvc.DataType.TEXT),
            wvc.Property(name="ner_A", data_type=wvc.DataType.TEXT_ARRAY),  # Address entities
            wvc.Property(name="ner_G", data_type=wvc.DataType.TEXT_ARRAY),  # Geographical entities
            wvc.Property(name="ner_I", data_type=wvc.DataType.TEXT_ARRAY),  # Institution entities
            wvc.Property(name="ner_M", data_type=wvc.DataType.TEXT_ARRAY),  # Media entities
            wvc.Property(name="ner_N", data_type=wvc.DataType.TEXT_ARRAY),  # ???
            wvc.Property(name="ner_O", data_type=wvc.DataType.TEXT_ARRAY),  # Cultural artifacts
            wvc.Property(name="ner_P", data_type=wvc.DataType.TEXT_ARRAY),  # Person entities
            wvc.Property(name="ner_T", data_type=wvc.DataType.TEXT_ARRAY),  # Temporal entities
        ],
        references=[
            wvc.ReferenceProperty(
                name="document",
                target_collection="Books",
                cardinality="1"
            )
        ]
    )

def insert_documents(client: WeaviateClient, source_dir: str) -> None:
    doc_col = client.collections.get("Books")
    json_files = glob.glob(os.path.join(source_dir, "*.doc.json"))
    for json_file in tqdm(json_files):
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
            uuid = data["id"]
            del data["id"]
            data["yearIssued"] = int(data["dateIssued"].split('-')[0])
            data["dateIssued"] = data["dateIssued"].replace(' ', 'T') + "+00:00"
            doc_col.data.insert(
                uuid=uuid,
                properties=data,
            )

def insert_chunks(client: WeaviateClient, source_dir: str) -> None:
    jsonl_files = glob.glob(os.path.join(source_dir, "*.chunk.jsonl"))
    with client.batch.fixed_size(batch_size=32, concurrent_requests=2) as batch:
        for jsonl_file in tqdm(jsonl_files):
            tqdm.write(jsonl_file)
            with open(jsonl_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    data = json.loads(line)
                    if 'id' not in data:
                        continue
                    uuid = data["id"]
                    del data["id"]

                    document_uuid = data["document"]
                    del data["document"]
                    vector = data["vector"]
                    del data["vector"]


                    batch.add_object(
                        uuid=uuid,
                        collection="Chunks",
                        properties=data,
                        vector=vector,
                        references={
                            "document": document_uuid
                        }
                    )

def extract_attributes_from_jsonl(source_dir: str, max_files=500) -> list[str]:
    files = glob.glob(os.path.join(source_dir, "*.jsonl"))
    if len(files) > max_files:
        files = random.sample(files, max_files)
    attributes = set()
    for file_name in files:
        with open(file_name, 'r', encoding='utf-8') as f:
            lines = [l.strip() for l in f.readlines()]
            lines = [line for line in lines if line]  # Remove empty lines
            for line in lines:
                data = json.loads(line)
                attributes.update(data.keys())
    return sorted(attributes)


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

    keys = extract_attributes_from_jsonl(args.source_dir)

    chunk_keys = ['id', 'text', 'start_page_id', 'from_page', 'to_page', 'order', 'end_paragraph', 'document',
                  'language', 'ner_A', 'ner_G', 'ner_I', 'ner_M', 'ner_N', 'ner_O', 'ner_P', 'ner_T']

    document_keys = [k for k in keys if k not in chunk_keys]

    # Create schema
    create_schema(client, args.delete_old)

    # Insert documents and chunks
    insert_documents(client, args.source_dir)
    insert_chunks(client, args.source_dir)

    client.close()

if __name__ == "__main__":
    main()
