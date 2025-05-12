import torch
from transformers import AutoModel, AutoTokenizer
import weaviate
import argparse
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
import glob
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Weaviate Benchmark Inserts Document embeddings")
    parser.add_argument("--source-dir", type=str,
                        default="/home/ihradis/projects/2024-12-02_Argument/ALL_texts.sample.20k",
                        help="Where to read source text files from")
    return parser.parse_args()


def main():
    args = parse_args()

    # Instantiate the client
    client = WeaviateClient(
        connection_params= ConnectionParams.from_params(
            http_host="localhost", http_port=8080, http_secure=False,
            grpc_host="localhost", grpc_port=50051, grpc_secure=False,
        ))
    client.connect()
    # Check if Weaviate is ready
    if not client.is_ready():
        print("Weaviate is not ready.")
        return

    # Helpers to get the two collections
    doc_col = client.collections.get("Document")
    chunk_col = client.collections.get("TextChunk")

    model_name = "Seznam/retromae-small-cs"  # link name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device)


    with client.batch.fixed_size(batch_size=512, concurrent_requests=4) as batch:
        for file in tqdm(glob.glob(args.source_dir + "/*.txt")[1700:]):
            with open(file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
                lines = [' '.join(line.strip().split(",")[4:]) for line in lines]
                lines = [line for line in lines if len(line) > 0]

                if not lines:
                    continue

                batch_dict = tokenizer(lines, max_length=128, padding=True, truncation=True, return_tensors='pt')
                batch_dict = {key: value.to(device) for key, value in batch_dict.items()}
                with torch.no_grad():
                    outputs = model(**batch_dict)
                embeddings = outputs.last_hidden_state[:, 0].cpu()

                doc_id = doc_col.data.insert(
                    properties={
                        "title": file,
                        "subtitle": "",
                        "partNumber": 1,
                        "partName": "",
                        "dateIssued": None,
                        "author": ["Lidové noviny"],
                        "publisher": "",
                        "language": ["cs"],
                        "description": "",
                        "url": "",
                        "public": True,
                        "documentType": "textfile",
                        "keywords": [],
                        "genre": "",
                        "placeOfPublication": "",
                    }
                )

                for text, emb in zip(lines, embeddings):
                    batch.add_object(
                        collection="TextChunk",
                        properties={
                            "text": text,
                        },
                        vector=emb.numpy().tolist(),
                        references = {
                            "document": doc_id
                        }
                    )

    client.batch.flush()
    print("🎉 Documents inserted into Weaviate.")

if __name__ == "__main__":
    main()
