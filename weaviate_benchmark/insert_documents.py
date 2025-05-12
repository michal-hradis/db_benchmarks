import torch
from transformers import AutoModel, AutoTokenizer
import weaviate
import argparse
from weaviate import WeaviateClient
from weaviate.connect import ConnectionParams
import glob
from tqdm import tqdm
import re

def parse_args():
    parser = argparse.ArgumentParser(description="Weaviate Benchmark Inserts Document embeddings")
    parser.add_argument("--source-dir", type=str,
                        default="/home/ihradis/projects/2024-12-02_Argument/ALL_texts.sample.20k",
                        help="Where to read source text files from")
    return parser.parse_args()


def parse_filename(filename: str) -> dict:
    # Define the regex pattern
    pattern = r"(\d{4}-\d{2}-\d{2})_(.+?)\.(A\d{6}_\d{6})_(.+?)_(\w+)\.txt"
    match = re.match(pattern, filename)

    if match:
        date, title, file_id, rubrik, last_part = match.groups()
        return {
            "date": date,
            'year': date.split("-")[0],
            "title": title.replace("-", " "),  # Replace hyphens with spaces for readability
            "id": file_id,
            "rubrik": rubrik,
            "last_part": last_part
        }
    else:
        return None


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
    print("Document collection size:", doc_col.aggregate.over_all(total_count=True).total_count)
    print("TextChunk collection size:", chunk_col.aggregate.over_all(total_count=True).total_count)

    model_name = "Seznam/retromae-small-cs"  # link name
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = AutoModel.from_pretrained(model_name).to(device)

    nn_batch_size = 16

    with client.batch.fixed_size(batch_size=512, concurrent_requests=4) as batch:
        for file in tqdm(glob.glob(args.source_dir + "/*.txt")[1700:]):
            with open(file, 'r', encoding='utf-8') as f:
                parsed_filename = parse_filename(file)
                #print(f"Parsed filename: {parsed_filename}")
                if parsed_filename is None:
                    print(f"Filename {file} does not match the expected format.")
                    continue

                lines = f.readlines()
                lines = [' '.join(line.strip().split(",")[4:]) for line in lines]
                lines = [line for line in lines if len(line) > 0]

                if not lines:
                    continue

                doc_id = doc_col.data.insert(
                    properties={
                        "title": parsed_filename["title"],
                        "subtitle": "",
                        "partNumber": 1,
                        "partName": "",
                        "dateIssued": parsed_filename["year"],
                        "author": parsed_filename["last_part"],
                        "publisher": "",
                        "language": ["cs"],
                        "description": "",
                        "url": parsed_filename["id"],
                        "public": True,
                        "documentType": "textfile",
                        "keywords": [],
                        "genre": parsed_filename["rubrik"],
                        "placeOfPublication": "",
                    }
                )

                while len(lines) > 0:
                    lines_batch = lines[:nn_batch_size]
                    batch_dict = tokenizer(lines_batch, max_length=128, padding=True, truncation=True, return_tensors='pt')
                    batch_dict = {key: value.to(device) for key, value in batch_dict.items()}
                    with torch.no_grad():
                        outputs = model(**batch_dict)
                    embeddings = outputs.last_hidden_state[:, 0].cpu()

                    for text, emb in zip(lines_batch, embeddings):
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
