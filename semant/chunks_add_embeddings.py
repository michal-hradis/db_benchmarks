import json
import os
import argparse
import glob
from tqdm import tqdm
from weaviate_benchmark.text_embedders import EmbeddingGemma

def parse_args():
    parser = argparse.ArgumentParser(description="Read text cunks from jsonl files, extract text exmbeddings and store them again into jsonl files.")
    parser.add_argument("--source-dir", type=str, required=True, help="Where to read source jsonl files from.")
    parser.add_argument("--target-dir", type=str, required=True, help="Where to write target jsonl files to.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for processing.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Get all jsonl files in the source directory
    jsonl_files = glob.glob(os.path.join(args.source_dir, "*.jsonl"))

    if not jsonl_files:
        print(f"No jsonl files found in {args.source_dir}.")
        return

    embedder = EmbeddingGemma()

    # Process each file
    for jsonl_file in tqdm(jsonl_files, position=0, desc="Processing files"):
        target_file = os.path.join(args.target_dir, os.path.basename(jsonl_file))
        if os.path.exists(target_file):
            tqdm.write(f"Target file {target_file} already exists, skipping.")
            continue

        with open(jsonl_file, 'r') as f:
            lines = f.readlines()

        chunks = []
        # Process in batches
        for i in tqdm(range(0, len(lines), args.batch_size), leave=False, position=1, desc="Processing chunks"):
            batch_lines = lines[i:i + args.batch_size]
            batch_lines = [json.loads(line) for line in batch_lines]
            texts = [chunk["text"] for chunk in batch_lines]
            embeddings = embedder.embed_documents(texts)
            for j, line in enumerate(batch_lines):
                line["vector"] = embeddings[j].tolist()
                chunks.append(line)

        # Write to target file
        with open(target_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")


if __name__ == "__main__":
    main()


