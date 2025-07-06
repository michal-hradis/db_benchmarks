import json
import os
import argparse
import glob
from tqdm import tqdm
import numpy as np
from weaviate_benchmark.text_embedders import EmbeddingGemma

def parse_args():
    parser = argparse.ArgumentParser(description="Read text cunks from jsonl files, extract text exmbeddings and store them again into jsonl files.")
    parser.add_argument("--source-dir", type=str, required=True, help="Where to read source jsonl files from.")
    parser.add_argument("--target-dir", type=str, required=True, help="Where to write target jsonl files to.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for processing.")
    parser.add_argument("--json-vectors", action='store_true', help="If set, store vectors directly in the json lines file. By default, vectors are stored as numpy arrays.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Get all jsonl files in the source directory
    jsonl_files = glob.glob(os.path.join(args.source_dir, "*.jsonl"))

    if not jsonl_files:
        print(f"No jsonl files found in {args.source_dir}.")
        return

    if not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir)

    embedder = EmbeddingGemma()

    # shuffle the files to ensure random processing order
    np.random.shuffle(jsonl_files)

    # Process each file
    for jsonl_file in tqdm(jsonl_files, position=0, desc="Processing files"):
        target_file = os.path.join(args.target_dir, os.path.basename(jsonl_file))
        if os.path.exists(target_file):
            tqdm.write(f"Target file {target_file} already exists, skipping.")
            continue

        os.open(target_file, 'a').close() # Claim this file for when running in parallel

        with open(jsonl_file, 'r') as f:
            lines = f.readlines()

        chunks = []
        # Process in batches
        embedding_tensor = np.zeros((len(lines), embedder.model.get_sentence_embedding_dimension()), dtype=np.float16)
        for i in tqdm(range(0, len(lines), args.batch_size), leave=False, position=1, desc="Processing chunks"):
            batch_lines = lines[i:i + args.batch_size]
            batch_lines = [json.loads(line) for line in batch_lines]
            texts = [chunk["text"] for chunk in batch_lines]
            embeddings = embedder.embed_documents(texts)
            for j, line in enumerate(batch_lines):
                if args.json_vectors:
                    line["vector"] = embeddings[j].tolist()
                else:
                    embedding_tensor[i + j, ...] = embeddings[j]
                    line["vector_index"] = i + j
                chunks.append(line)

        # Write the embedding tensor to a file if not using JSON vectors
        if not args.json_vectors:
            embedding_file = os.path.join(args.target_dir, os.path.basename(jsonl_file).replace('.jsonl', '_embeddings.npy'))
            np.save(embedding_file, embedding_tensor)

        # Write to target file
        with open(target_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")


if __name__ == "__main__":
    main()


