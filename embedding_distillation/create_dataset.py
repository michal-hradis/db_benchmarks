import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import os
import argparse
from glob import glob
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO)


def parse_args():
    parser = argparse.ArgumentParser(description="Create a Parquet dataset from JSONL files and numpy arrays containing embeddings.")
    parser.add_argument("--source-dir", type=str, required=True, help="Directory to read source JSONL files from.")
    parser.add_argument("--target-file", type=str, required=True, help="Path to the target Parquet file.")
    parser.add_argument("--embedding-dim", type=int, default=2048, help="Dimension of the embedding.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Get all JSONL files in the source directory
    jsonl_files = glob(os.path.join(args.source_dir, "*.jsonl"))

    if not jsonl_files:
        print(f"No JSONL files found in {args.source_dir}.")
        return

    schema = pa.schema([
        ("text", pa.string()),
        ("embedding", pa.list_(pa.float16(), args.embedding_dim))
    ])

    writer = pq.ParquetWriter(args.target_file, schema)

    buffer_texts, buffer_embs = [], []
    batch_size = 100_000

    for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:

            embeddings = np.load(jsonl_file.replace('.jsonl', '_embeddings.npy'))

            for line in f:
                record = json.loads(line)
                text = record["text"]
                vector_index = record["vector_index"]
                embedding = embeddings[vector_index].astype(np.float16)

                buffer_texts.append(text)
                buffer_embs.append(embedding)  # emb is a float16 array of length  args.embedding_dim

                # flush once we have enough rows
                if len(buffer_texts) >= batch_size:
                    # Shuffle the batch to ensure randomness
                    indices = np.random.permutation(len(buffer_texts))
                    buffer_texts = [buffer_texts[i] for i in indices]
                    buffer_embs = [buffer_embs[i] for i in indices]

                    # build Arrow columns
                    arr_text = pa.array(buffer_texts, pa.string())
                    flat = pa.array(
                        np.stack(buffer_embs).reshape(-1),
                        pa.float16()
                    )
                    arr_emb = pa.FixedSizeListArray.from_arrays(flat, args.embedding_dim)

                    tbl = pa.Table.from_arrays([arr_text, arr_emb], schema.names)
                    writer.write_table(tbl)

                    buffer_texts.clear()
                    buffer_embs.clear()

            # Write any remaining records
            if buffer_texts:
                indices = np.random.permutation(len(buffer_texts))
                buffer_texts = [buffer_texts[i] for i in indices]
                buffer_embs = [buffer_embs[i] for i in indices]

                arr_text = pa.array(buffer_texts, pa.string())
                flat = pa.array(
                    np.stack(buffer_embs).reshape(-1),
                    pa.float16()
                )
                arr_emb = pa.FixedSizeListArray.from_arrays(flat, args.embedding_dim)

                tbl = pa.Table.from_arrays([arr_text, arr_emb], schema.names)
                writer.write_table(tbl)

    writer.close()

    print(f"Parquet dataset created at {args.target_file}")

if __name__ == "__main__":
    main()
