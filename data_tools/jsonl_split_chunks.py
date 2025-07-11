import json
from glob import glob
import argparse
from parse_newton_export import split_record
from tqdm import tqdm
import os

def parse_args():
    parser = argparse.ArgumentParser(description="Split jsonl records into chunks based on character count.")
    parser.add_argument('-i', '--input-dir', required=True, type=str,
                        help="Path to the input directory containing jsonl files with articles.")
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help="Path to the output directory to save extracted article chunks.")
    parser.add_argument('--target-chunk-chars', type=int, default=1024, help="Target number of characters per chunk.")
    return parser.parse_args()


def main():
    args = parse_args()

    # Get all jsonl files in the input directory
    jsonl_files = glob(f"{args.input_dir}/*.jsonl")
    if not jsonl_files:
        print(f"No jsonl files found in {args.input_dir}.")
        return

    os.makedirs(args.output_dir, exist_ok=True)

    for jsonl_file in tqdm(jsonl_files, desc="Processing files"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]

        chunks = []
        for record in records:
            chunks.extend(split_record(record, target_chunk_chars=args.target_chunk_chars))

        if args.output_dir:
            output_file = os.path.join(args.output_dir, os.path.basename(jsonl_file))
            with open(output_file, 'w', encoding='utf-8') as out_f:
                for chunk in chunks:
                    out_f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        else:
            for chunk in chunks:
                print(json.dumps(chunk, ensure_ascii=False))  # Print to stdout if no output dir is specified


if __name__ == "__main__":
    main()


