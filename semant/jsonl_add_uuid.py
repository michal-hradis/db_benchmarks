import argparse
import json
import os
from tqdm import tqdm
from glob import glob
from uuid import uuid4


def parse_args():
    parser = argparse.ArgumentParser(description="Add key 'id' (uuid4) to records in JSONL files. If key 'id' already exists, it will be skipped. Rewrite the files.")
    parser.add_argument("--dir", type=str, required=True, help="Directory containing the .jsonl files.")
    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    # Get all jsonl files in the directory
    jsonl_files = glob(os.path.join(args.dir, "*.jsonl"))

    if not jsonl_files:
        print(f"No jsonl files found in {args.dir}.")
        return

    # Process each file
    for jsonl_file in tqdm(jsonl_files, position=0, desc="Processing files"):
        with open(jsonl_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        updated = False
        # Add 'id' key to each line if it doesn't exist
        updated_lines = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            if '"id": "' not in line:
                # just replace '}' with ', "id": "uuid4"}'
                line = line.rstrip('}') + f', "id": "{str(uuid4())}"}}'
                updated = True
            updated_lines.append(line)

        # Write to the same file
        if updated:
            with open(jsonl_file, 'w', encoding='utf-8') as f:
                f.write("\n".join(updated_lines))


if __name__ == "__main__":
    main()
