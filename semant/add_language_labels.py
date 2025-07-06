import json
import os
import argparse
import glob
from tqdm import tqdm
import fasttext
from huggingface_hub import hf_hub_download


def parse_args():
    parser = argparse.ArgumentParser(description="Read text cunks from jsonl files, extract text exmbeddings and store them again into jsonl files.")
    parser.add_argument("--source-dir", type=str, required=True, help="Where to read source jsonl files from.")
    parser.add_argument("--target-dir", type=str, required=True, help="Where to write target jsonl files to.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for processing.")
    args = parser.parse_args()
    return args


class FasttextLanguageIdentification:
    def __init__(self):
        self.model_path = hf_hub_download(repo_id="facebook/fasttext-language-identification", filename="model.bin")
        self.model = fasttext.load_model(self.model_path)
        self.conf_threshold = 0.6

    def predict(self, text):
        labels, probs = self.model.predict(text)
        prob = probs[0]
        if prob < self.conf_threshold:
            return None
        language = labels[0].replace("__label__", "").split("_")[0]
        return language


def main():
    args = parse_args()

    # Get all jsonl files in the source directory
    jsonl_files = glob.glob(os.path.join(args.source_dir, "*.jsonl"))

    if not jsonl_files:
        print(f"No jsonl files found in {args.source_dir}.")
        return

    model = FasttextLanguageIdentification()

    if args.target_dir and not os.path.exists(args.target_dir):
        os.makedirs(args.target_dir, exist_ok=True)

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
        for i in tqdm(range(0, len(lines)), leave=False, position=1, desc="Processing chunks"):
            batch_line = json.loads(lines[i])
            text = batch_line["text"]
            text = ' '.join(text.split('\n'))
            batch_line["language"] = model.predict(text)
            chunks.append(batch_line)

        # Write to target file
        with open(target_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk) + "\n")


if __name__ == "__main__":
    main()


