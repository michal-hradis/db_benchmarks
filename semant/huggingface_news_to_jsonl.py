import json
from datasets import load_dataset
from uuid import uuid4
from tqdm import tqdm
import os


def split_record(record, target_chunk_chars=1024):
    """
    Splits a record into smaller chunks based on character limits.
    """

    text = record['content']
    text = text.split('. ')
    chunks = []
    current_chunk = ""
    for sentence in text:
        if len(current_chunk) + len(sentence) + 1 > target_chunk_chars:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + '. '
        else:
            current_chunk += sentence + '. '

    if current_chunk:
        chunks.append(current_chunk.strip())

    del record['content']  # Remove the original content field
    output_records = []
    for i, chunk in enumerate(chunks):
        new_record = record.copy()
        new_record['id'] = str(uuid4())
        new_record['document_index'] = i
        new_record['text'] = chunk.strip()
        output_records.append(new_record)
    return output_records


def main():
    # Stream-load all splits of the dataset
    ds_dict = load_dataset("hynky/czech_news_dataset_v2", split=None, streaming=True)
    output_path = "/mnt/zfs1/data2/2025-05-11_db_benchmarks/czech_news_dataset_v2/chunks"
    if not os.path.exists(output_path):
        os.makedirs(output_path, exist_ok=True)

    for split_name, split_ds in tqdm(ds_dict.items(), desc="Processing splits", position=0):
        print(f"Writing split '{split_name}' to {output_path}…")

        for record in tqdm(split_ds, desc="Processing records", position=1):
            record['document'] = str(uuid4())  # Assign a unique document ID
            record['split'] = split_name  # Add the split name to the record
            output_records = split_record(record)
            output_file_name = f"{output_path}/{record['document']}.jsonl"
            with open(output_file_name, "w", encoding="utf-8") as out_file:
                for out_rec in output_records:
                    out_file.write(json.dumps(out_rec, ensure_ascii=False, default=str) + "\n")


if __name__ == "__main__":
    main()