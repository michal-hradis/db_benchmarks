import argparse
import json
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def parse_args():
    parser = argparse.ArgumentParser(
        description="Merges a jsonl file with text chunks with a jsonl file with generated queries - from OpenAI API. "
                    "The chunk file contains key 'id' which is used to match with 'custom_id' in the query file. "
                    "Include all fields from both files in the output.")
    parser.add_argument("--chunk-file", required=True, help="Path to the input jsonl file with text chunks.")
    parser.add_argument("--query-file", required=True, help="Path to the input jsonl file with generated queries.")
    parser.add_argument("--output-file", required=True, help="Path to save the merged output jsonl file.")
    return parser.parse_args()

def load_chunks(chunk_file):
    """Load chunks from JSONL file and index by 'id' field."""
    chunks = {}
    with open(chunk_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                record = json.loads(line)
                if 'id' not in record:
                    logging.warning(f"Line {line_num}: Missing 'id' field, skipping")
                    continue
                chunks[record['id']] = record
    logging.info(f"Loaded {len(chunks)} chunks from {chunk_file}")
    return chunks

def load_queries(query_file):
    """Load queries from OpenAI API JSONL file and index by 'custom_id'."""
    queries = {}
    with open(query_file, 'r', encoding='utf-8') as f:
        for line_num, line in enumerate(f, 1):
            if line.strip():
                record = json.loads(line)
                if 'custom_id' not in record:
                    logging.warning(f"Line {line_num}: Missing 'custom_id' field, skipping")
                    continue

                # Extract the OpenAI API response content
                try:
                    content = record['response']['body']['choices'][0]['message']['content']
                    # Decode the JSON content
                    parsed_content = json.loads(content)
                    # Store only the parsed content keys with custom_id as key
                    queries[record['custom_id']] = parsed_content
                except (KeyError, IndexError, json.JSONDecodeError) as e:
                    logging.warning(f"Line {line_num}: Error extracting/parsing content: {e}, skipping")
                    continue

    logging.info(f"Loaded {len(queries)} queries from {query_file}")
    return queries

def merge_data(chunks, queries):
    """Merge chunks and queries based on id/custom_id match."""
    merged = []
    matched = 0
    unmatched_chunks = 0

    for chunk_id, chunk_data in chunks.items():
        if chunk_id in queries:
            # Merge all fields from both records
            merged_record = {**chunk_data, **queries[chunk_id]}
            merged.append(merged_record)
            matched += 1
        else:
            unmatched_chunks += 1

    logging.info(f"Matched {matched} records")
    if unmatched_chunks > 0:
        logging.warning(f"{unmatched_chunks} chunks had no matching query")

    unmatched_queries = len(queries) - matched
    if unmatched_queries > 0:
        logging.warning(f"{unmatched_queries} queries had no matching chunk")

    return merged

def save_output(merged_data, output_file):
    """Save merged data to JSONL file."""
    Path(output_file).parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, 'w', encoding='utf-8') as f:
        for record in merged_data:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')

    logging.info(f"Saved {len(merged_data)} merged records to {output_file}")

def main():
    args = parse_args()

    chunks = load_chunks(args.chunk_file)
    queries = load_queries(args.query_file)
    merged = merge_data(chunks, queries)
    save_output(merged, args.output_file)

    logging.info("Merge completed successfully")

if __name__ == "__main__":
    main()
