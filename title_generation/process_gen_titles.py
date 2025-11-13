import json
import argparse
import logging
from typing import Any, Dict, List


def parse_args():
    parser = argparse.ArgumentParser(
        description="Merge generated title (and related metadata) results into chunk JSONL file.\n"
                    "Inputs: chunk-file (JSONL of chunk dicts with 'text'), batch-file (JSONL of OpenAI batch requests), "
                    "result-file (JSONL of OpenAI batch responses). Merges by custom_id, locates the chunk via the text "
                    "embedded in messages[1].content (template: 'query: {query}\\nretrieved text chunk: {chunk}\\n'). "
                    "Parses response.body.choices[0].message.content as JSON; if it is a JSON object its keys/values are merged into the entry. "
                    "If not valid JSON, falls back to extracting a single 'generated_title' from raw text (prefix 'title:' removed). "
                    "Primary chunk matching is by the literal chunk text; if that fails a fallback uses the chunk id parsed from custom_id (pattern '<chunk_id>_q<idx>_p<idx>'). "
                    "Each appended entry contains id, model, query, prompt plus decoded keys. Updated chunks overwrite the original chunk-file."  # noqa: E501
    )
    parser.add_argument(
        "--chunk-file", type=str, required=True,
        help="JSONL file containing text chunks (each line a JSON object with at least 'text')."
    )
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSONL file to save updated chunks with generated titles.")
    parser.add_argument(
        "--batch-file", type=str, required=True,
        help="JSONL file with queries prepared for OpenAI API (has custom_id, body.messages, body.model)."
    )
    parser.add_argument(
        "--result-file", type=str, required=True,
        help="JSONL file with OpenAI API responses (has custom_id, response.body.choices[0].message.content)."
    )
    parser.add_argument(
        "--inplace", action="store_true", default=False,
        help="If set, overwrite the chunk-file (default). Currently only inplace modification is supported."  # future option
    )
    return parser.parse_args()


def read_jsonl(file_path: str) -> List[Dict[str, Any]]:
    records: List[Dict[str, Any]] = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as e:
                logging.warning("Skipping malformed JSON (line %d) in %s: %s", line_no, file_path, e)
    return records


def write_jsonl(file_path: str, records: List[Dict[str, Any]]) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")




# Chunk file - identified by fild text
# batch file:
# - custom_id
# - body.messages[0].content -> prompt
# - body.messages[1].content -> has template "query: {query}\nretrieved text chunk: {chunk}\n"
# - body.model -> model
# result file:
# - custom_id: matches batch file
# - response.body.choices[0].message.content -> generated title text - prefix "title: " will be removed
# Save all information to chunks
# - merge batch and result by custom_id
# - find chunk by text of {chunk} in batch file
# - add filed generated_title to chunk which is a list of generated titles:
# -- id: custom_id
# -- generated_title: extracted title
# -- model: from batch file
# -- query: from batch file
# -- prompt: from batch file


def merge_titles(batch_items: List[Dict[str, Any]], result_items: List[Dict[str, Any]]) -> int:
    batch_items = {item['custom_id']: item for item in batch_items}
    output_items = []

    for result_item in result_items:
        custom_id = result_item.get('custom_id')
        batch_item = batch_items.get(custom_id)
        if not batch_item:
            logging.warning(f"Custom ID {custom_id} in results not found in batch items, skipping")
            continue

        title_text = result_item['response']['body']['choices'][0]['message']['content']
        title_text = title_text.strip().lstrip("title:").strip()
        prompt = batch_item['body']['messages'][0]['content']
        user_message = batch_item['body']['messages'][1]['content']

        query = user_message.split("query:")[1].strip().split("\n")[0].strip()
        # take everything after "retrieved text chunk:"
        chunk_text = user_message.split("retrieved text chunk: ")[1].rstrip("\n")

        model = batch_item['body']['model']
        title_entry = {
            "id": custom_id,
            "generated_title": title_text,
            "model": model,
            "query": query,
            "prompt": prompt,
            "chunk": chunk_text
        }
        output_items.append(title_entry)
    return output_items


def main():
    args = parse_args()
    logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')

    chunks = read_jsonl(args.chunk_file)
    chunks = {c['text']: c for c in chunks if 'id' in c}
    batch_items = read_jsonl(args.batch_file)
    result_items = read_jsonl(args.result_file)

    merged_titles = merge_titles(batch_items, result_items)

    update_counter = 0

    for title in merged_titles:
        if not title['chunk'] in chunks:
            logging.warning(f"Chunk text not found in chunks: {title['chunk']}")
            continue

        if "generated_titles" not in chunks[title['chunk']]:
            chunks[title['chunk']]["generated_titles"] = []
        chunks[title['chunk']]["generated_titles"].append(title)
        update_counter += 1

    write_jsonl(args.output_file, list(chunks.values()))
    logging.info("Updated chunk file with merged titles: %s", args.output_file)
    logging.info("Total chunks updated with titles: %d", update_counter)


if __name__ == "__main__":
    main()