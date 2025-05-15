import json
import os
import argparse
import glob
from tqdm import tqdm
from ollama import Client

def parse_args():
    parser = argparse.ArgumentParser(
        description="Read JSONL files, process each record via Ollama Python client, and store responses."
    )
    parser.add_argument(
        "--source-dir", type=str, required=True,
        help="Directory to read source JSONL files from."
    )
    parser.add_argument(
        "--target-dir", type=str, required=True,
        help="Directory to write processed JSONL files to."
    )
    parser.add_argument(
        "--server", type=str, required=True,
        help="Ollama server URL, e.g. http://localhost:11434"
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Model name on the Ollama server"
    )
    parser.add_argument(
        "--prompt", type=str, required=True,
        help="Prompt template for the model. Use {text} as placeholder for the record's text"
    )
    parser.add_argument(
        "--response-key", type=str, default="ollama_response",
        help="Key under which to store the model's response in each JSON record"
    )
    return parser.parse_args()


def call_ollama(client: Client, model: str, prompt_template: str, text: str) -> str:
    """
    Use the Ollama Python client to generate a response from the model.
    """
    prompt = prompt_template.format(text=text)
    # The generate method returns a dict similar to HTTP JSON
    response = client.generate(model=model, prompt=prompt)

    # Extract text or message content
    if isinstance(response, dict) and "choices" in response and response["choices"]:
        choice = response["choices"][0]
        # Chat-style
        if isinstance(choice, dict) and "message" in choice:
            return choice["message"].get("content", "")
        # Completion-style
        if isinstance(choice, dict) and "text" in choice:
            return choice.get("text", "")
    # Fallback
    for key in ("response", "result"):
        if key in response:
            return response.get(key, "")
    return ""


def main():
    args = parse_args()
    # Initialize Ollama client
    client = Client(args.server)

    os.makedirs(args.target_dir, exist_ok=True)

    jsonl_files = glob.glob(os.path.join(args.source_dir, "*.jsonl"))
    if not jsonl_files:
        print(f"No JSONL files found in {args.source_dir}.")
        return

    for src_path in tqdm(jsonl_files, desc="Files"):
        dst_path = os.path.join(args.target_dir, os.path.basename(src_path))
        if os.path.exists(dst_path):
            tqdm.write(f"Skipping existing file: {dst_path}")
            continue

        processed = []
        with open(src_path, 'r', encoding='utf-8') as infile:
            for line in tqdm(infile.readlines(), desc="Records", leave=False):
                record = json.loads(line)
                text = record.get("text", "").replace("\n", " ")
                try:
                    result = call_ollama(
                        client,
                        args.model,
                        args.prompt,
                        text
                    )
                except Exception as e:
                    result = f"<error: {e}>"
                record[args.response_key] = result
                processed.append(record)

        with open(dst_path, 'w', encoding='utf-8') as outfile:
            for rec in processed:
                outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Processing complete.")

if __name__ == "__main__":
    main()
