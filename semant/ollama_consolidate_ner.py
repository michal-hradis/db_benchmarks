import json
import os
import argparse
import glob
from tqdm import tqdm
from ollama import Client
from concurrent.futures import ThreadPoolExecutor, as_completed

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
         "--threads", type=int, default=1,
    )
    return parser.parse_args()

def process_prompt(client: Client, model: str, prompt: str) -> str | None:
    try:
        response = client.generate(model=model, prompt=prompt)
    except Exception as e:
        print(f"Error calling model {model}: {e}")
        return None

        # Extract text or message content
    if isinstance(response, dict) and "choices" in response and response["choices"]:
        choice = response["choices"][0]
        # Chat-style
        if isinstance(choice, dict) and "message" in choice:
            return choice["message"].get("content", "")
        # Completion-style
        if isinstance(choice, dict) and "text" in choice:
            return choice.get("text", "")
        # Fallback keys
    for key in ("response", "result"):
        if key in response:
            return response.get(key, "")
    return None


prompts = {
    "ner_P": "Z následujícího seznamu změň všechny jména do prního pádu jednotného čísla standardního tvaru. "
             "Každou osobu uveď jen jednou.  Pokud se nejedná o jméno, vynechej ho."
             "Příklad vstupu: Josefu Kajetánu; tabák; pes; J. Kajetánovi; Josefu; Lhotské; Tonda Dostálův - "
             "Výstup: Josef Kajetán; Lhotská; Tonda Dostál - Seznam pro zpracování je: {text} \n"
             "Vypiš jen seznam oddělený středníky.",
    "ner_G": "Z následujícího seznamu změň všechna geografická jména do prního pádu jednotného čísla standardního tvaru. "
             "Každé geografické jména uveď jen jednou. Pokud se nejedná o jméno, vynechej ho. "
             "Příklad vstupu: Rakousku; tabák; pes; Rakousko; UK; Spojené Království; Sázavou, Prahy, Ferdinandova ul. - "
             "Výstup: Rakousko; Spojené Královstí; Sázava, Praha, Ferdinandova ul. - Seznam pro zpracování je: {text} \n"
             "Vypiš jen seznam oddělený středníky.",
    "ner_I": "Z následujícího seznamu změň všechny jména do prního pádu jednotného čísla standardního tvaru. "
             "Kažou instituci/firmu uveď jen jednou. Pokud se nejedná o jméno, vynechej ho."
             "Příklad vstupu: Komerční banka; tabák; pes; Komerční bankou; Komerční bance; U tří čápů; Dukovany - "
             "Výstup: Komerční banka; U tří čápů; Dukovany - Seznam pro zpracování je: {text} \n"
             "Vypiš jen seznam oddělený středníky.",
    "ner_O": "Z následujícího seznamu změň všechny jména do prního pádu jednotného čísla standardního tvaru. "
             "Každý výrobek, dílo, knihu, film, ... uveď jen jednou. Pokud se nejedná o jméno, vynechej ho."
             "Příklad vstupu: Ford model T; tabák; pes; Hvězdné války; Hvězdných válek; Bible; Bibli - "
             "Výstup: Ford model T; Hvězdné války; Bible - Seznam pro zpracování je: {text} \n"
             "Vypiš jen seznam oddělený středníky.",
}


def call_ollama(client: Client, model: str, obj: dict[str, list[str]]) -> dict[str, list[str]] | None:
    """
    Use the Ollama Python client to generate a response from the model.
    """
    results = {}
    for key in ["ner_P", "ner_T", "ner_A", "ner_G", "ner_I", "ner_M", "ner_O"]:
        if key in obj and obj[key]:
            prompt = prompts[key].format(text='; '.join(obj[key]))
            response = process_prompt(client, model, prompt)
            if response is None:
                print(f"Error processing {key} for record: {obj}")
                return None

            print(f"Response for {key}: {'; '.join(obj[key])}  -->>  {response}")
            response = [r.strip for r in response.split(';')]
            results[key] = response
    return results

def main():
    args = parse_args()
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
        else:
            # Create an empty target file to take possesion of it
            # when multiple processes are running on the same directory
            open(dst_path, 'a').close()

        # 1) Load all records into memory
        with open(src_path, 'r', encoding='utf-8') as infile:
            records = [json.loads(line) for line in infile]

        # 2) Dispatch all Ollama calls in parallel
        processed = []
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            future_to_rec = {
                executor.submit(
                    call_ollama,
                    client,
                    args.model,
                    rec
                ): rec
                for rec in records
            }

            # 3) Collect results as they complete, with a progress bar
            for future in tqdm(as_completed(future_to_rec),
                               desc="Records", total=len(future_to_rec), leave=False):
                rec = future_to_rec[future]
                try:
                    result_value = future.result()
                    if result_value is None:
                        tqdm.write("No response from model for record. Exiting.")
                        exit(1)

                    for key, value in rec.items():
                        rec[key] = value.replace("\n", " ")

                except Exception as e:
                    print(f"<error: {e}>")
                processed.append(rec)

        # 4) Write out the processed records
        with open(dst_path, 'w', encoding='utf-8') as outfile:
            for rec in processed:
                outfile.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print("Processing complete.")

if __name__ == "__main__":
    main()
