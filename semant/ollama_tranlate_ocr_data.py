import json
import os
import argparse
import glob
from tqdm import tqdm
from ollama import Client
from concurrent.futures import ThreadPoolExecutor, as_completed
import random


languages = {
    "en": "English",
    "es": "Spanish",
    "fr": "French",
    "de": "German",
    "it": "Italian",
    "pt": "Portuguese",
    "pl": "Polish",
    "ru": "Russian",
    "ar": "Arabic",
    "tr": "Turkish",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "no": "Norwegian",
    "cs": "Czech",
    "hu": "Hungarian",
    "ro": "Romanian",
    "sk": "Slovak",
    "uk": "Ukrainian",
    "el": "Greek",
    "bg": "Bulgarian",
    "sr": "Serbian",
    "hr": "Croatian",
    "sl": "Slovenian",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "et": "Estonian",
    "is": "Icelandic",
    "mt": "Maltese",
    "ga": "Irish",
    "cy": "Welsh",
    "eu": "Basque",
    "ca": "Catalan",
    "hy": "Armenian",
    "az": "Azerbaijani",
    "bs": "Bosnian",
    "la": "Latin",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Read PERO OCR training textline file and translate it to other languages."
    )
    parser.add_argument(
        "--source-file", type=str, required=True,
        help="File to process. Each line should contation: <record_id> <some_number> <text>"
    )
    parser.add_argument(
        "--target-file", type=str, required=True,
        help="File to write the processed records to. Each line will have structure: <record_id> <some_number> <language_code> <translated_text>"
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
        "--language", type=str, nargs="+", choices=list(languages.keys()), default=[
            "en", "es", "fr", "de", "it", "pt", "pl", "ru", "ar", "tr",
            "nl", "sv", "da", "no", "cs", "hu", "ro", "sk", "uk",
            "el", "bg", "sr", "hr", "sl", "lt", "lv", "et", "la"
        ], help="Language codes to translate to. Default is most supported languages."
    )
    parser.add_argument(
        "--languages-per-record", type=int, default=4,
        help="Number of random languages to translate each record to. Default is 4."
    )
    parser.add_argument(
        "--prompt", type=str, default='Translate the following line of text into {language}. Output only the translated text without anything else. Text is: "{text}"',
        help="Prompt template for the model. Use {text} as placeholder for the record's text and {language} for the target language."
    )
    parser.add_argument()
    parser.add_argument(
         "--threads", type=int, default=1,
    )
    return parser.parse_args()


def call_ollama(client: Client, model: str, prompt: str) -> str | None:
    """
    Use the Ollama Python client to generate a response from the model.
    """
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

def main():
    args = parse_args()
    client = Client(args.server)

    dst_path = os.path.join(args.target_dir, os.path.basename(src_path))

    # 1) Load all lines into memory
    with open(args.source_file, 'r', encoding='utf-8') as infile:
        records = [json.loads(line) for line in infile]

    # Read already processed line_ids from target_file if it exists
    processed_ids = set()
    if os.path.exists(args.target_file):
        with open(args.target_file, 'r', encoding='utf-8') as f:
            for line in f:
                parts = line.strip().split()
                if parts:
                    processed_ids.add(parts[0])

    # 2) For each record, pick N random target languages and prepare translation jobs
    jobs = []
    for rec in records:
        line_id = rec.split()[0]
        if line_id in processed_ids:
            continue
        some_number = rec.split()[1]
        text = " ".join(rec.split()[2:])
        chosen_langs = random.sample(args.language, min(args.languages_per_record, len(args.language)))
        for lang_code in chosen_langs:
            lang_name = languages[lang_code]
            prompt = args.prompt.format(text=text, language=lang_name)
            jobs.append({"line_id": line_id, "some_number": some_number, "lang_code": lang_code, "prompt": prompt})

    # 3) Dispatch all Ollama calls in parallel
    with open(args.target_file, 'w', encoding='utf-8') as outfile:
        with ThreadPoolExecutor(max_workers=args.threads) as executor:
            future_to_job = {
                executor.submit(call_ollama, client, args.model, job["prompt"]): job
                for job in jobs
            }

            # 4) Collect results as they complete, with a progress bar
            for future in tqdm(as_completed(future_to_job), desc="Translating", total=len(future_to_job), leave=False):
                job = future_to_job[future]
                try:
                    translated = future.result()
                    if translated is None:
                        tqdm.write(f"No response from model for record {job['line_id']}. Skipping.")
                        continue
                    outfile.write(f"{job['line_id']} {job['some_number']} {job['lang_code']} {translated}\n")
                except Exception as e:
                    tqdm.write(f"Error processing record {job['line_id']}: {e}")
                    continue

if __name__ == "__main__":
    main()

