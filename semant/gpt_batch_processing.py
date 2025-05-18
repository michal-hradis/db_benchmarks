import argparse
import json
import os
from pathlib import Path
from typing import List
from openai import OpenAI

# Define your message templates
TEMPLATES = {
    "specific": [
        {"role": "system",
         "content": "Na základě přiloženého textu, napiš seznam "
         "dvaceti otázek, které by mohl zadat historik, student, lingvista "
         "nebo podobní lidé při vyhledávání informací ve velké kolekci dokumentů."
         "Na otázky musí být v přiloženém textu možné najít apoň částečnou odpověď. "
         "Otázky by měly být konkrétní na konkrétní fakta. "
         "Otázky by měly být takové, které by mohl člověk zadávat při vyhledávání ve velké kolekci historických dokumentů. "
         "Vypiš pouze otázky bez dalšího textu. Ať jsou otázky různorodé svým obsahem i stylem. "
         "Otázky se nesmí přímo odkazovat na uvedený text a jeho obsah, nebo jedna na druhou."},
        {"role": "user", "content": "Text je: {text}"}
    ],
    "general": [
        {"role": "system",
         "content": "Na základě přiloženého textu, napiš seznam "
         "dvaceti otázek, které by mohl zadat historik, student, lingvista "
         "nebo podobní lidé při vyhledávání informací ve velké kolekci dokumentů."
         "Na otázky musí být v přiloženém textu možné najít apoň částečnou odpověď. "
         "Otázky by měly být konkrétní na konkrétní fakta. "
         "Otázky by měly být obecné - takové, na které by mohl člověk vyhledávat odpovědi ve velké kolekci historických dokumentů. "
         "Vypiš pouze otázky bez dalšího textu. Ať jsou otázky různorodé svým obsahem i stylem. "
         "Otázky se nesmí přímo odkazovat na uvedený text a jeho obsah, nebo jedna na druhou."},
        {"role": "user", "content": "What is the sentiment of this text? {text}"}
    ]
}

def load_jsonl(file_path: str) -> List[dict]:
    with open(file_path, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def create_batch_tasks(jsonl_files: List[str], template: List[dict], model: str) -> List[dict]:
    """
    Create a list of tasks formatted for the OpenAI Batch API.
    Each task includes a custom_id, HTTP method, endpoint URL, and request body.
    """
    tasks = []
    task_counter = 0
    for file_path in jsonl_files:
        for item in load_jsonl(file_path):
            text = item.get("text", "")
            messages = [
                {"role": msg["role"], "content": msg["content"].replace("{text}", text)}
                for msg in template
            ]
            tasks.append({
                "custom_id": f"task-{task_counter}",
                "method": "POST",
                "url": "/v1/chat/completions",
                "body": {
                    "model": model,
                    "messages": messages
                }
            })
            task_counter += 1
    return tasks

def save_batch_file(tasks: List[dict], output_path: str):
    """Save the list of batch tasks to a JSONL file."""
    with open(output_path, "w", encoding="utf-8") as f:
        for task in tasks:
            f.write(json.dumps(task) + "\n")

def submit_batch(batch_file: str, client: OpenAI, endpoint: str = "/v1/chat/completions", completion_window: str = "24h") -> str:
    """
    Upload the batch file to OpenAI, create a batch job, and return the batch ID.
    """
    with open(batch_file, "rb") as f:
        batch_file_obj = client.files.create(file=f, purpose="batch")
    batch_job = client.batches.create(
        input_file_id=batch_file_obj.id,
        endpoint=endpoint,
        completion_window=completion_window
    )
    print("Batch submitted. Batch ID:", batch_job.id)
    return batch_job.id

def download_results(batch_id: str, client: OpenAI, output_path: str):
    """
    Retrieve and download the results of a completed batch job.
    """
    batch_job = client.batches.retrieve(batch_id)
    if batch_job.status != "completed":
        print(f"Batch {batch_id} status: {batch_job.status}. Results not ready yet.")
        return
    result_file_id = batch_job.output_file_id
    content = client.files.content(result_file_id).content
    with open(output_path, "wb") as f:
        f.write(content)
    print(f"Results downloaded to {output_path}")

def main():
    parser = argparse.ArgumentParser(description="OpenAI Batch API Processing Script")
    parser.add_argument("--input-dir", required=True, help="Directory with jsonl files")
    parser.add_argument("--template", required=True, choices=TEMPLATES.keys(), help="Template name")
    parser.add_argument("--batch-file", default="batch_tasks.jsonl", help="Path to save batch tasks file")
    parser.add_argument("--model", default="gpt-3.5-turbo", help="OpenAI model name")
    parser.add_argument("--submit", action="store_true", help="Submit batch to OpenAI")
    parser.add_argument("--batch-id", help="Batch ID to download results")
    parser.add_argument("--download-results", help="Path to save downloaded results")
    args = parser.parse_args()

    # Initialize OpenAI client using API key from environment
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # If downloading results, do that and exit
    if args.batch_id and args.download_results:
        download_results(args.batch_id, client, args.download_results)
        return

    # Prepare batch tasks
    jsonl_files = [str(p) for p in Path(args.input_dir).glob("*.jsonl")]
    tasks = create_batch_tasks(jsonl_files, TEMPLATES[args.template], args.model)
    save_batch_file(tasks, args.batch_file)
    print(f"Batch tasks file saved to {args.batch_file}")

    # Submit batch if requested
    if args.submit:
        batch_id = submit_batch(args.batch_file, client)
        print(f"Batch submitted. Batch ID: {batch_id}")

if __name__ == "__main__":
    main()
