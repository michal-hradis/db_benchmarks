from pathlib import Path
import colorlog
from transformers import AutoTokenizer, AutoModelForTokenClassification
import torch
import tqdm
import json
import random
import logging
from tqdm.contrib.logging import logging_redirect_tqdm
from logging import Logger


def load_model(model_name: str, device: torch.device) -> tuple[AutoTokenizer, AutoModelForTokenClassification]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForTokenClassification.from_pretrained(
        model_name).to(device).eval()
    return tokenizer, model


def run_inference(model, inputs, max_batch_size: int = 1) -> torch.Tensor:
    with torch.inference_mode():
        batch_size = inputs['input_ids'].shape[0]
        if batch_size > max_batch_size:
            # Split the input into smaller batches
            outputs = []
            for i in range(0, batch_size, max_batch_size):
                batch_inputs = {k: v[i:i + max_batch_size]
                                for k, v in inputs.items()}
                batch_outputs = model(**batch_inputs)
                outputs.append(batch_outputs.logits)
            outputs = torch.cat(outputs, dim=0)
        else:
            outputs = model(**inputs).logits
    return outputs


def pre_process_text(input: str, tokenizer: AutoTokenizer, device: torch.device, overlap: int, model_max_length: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    inputs = tokenizer([input], return_tensors="pt",
                       is_split_into_words=False).to(device)
    # Tokenize the input text into chunks with overlap
    input_ids = inputs['input_ids'][0, 1:-1]
    bos = inputs['input_ids'][0, :1]
    eos = inputs['input_ids'][0, -1:]

    chunks = []
    for i in range(0, len(input_ids), (model_max_length - overlap - 2)):
        chunk = torch.cat([bos, input_ids[i:i + (model_max_length - 2)], eos])
        padding = model_max_length - len(chunk)
        chunk = torch.cat([chunk, torch.tensor(
            [tokenizer.pad_token_id] * padding, device=device, dtype=input_ids.dtype)])
        chunks.append(chunk)

    # Create the new inputs_trunc with the split chunks
    inputs_processed = {
        "input_ids": torch.stack(chunks),
    }
    inputs_processed["attention_mask"] = torch.ones_like(
        inputs_processed["input_ids"])
    return inputs, inputs_processed, bos, eos


def post_process_splitted_text(input_ids: torch.Tensor, output_logits: torch.Tensor, bos: torch.Tensor, eos: torch.Tensor, overlap: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    large_logit = []
    large_ids = []
    overlap_logits = None
    overlap_ids = None
    for logits, input_ids in zip(output_logits, input_ids):
        bos_index = (input_ids == bos).nonzero(as_tuple=True)[0][0]
        eos_index = (input_ids == eos).nonzero(as_tuple=True)[0][0]
        logits = logits[bos_index+1:eos_index]
        input_ids = input_ids[bos_index+1:eos_index]
        if overlap_logits is not None:
            pre_overlap = (logits[:overlap] + overlap_logits) / 2
        else:
            pre_overlap = logits[:overlap]
        large_logit.append(pre_overlap)
        large_logit.append(logits[overlap:-overlap])
        large_ids.append(input_ids[:overlap])
        large_ids.append(input_ids[overlap:-overlap])
        overlap_logits = logits[-overlap:]
        overlap_ids = input_ids[-overlap:]

    large_logit.append(overlap_logits)
    large_ids.append(overlap_ids)

    o = torch.zeros_like(large_logit[0][:1])
    o[0, 0] = 1

    logit = torch.cat([o] + large_logit + [o])
    classes = torch.argmax(logit, dim=-1)
    ids = torch.cat([bos] + large_ids + [eos])
    # assert (large_ids == inputs["input_ids"][0]).all().item(), "Mismatch between large_ids and inputs['input_ids'][0]"
    return logit, classes, ids


def post_process_bio(id2label: dict[int, str], predicted_classes: torch.Tensor, inputs) -> list[tuple[str, str]]:
    labels: list[tuple[str, str]] = [
        (id2label[t.item()], w) for t, w in zip(predicted_classes, inputs.tokens())
    ]
    texts = [l[1] for l in labels]
    locs = [l[0].split("-")[0] for l in labels]
    types = [''.join(l[0].split("-")[1:]) for l in labels]
    word_indicies = inputs.word_ids()

    entities = []
    current_entity_type = ''
    current_entity = ''
    last_word_index = -1

    for text, loc, type, word_index in zip(texts, locs, types, word_indicies):
        if text[0] == '▁' and last_word_index != word_index:
            text = ' ' + text[1:]

        match (text, loc, current_entity_type, type):
            case (_, 'B', 'pf', 'ps'):
                current_entity += text

            case (_, 'B', _, _):
                entities.append(
                    (current_entity_type, current_entity.strip()))
                current_entity = text
                current_entity_type = type
            case (_, 'I', _, _):
                current_entity += text
            case (_, 'O', _, _):
                entities.append(
                    (current_entity_type, current_entity.strip()))
                current_entity = text
                current_entity_type = ''
        last_word_index = word_index
    return entities


def process_single_str_object(
    model,
    tokenizer,
    device,
    input_text: str,
    overlap: int,
    max_batch_size: int,
    max_input_length: int,
    entities_only: bool
):
    inputs, inputs_processed, bos, eos = pre_process_text(
        input_text, tokenizer, device, overlap=overlap, model_max_length=max_input_length)
    outputs = run_inference(model, inputs_processed,
                            max_batch_size=max_batch_size)
    logit, classes, ids = post_process_splitted_text(
        inputs_processed["input_ids"], outputs, bos, eos, overlap=overlap)
    entities = post_process_bio(
        model.config.id2label, classes, inputs)
    entities = [{
        'type': e[0],
        'text': e[1],
    } for e in entities if (e[0] != '' or not entities_only)]

    entities_alt = {}

    for e in entities:
        if e['type'] not in entities_alt:
            entities_alt[e['type']] = []
        entities_alt[e['type']].append(e['text'])

    return entities_alt


def process_file(
    file_path: Path,
    output_file_path: Path,
    model: AutoModelForTokenClassification,
    tokenizer: AutoTokenizer,
    device: torch.device,
    overlap: int,
    max_batch_size: int,
    max_input_length: int,
    entities_only: bool,
):
    objects = [json.loads(line)
               for line in file_path.open("r", encoding="utf-8")]
    for i, obj in enumerate(objects):
        text = obj["text"]
        with logging_redirect_tqdm():
            logging.debug(
                f"Processing line {i} from {file_path} with text: {text[:20]}...")
        ners = process_single_str_object(
            model, tokenizer, device, text, overlap=overlap, max_batch_size=max_batch_size, max_input_length=max_input_length, entities_only=entities_only)

        ners = {
            'ner_' + k: v for k, v in ners.items()
        }

        with output_file_path.open("a", encoding="utf-8") as f:
            obj.update(ners)
            f.write(json.dumps(obj) + "\n")


def get_not_yet_processed(input_dir: Path, output_dir: Path) -> list[Path]:
    input_files = set(map(lambda x: x.relative_to(
        input_dir), input_dir.rglob("*.jsonl")))
    output_files = set(map(lambda x: x.relative_to(
        output_dir), output_dir.rglob("*.jsonl")))
    not_yet_processed = list(
        map(lambda x: input_dir / x, input_files - output_files))
    random.shuffle(not_yet_processed)
    return not_yet_processed


def get_all_files(input_dir: Path) -> list[Path]:
    return list(input_dir.rglob("*.jsonl"))


def main(
    model: str,
    device: torch.device,
    input_dir: Path,
    output_dir: Path,
    overlap: int,
    max_batch_size: int,
    max_input_length: int,
):
    device = torch.device(device)
    logging.info(f"Loading model {model} on device {device}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    logging.info(f"Loaded tokenizer {type(tokenizer).__name__}")
    model = AutoModelForTokenClassification.from_pretrained(
        model).to(device).eval()
    logging.info(f"Loaded model {type(model).__name__}")

    all_files = list(tqdm.tqdm(input_dir.rglob("*.jsonl"),
                     desc="Loading files", total=len(get_all_files(input_dir))))
    number_of_files = len(all_files)
    processing_progress = tqdm.tqdm(
        desc="Processing files", total=number_of_files)

    while len((not_yet_processed := get_not_yet_processed(input_dir, output_dir))) > 0:
        number_of_files_to_process = len(not_yet_processed)
        processing_progress.n = number_of_files - number_of_files_to_process
        processing_progress.refresh()

        input_file = not_yet_processed[0]
        output_file = output_dir / input_file.relative_to(input_dir)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        if output_file.exists():
            continue
        output_file.touch(exist_ok=True)
        process_file(
            input_file, output_file, model, tokenizer, device,
            overlap=overlap, max_batch_size=max_batch_size,
            max_input_length=max_input_length,
            entities_only=True
        )

    processing_progress.n = number_of_files
    processing_progress.refresh()


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Process NER JSONL files.")
    parser.add_argument("-s", "--source-dir", type=Path, required=True,
                        help="Path to input directory containing JSONL files.")
    parser.add_argument("-t", "--target-dir", type=Path, required=True,
                        help="Path to output directory that will contain output JSONL files.")
    parser.add_argument("--model", type=str, default='stulcrad/CNEC1_1_xlm-roberta-large',
                        help="HuggingFace model name or path.")
    parser.add_argument("--device", type=torch.device, default=torch.device("cuda:0"),
                        help="Device to use (e.g., 'cuda', 'cuda:0' or 'cpu').")
    parser.add_argument("--overlap", type=int, default=50,
                        help="Overlap size for chunking.")
    parser.add_argument("--max_batch_size", type=int, default=32,
                        help="Maximum batch size for inference. Only applicable if single text input is larger than the max_input_length. Different texts are always processed in different batches.")
    parser.add_argument("--max_input_length", type=int, default=512,
                        help="Maximum input length for the model.")
    args = parser.parse_args()

    handler = colorlog.StreamHandler()
    handler.setFormatter(colorlog.ColoredFormatter(
        '%(red)s%(levelname)s:%(name)s:%(message)s'))
    logging.basicConfig(level=logging.INFO, handlers=[handler])

    main(
        model=args.model,
        device=args.device,
        input_dir=args.source_dir,
        output_dir=args.target_dir,
        overlap=args.overlap,
        max_batch_size=args.max_batch_size,
        max_input_length=args.max_input_length
    )
