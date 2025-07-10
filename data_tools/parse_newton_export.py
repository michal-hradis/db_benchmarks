import re
import argparse
import json
from uuid import uuid4


def split_record(record, target_chunk_chars=1024):
    """
    Splits a record into smaller chunks based on character limits.
    """

    text = record['text']
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

    del record['text']  # Remove the original content field
    output_records = []
    for i, chunk in enumerate(chunks):
        new_record = record.copy()
        new_record['id'] = str(uuid4())
        new_record['document_index'] = i
        new_record['text'] = chunk.strip()
        output_records.append(new_record)
    return output_records


def extract_articles(file_path):

    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Split on any line consisting solely of hyphens
    raw_articles = re.split(r'^-{5,}$', content, flags=re.MULTILINE)
    raw_articles = raw_articles[1:]

    articles = []

    for chunk in raw_articles:
        chunk = chunk.strip()
        if not chunk:
           continue

        lines = [l.strip() for l in chunk.splitlines()]

        blocks = []
        current_block = []
        for line in lines:
            if not line:
                if current_block:
                    blocks.append(current_block)
                    current_block = []
                else:
                    current_block = []
            else:
                current_block.append(line)
        if current_block:
            blocks.append(current_block)

        title_block = blocks[0]
        metadata_block = blocks[1]
        body = blocks[2:]
        body = '\n\n'.join(['\n'.join(b) for b in body]).strip()

        m = re.match(r'\d+\.\s*(.*)', title_block[0])
        title = m.group(1).strip() if m else lines[0].strip()

        data = {'title': title}
        for line in metadata_block:
            if ':' in line:
                key, val = line.split(':', 1)
                data[key.strip()] = val.strip()

        clean_body = re.sub(r'\{\[p[\s\S]*?\{\[\/p\]\}', '', body)
        data['text'] = clean_body.strip()
        data['document'] = str(uuid4())  # Assign a unique document ID
        articles.append(data)

    return articles


def main():
    parser = argparse.ArgumentParser(description="Extract articles from a Blesk newspaper file.")
    parser.add_argument('-i', '--input-file', required=True, type=str, help="Path to the input file containing newspaper articles.")
    parser.add_argument('-o', '--output-file', type=str, help="Path to the output jsonl file to save extracted article chunks.")
    parser.add_argument('--target-chunk-chars', type=int, default=1024, help="Target number of characters per chunk.")
    args = parser.parse_args()

    articles = extract_articles(args.input_file)
    chunks = []
    for article in articles:
        chunks.extend(split_record(article, target_chunk_chars=args.target_chunk_chars))
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
