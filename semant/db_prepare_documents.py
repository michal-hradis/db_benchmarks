import argparse
import os
import json
from pero_ocr.core.layout import PageLayout
import zipfile
from glob import glob
from uuid import uuid4
from tqdm import tqdm
from collections import defaultdict
from sqlalchemy import create_engine, select, MetaData
import logging

def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract text from a list of PAGE XML files. Input is a list of .zip files with PAGE XML files. The script reads metadata from a database.")
    parser.add_argument("-i", "--input-file", required=True, type=str, help="Input json file.")
    parser.add_argument("--output-chunk-dir", required=True, type=str, help="Output jsonl directory.")
    parser.add_argument("--page-xml-dir", required=True, type=str, help="Directory with PAGE XML files.")
    parser.add_argument("--line-confidence", default=0.6, type=float,
                        help="Minimum line confidence to include in the chunk.")
    return parser.parse_args()


def save_jsonl(data, filename):
    to_save = {}
    to_save["id"] = data["id"]
    to_save["dateIssued"] = data["startDate"]
    for key in ["title", "subTitle", "placeTerm", "publisher", "author"]:
        if key in data:
            if data[key] == None:
                to_save[key] = None
                continue
            to_save[key] = data[key][0]
            if type(to_save[key]) == list:
                to_save[key] = to_save[key][0]

    json.dump(to_save, open(filename, "w", encoding="utf-8"), ensure_ascii=False, indent=4)


def process_document(db_doc, db_connection, db_model):
    doc_id = db_doc.id
    pages_result = db_connection.execute(select(db_model['meta_records']).where(db_model['meta_records'].c.parent_id == doc_id))
    db_pages = pages_result.fetchall()
    db_pages = sorted(db_pages, key=lambda p: p.order)
    image_count = 0
    mods_count = 0
    page_xml_count = 0
    for db_page in db_pages:
        if db_page.image_path:
            image_count += 1
        if db_page.mods_path:
            mods_count += 1
        if db_page.page_xml_path:
            page_xml_count += 1

    return image_count, mods_count, page_xml_count, len(db_pages)


def extract_chunks(doc_id, db_pages, line_confidence,
                   min_chunk_chars=768, max_chunk_chars = 1024 + 128):
    chunks = []
    page_xml_dir = f'{doc_id}.temp'
    os.makedirs(page_xml_dir, exist_ok=True)

    zip_file_path = db_pages[0].page_xml_path

    if not zip_file_path or not os.path.exists(zip_file_path):
        logging.error(f"Page XML file {zip_file_path} does not exist.")
        return chunks

    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(page_xml_dir)

    for db_page in db_pages:
        layout = PageLayout()
        page_xml_file = os.path.join(page_xml_dir, f'{db_page.id}.xml')
        try:
            layout.from_pagexml(page_xml_file)
        except OSError:
            logging.error(f"{page_xml_file} is not a valid PAGE XML file or does not exist.")
            continue

        for paragraph in layout.regions:
            # confidences = ' '.join([str(line.transcription_confidence) for line in paragraph.lines])
            # print(confidences)
            if not chunks:
                chunks.append(
                    {"text": "", "start_page_id": db_page.id, "from_page": db_page.order, "order": len(chunks)})

            paragraph_lines = [line for line in paragraph.lines if
                               line.transcription and line.transcription_confidence > line_confidence]

            while paragraph_lines:
                paragraph_text = '\n'.join([line.transcription for line in paragraph_lines])
                if len(paragraph_text) <= 10:
                    break

                if len(chunks[-1]["text"]) + len(paragraph_text) > max_chunk_chars:
                    chunks[-1]["to_page"] = db_page.order
                    chunks[-1]["end_paragraph"] = False
                    chunks[-1]["text"] = chunks[-1]["text"] + '\n'
                    while len(chunks[-1]["text"]) < min_chunk_chars:
                        chunks[-1]["text"] = chunks[-1]["text"] + f'\n{paragraph_lines[0].transcription}'
                        paragraph_lines = paragraph_lines[1:]
                    chunks.append(
                        {"text": "", "start_page_id": db_page.id, "from_page": db_page.order, "order": len(chunks)})

                elif len(chunks[-1]["text"]) + len(paragraph_text) > min_chunk_chars:
                    chunks[-1]["to_page"] = db_page.order
                    chunks[-1]["end_paragraph"] = True
                    chunks[-1]["text"] = f'{chunks[-1]["text"]}\n\n{paragraph_text}'
                    chunks.append(
                        {"text": "", "start_page_id": db_page.id, "from_page": db_page.order, "order": len(chunks)})
                    paragraph_lines = []
                else:
                    chunks[-1]["text"] = f'{chunks[-1]["text"]}\n\n{paragraph_text}'
                    paragraph_lines = []

    # delete the temporary extracted files and the directory
    for file_name in glob(os.path.join(page_xml_dir, "*.xml")):
        os.remove(file_name)
    os.rmdir(page_xml_dir)

    if len(chunks) > 0:
        chunks[-1]["to_page"] = db_page.order
        chunks[-1]["end_paragraph"] = True

    if not chunks[-1]["text"]:
        chunks.pop()

    for chunk in chunks:
        chunk["id"] = str(uuid4())
        chunk["text"] = chunk["text"].strip()

    return chunks



def main():
    args = parse_args()
    if os.path.exists(args.output_chunk_file):
        print("Output chunk file already exists, exiting.")
        return

    DATABASE_URL = os.getenv("DATABASE_URL", None)  # Replace with your actual database URL
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")

    doc_to_process = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading input file", unit="line"):
            file_path = line.strip()
            doc_id = os.path.basename(file_path).split(".")[1]
            doc_to_process.append(doc_id)

    db_engine = create_engine(DATABASE_URL)

    db_model = MetaData()
    db_model.reflect(bind=db_engine)
    db_model = db_model.tables

    image_count = 0
    mods_count = 0
    page_xml_count = 0
    page_count = 0

    count_histogram = defaultdict(int)
    for counter, doc_id in tqdm(enumerate(doc_to_process), desc="Counting documents", unit="document"):
        try:
            with db_engine.connect() as db_connection:
                result = db_connection.execute(select(db_model['meta_records']).where(db_model['meta_records'].c.id == doc_id))
                result = result.fetchall()
                if not result:
                    result = []
                count_histogram[len(result)] += 1
                if result:
                    image_c, mods_c, page_xml_c, page_c = process_document(result[0], db_connection, db_model)
                    image_count += image_c
                    mods_count += mods_c
                    page_xml_count += page_xml_c
                    page_count += page_c
                    chunks = process_document(result[0], db_connection, db_model)

                    if len(chunks) > 0:
                        chunk_lengths = [len(chunk["text"]) for chunk in chunks]
                        chunk_sum = sum(chunk_lengths)
                        chunk_count = len(chunks)
                        chunk_avg = chunk_sum / chunk_count
                        chunk_min = min(chunk_lengths)
                        chunk_max = max(chunk_lengths)
                        chunk_median = sorted(chunk_lengths)[chunk_count // 2]
                        print(
                            f'{result[0].id}, {chunk_count}, {chunk_sum}, {chunk_avg:.2f}, {chunk_min}, {chunk_median}, {chunk_max}')
                    else:
                        print(f'{result[0].id}, 0, 0, 0.00, 0, 0, 0')

                    output_chunk_file = os.path.join(args.output_chunk_dir, f"{result[0].id}.jsonl")
                    with open(output_chunk_file, "w", encoding="utf-8") as f:
                        for i, chunk in enumerate(chunks):
                            chunk["document"] = result[0].id
                            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

            if counter % 10000 == 0:
                for count, num_docs in count_histogram.items():
                    print(f"Documents with {count} records: {num_docs}")
                    print(f"page_count: {page_count}, image_count: {image_count}, mods_count: {mods_count}, page_xml_count: {page_xml_count}")
        except KeyboardInterrupt as e:
            print("Interrupted by user, exiting.")
            exit(-1)
        except Exception as e:
            print(f"Error processing document {doc_id}: {e}")
            continue

    exit(-1)







if __name__ == "__main__":
    main()