import argparse
import os
import json
from pero_ocr.core.layout import PageLayout
import zipfile
from glob import glob
from uuid import uuid4

def parse_args():
    parser = argparse.ArgumentParser(description="Load document json files with metadata and pages, load PAGE XML files and prepare jsonl files with text chunks and document metadata.")
    parser.add_argument("-i", "--input-file", required=True, type=str, help="Input json file.")
    parser.add_argument("--output-chunk-file", required=True, type=str, help="Output jsonl file.")
    parser.add_argument("--output-doc-file", required=True, type=str, help="Output json file with metadata.")
    parser.add_argument("--page-xml-dir", required=True, type=str, help="Directory with PAGE XML files.")
    parser.add_argument("--line-confidence", default=0.6, type=float, help="Minimum line confidence to include in the chunk.")
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

def main():
    args = parse_args()
    if os.path.exists(args.output_chunk_file):
        print("Output chunk file already exists, exiting.")
        return

    data = json.load(open(args.input_file, "r", encoding="utf-8"))

    if not os.path.exists(args.page_xml_dir):
        os.mkdir(args.page_xml_dir)


    # unzip PAGE XML files
    zip_file_path = list(data["page_to_xml_mapping"].values())[0]
    with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
        zip_ref.extractall(args.page_xml_dir)

    min_chunk_chars = 768
    max_chunk_chars = 1024 + 128
    chunks = []

    save_jsonl(data["elements"][0], args.output_doc_file)
    for page in data["elements"][1:]:
        layout = PageLayout()
        try:
            layout.from_pagexml(os.path.join(args.page_xml_dir, page["id"] + ".xml"))
        except OSError:
            print(f"Error reading {os.path.join(args.page_xml_dir, page['id'] + '.xml')}")
            continue

        for paragraph in layout.regions:
            #confidences = ' '.join([str(line.transcription_confidence) for line in paragraph.lines])
            #print(confidences)
            if not chunks:
                chunks.append({"text": "", "start_page_id": page["id"], "from_page": page["pageIndex"], "order": len(chunks)})

            paragraph_lines = [line for line in paragraph.lines if line.transcription and line.transcription_confidence > args.line_confidence]
            while paragraph_lines:
                paragraph_text = '\n'.join([line.transcription for line in paragraph_lines])
                if len(paragraph_text) <= 10:
                    break

                if len(chunks[-1]["text"]) + len(paragraph_text) > max_chunk_chars:
                    chunks[-1]["to_page"] = page["pageIndex"]
                    chunks[-1]["end_paragraph"] = False
                    chunks[-1]["text"] = chunks[-1]["text"] + '\n'
                    while len(chunks[-1]["text"]) < min_chunk_chars:
                        chunks[-1]["text"] = chunks[-1]["text"] + f'\n{paragraph_lines[0].transcription}'
                        paragraph_lines = paragraph_lines[1:]
                    chunks.append({"text": "", "start_page_id": page["id"], "from_page": page["pageIndex"], "order": len(chunks)})

                elif len(chunks[-1]["text"]) + len(paragraph_text) > min_chunk_chars:
                    chunks[-1]["to_page"] = page["pageIndex"]
                    chunks[-1]["end_paragraph"] = True
                    chunks[-1]["text"] = f'{chunks[-1]["text"]}\n\n{paragraph_text}'
                    chunks.append({"text": "", "start_page_id": page["id"], "from_page": page["pageIndex"], "order": len(chunks)})
                    paragraph_lines = []
                else:
                    chunks[-1]["text"] = f'{chunks[-1]["text"]}\n\n{paragraph_text}'
                    paragraph_lines = []

    # delete the temporary extracted files
    for file_name in glob(os.path.join(args.page_xml_dir, "*.xml")):
        os.remove(file_name)

    if len(chunks) > 0:
        chunks[-1]["to_page"] = page["pageIndex"]
        chunks[-1]["end_paragraph"] = True

    if not chunks[-1]["text"]:
        chunks.pop()
        
    for chunk in chunks:
        chunk["id"] = str(uuid4())
        chunk["text"] = chunk["text"].strip() 

    if len(chunks) > 0:
        chunk_lengths = [len(chunk["text"]) for chunk in chunks]
        chunk_sum = sum(chunk_lengths)
        chunk_count = len(chunks)
        chunk_avg = chunk_sum / chunk_count
        chunk_min = min(chunk_lengths)
        chunk_max = max(chunk_lengths)
        chunk_median = sorted(chunk_lengths)[chunk_count // 2]
        print(f'{data["elements"][0]["id"]}, {chunk_count}, {chunk_sum}, {chunk_avg:.2f}, {chunk_min}, {chunk_median}, {chunk_max}')
    else:
        print(f'{data["elements"][0]["id"]}, 0, 0, 0.00, 0, 0, 0')

    with open(args.output_chunk_file, "w", encoding="utf-8") as f:
        for i, chunk in enumerate(chunks):
            chunk["document"] = data["elements"][0]["id"]
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()