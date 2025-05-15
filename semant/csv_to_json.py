import json
import argparse
import os
from glob import glob
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser(description="Convert CSV files to JSON format.")
    parser.add_argument("--source", type=str, required=True, help="Path to the source CSV file.")
    parser.add_argument("--target-dir", type=str, required=True, help="Directory to save JSON files.")
    parser.add_argument("--image-path", type=str, required=True, help="Path to the image files.")
    args = parser.parse_args()
    return args


def main():
    args = parse_args()

    with open(args.source, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    lines = [line.strip().split() for line in lines if line.strip()]

    for volume_id, year_id, year in tqdm(lines):
        images = glob(os.path.join(args.image_path, f"{volume_id}.images/*.jpg"))

        data = {
            "batch_id": "6148c058-bbb7-48cd-9284-8995f2578aab",
            "library": "mzk",
            "elements": [
                {
                    "type": "volume",
                    "periodical": True,
                    "id": volume_id,
                    "parent_id": year_id,
                    "partNumber": None,
                    "partName": None,
                    "dateIssued": None,
                    "startDate": f"{year}-01-01 00:00:00",
                    "endDate": None,
                    "title": "Lidové noviny",
                    "subTitle": None,
                    "edition": None,
                    "placeTerm": None,
                    "publisher": "Brno: Vydavatelské družstvo Lidové strany v Brně",
                    "manufacturePublisher": None,
                    "manufacturePlaceTerm": None,
                    "author": None,
                    "illustrator": None,
                    "translator": None,
                    "editor": None,
                    "seriesName": None,
                    "seriesNumber": None
                },
            ],
        }

        for index, image in enumerate(images):
            page_uuid = os.path.basename(image).split(".")[0].split(":")[0]
            data["elements"].append(
                {
                    "type": "page",
                    "id": page_uuid,
                    "batch_id": "6148c058-bbb7-48cd-9284-8995f2578aab",
                    "parent_id": volume_id,
                    "pageIndex": index
                })
        data["page_to_xml_mapping"] = {
            'something': f'/mnt/matylda0/ihradis/digiknihovna_public/mzk.ocr/{volume_id}.page_xml.zip'
        }

        # Save the JSON data to a file
        json_file_path = os.path.join(args.target_dir, f"{volume_id}.json")
        with open(json_file_path, 'w', encoding='utf-8') as json_file:
            json.dump(data, json_file, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()




