import argparse
import json
import os
import logging
from uuid import uuid4
import glob
from tqdm import tqdm
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(description="Convert idnes discussion csv files to common JSONL format.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the input CSV files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the output JSONL files.")
    parser.add_argument('--min-length', type=int, default=20, help="Minimum comment length to include in the output.")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.isdir(args.input_dir):
        logging.error(f"Input directory {args.input_dir} does not exist.")
        return

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    good_count = 0
    bad_count = 0
    short_count = 0

    for filename in tqdm(os.listdir(args.input_dir), desc="Processing files"):
        input_file = os.path.join(args.input_dir, filename)
        output_file = os.path.join(args.output_dir, filename.replace('.csv', '.jsonl'))

        if os.path.exists(output_file):
            logging.info(f"Output file {output_file} already exists, skipping.")
            continue

        with open(input_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()

        comments = []
        for line in lines[1:]:  # Skip header
            parts = line.strip().split(',')
            if len(parts) < 5:
                bad_count += 1
                continue


            # find the date
            date = None
            date_pos = -1
            for i, part in enumerate(parts[:5]):
                try:
                    date = datetime.strptime(part.strip(), "%d.%m.%Y %H:%M")
                    date_pos = i
                    break
                except ValueError:
                    pass

            if date is None:
                logging.warning(f"Invalid date format in line: {line.strip()}")
                bad_count += 1
                continue

            user_id = parts[0].strip()
            user_name = ', '.join(parts[1:date_pos])  # User name can be multiple parts
            votes = parts[date_pos + 1].strip()

            text = ','.join(parts[date_pos + 2:]).strip()
            if len(text) < args.min_length:
                short_count += 1
                continue

            # votes are in format +10/-5
            try:
                positive_votes, negative_votes = votes.split('/')
                positive_votes = int(positive_votes.replace('+', '').strip())
                negative_votes = int(negative_votes.replace('−', '').strip())
            except ValueError as e:
                logging.warning(f"Invalid votes format: {votes} - {e}")
                positive_votes, negative_votes = None, None

            comments.append({
                "id": str(uuid4()),
                "text": text,
                "article": filename.replace('.txt', ''),
                user_id: user_id,
                "user_name": user_name,
                "date": date.isoformat() if date else None,
                "positive_votes": positive_votes,
                "negative_votes": negative_votes
            })
            good_count += 1

        with open(output_file, 'w', encoding='utf-8') as f:
            for comment in comments:
                f.write(json.dumps(comment, ensure_ascii=False) + '\n')
    logging.info(f"Processed {good_count} good comments, {bad_count} bad comments, and {short_count} short comments.")


if __name__ == "__main__":
    main()