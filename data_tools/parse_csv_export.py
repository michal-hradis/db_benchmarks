import re
import argparse
import json
import csv
from tqdm import tqdm
from uuid import uuid4


def extract_articles(file_path):
    """
    Just convert the csv rows to dicts and return them. The csv file is expected to have a header row.
    :param file_path: Path to the input csv file.
    :return: List of articles as dictionaries.
    """
    articles = []
    with open(file_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter=';')
        for row in reader:
            row['id'] = str(uuid4())
            articles.append(row)
    return articles

key_mapping = {
    'Kód článku': 'id_code',
    'Datum publikování': 'date',
    'Název': 'title',
    #'Téma': 'topic',
    'Zdroj': 'source',
    'Země': 'country',
    'Typ média': 'media_type',
    'Autor': 'author',
    'Strana': 'page',
    'Anotace': 'summary ',
    'Plné znění': 'text',
    'Typ zprávy': 'type',
    #'Štítky': 'tags',
    'Sentiment': 'sentiment',
    #'Důležitá zpráva': 'important',
    'Detail zprávy v NewtonOne': 'detail',
    #'Originální internetový zdroj': 'original_source',
    'Odkaz na sken': 'scan_link',
    'AVE': 'ave',
    'Datum importu': 'import_date',
    'Tištěný náklad': 'print_circulation',
    'Prodaný náklad': 'sold_circulation',
    'Rubrika': 'section',
    'Periodicita': 'periodicity',
    'Vydavatel': 'publisher',
    'Dosah': 'reach',
    #'Návštěvy za měsíc': 'monthly_visits',
    #'Celková návštěvnost': 'total_visits',
    #'RU/den': 'daily_users',
    #'RU/měsíc': 'monthly_users',
    #'Infotyp': 'infotype',
    'GRP': 'grp',
    #'Počet interakcí': 'interactions',
    #'Poznámka': 'note'
}

def normalize_keys(record: dict) -> dict:
    normalized_record = {key_mapping[k]: v for k, v in record.items() if k in key_mapping}
    return normalized_record

def main():
    parser = argparse.ArgumentParser(description="Extract articles from an exported csv file of newspaper articles.")
    parser.add_argument('-i', '--input-file', required=True, type=str, help="Path to the input csv file.")
    parser.add_argument('-o', '--output-file', type=str, help="Path to the output jsonl file to save extracted article chunks.")
    parser.add_argument('--target-chunk-chars', type=int, default=1024, help="Target number of characters per chunk.")
    args = parser.parse_args()

    articles = extract_articles(args.input_file)
    print(f'Extracted {len(articles)} articles from {args.input_file}')
    for i in range(len(articles)):
        articles[i]['document'] = str(uuid4())  # Assign a unique ID to each article
        articles[i]['id'] = articles[i]['document']
        articles[i] = normalize_keys(articles[i])  # Normalize keys according to the mapping

    with open(args.output_file, 'w', encoding='utf-8') as f:
        for article in articles:
            f.write(json.dumps(article, ensure_ascii=False) + '\n')


if __name__ == "__main__":
    main()
