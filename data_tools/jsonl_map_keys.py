import argparse
import json
from glob import glob
from tqdm import tqdm

def parse_args():
    parser = argparse.ArgumentParser(description="Map keys in JSONL files to new values.")
    parser.add_argument("--input-dir", type=str, required=True, help="Directory containing the .jsonl files.")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save the modified .jsonl files.")
    args = parser.parse_args()
    return args

key_mapping = {
    # CSV export from NewtonOne
    'Kód článku': 'id_code',
    'Datum publikování': 'date',
    'Název': 'title',
    'Téma': None,
    'Zdroj': 'source',
    'Země': 'country',
    'Typ média': 'media_type',
    'Autor': 'author',
    'Strana': 'page',
    'Anotace': 'summary ',
    'Plné znění': 'text',
    'Typ zprávy': 'type',
    'Štítky': None,
    'Sentiment': 'sentiment',
    'Důležitá zpráva': None,
    'Detail zprávy v NewtonOne': 'detail',
    'Originální internetový zdroj': None,
    'Odkaz na sken': 'scan_link',
    'AVE': 'ave',
    'Datum importu': 'import_date',
    'Tištěný náklad': 'print_circulation',
    'Prodaný náklad': 'sold_circulation',
    'Rubrika': 'section',
    'Periodicita': 'periodicity',
    'Vydavatel': 'publisher',
    'Dosah': 'reach',
    'Návštěvy za měsíc': None, #'monthly_visits',
    'Celková návštěvnost': None, #'total_visits',
    'RU/den': None, #'daily_users',
    'RU/měsíc': None, #'monthly_users',
    'Infotyp': None, #'infotype',
    'GRP': 'grp',
    'Počet interakcí': None, # 'interactions',
    'Poznámka': None, #'note'

    # text export from Annopress
    'Skóre': None,
    'Název': 'title',
    #'Zdroj': 'source',
    'Datum': 'date',
    'Odkaz': None,
    #'Rubrika': 'section',
    #'Autor': 'author',
    'Str.': 'page',
    'Oblast': 'region',
    'ProfilID': 'profile_id',
    'Zpracováno': 'import_date',
    'Ročník': 'year',
    'Číslo': 'issue',
    'Jazyk': None,
    'Zkratka oblasti': 'region_short',
    'Zkratka zdroje': 'source_short',
    'Identifikace': 'id_code',
    'Klíčová slova': 'keywords',
    'ProfilSymbol': 'profile_symbol',
    'Zkratka skupiny': 'group_short',
    'HASH': 'hash',
    'Mediatyp': 'media_type',
    'Náklad': 'circulation',
    'ISSN': 'issn',
    'Domicil': 'domicile',
    'Čtenost': 'reach',
    'Mutace': 'mutation',

    # Some other keys
    'Podtitulek': 'subtitle',

    # standard keys
    'document_index': 'document_index',
    'title': 'title',
    'id': 'id',
    'document': 'document',
    'vector_index': 'vector_index',
    'text': 'text',
    'language': 'language',
}


def map_keys(record: dict) -> tuple[dict, set]:
    """
    Map keys in the record according to the key_mapping dictionary.
    If a key is not in the mapping, it will be removed from the record.
    """
    missing_keys = {k for k in record.keys() if k not in key_mapping}
    mapped_record = {key_mapping[k]: v for k, v in record.items() if k in key_mapping and key_mapping[k] is not None}
    return mapped_record, missing_keys



def main():
    args = parse_args()
    input_files = glob(f"{args.input_dir}/*.jsonl")
    all_missing_keys = set()

    if args.input_dir == args.output_dir:
        print("Input and output directories cannot be the same. Please specify different directories.")
        return

    for input_file in tqdm(input_files, desc="Processing files"):
        output_file = f"{args.output_dir}/{input_file.split('/')[-1]}"

        with open(input_file, 'r', encoding='utf-8') as infile, open(output_file, 'w', encoding='utf-8') as outfile:
            for line in infile:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                mapped_record, missing_keys = map_keys(record)
                all_missing_keys.update(missing_keys)

                outfile.write(json.dumps(mapped_record, ensure_ascii=False) + '\n')

    print(f"Processed {len(input_files)} files.")
    if all_missing_keys:
        print(f"Missing keys in the input files: {', '.join(all_missing_keys)}")
    else:
        print("No missing keys found in the input files.")


if __name__ == "__main__":
    main()
