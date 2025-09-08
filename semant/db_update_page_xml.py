import argparse
import os
from collections import defaultdict
from tqdm import tqdm
from sqlalchemy import create_engine, select, MetaData
import logging
from multiprocessing import Pool

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def parse_args():
    parser = argparse.ArgumentParser(
        description="Add PAGE XML paths to database.")
    parser.add_argument("-i", "--input-file", required=True, type=str, help="Input json file.")
    parser.add_argument("--page-xml-dir", required=True, type=str, help="Root directory containing PAGE XML files.")
    return parser.parse_args()


class ProcessingWorker:
    def __init__(self, db_url, page_xml_dir):
        self.db_url = db_url
        self.page_xml_dir = page_xml_dir

        self.db_model = None
        self.db_engine = None

    def _init_db(self):
        self.db_engine = create_engine(self.db_url)
        self.db_model = MetaData()
        self.db_model.reflect(bind=self.db_engine)
        self.db_model = self.db_model.tables

    def __call__(self, request) -> dict | None:

        if self.db_model is None:
            self._init_db()

        doc_id, library_id = request

        function_result = {
                "doc_id": doc_id,
                "library": library_id,
                "doc_missing": True,
                "wrong_library": False,
                "page_xml_missing": False,
                "other_library": None,
                "updated_pages": 0,
                "updated_document": 0
            }

        with self.db_engine.connect() as db_connection:
            try:
                result = db_connection.execute(select(self.db_model['meta_records']).where(self.db_model['meta_records'].c.id == doc_id))
            except Exception as e:
                return function_result
            result = result.fetchall()
            if not result:
                return function_result

            db_doc = result[0]
            function_result["doc_missing"] = False

            #if db_doc.library != library_id:
            #    function_result["wrong_library"] = True
            #    function_result["other_library"] = f"{library_id}---{db_doc.library}"
            #    return function_result
            
            # Check if page XML file exists
            page_xml_file = os.path.join(self.page_xml_dir, library_id, f"{doc_id}.page_xml.zip")
            #if not os.path.exists(page_xml_file):
            #    function_result["page_xml_missing"] = True
            #    return function_result

            try:
                doc_id = db_doc.id
                pages_result = db_connection.execute(
                    select(self.db_model['meta_records']).where(self.db_model['meta_records'].c.parent_id == doc_id))
                db_pages = pages_result.fetchall()
                db_pages = sorted(db_pages, key=lambda p: p.order)

                # update page_xml file paths for all pages and document
                for page in db_pages:
                    if page.page_xml_path is None or page.page_xml_path == "":
                        function_result["updated_pages"] += 1
                        db_connection.execute(
                            self.db_model['meta_records'].update().where(self.db_model['meta_records'].c.id == page.id).values(page_xml_path=page_xml_file)
                        )

                if db_doc.page_xml_path is None or db_doc.page_xml_path == "":
                    db_connection.execute(
                        self.db_model['meta_records'].update().where(self.db_model['meta_records'].c.id == doc_id).values(page_xml_path=page_xml_file)
                    )
                    function_result["updated_document"] += 1

            except Exception as e:
                return function_result

            return function_result


_worker = None

def _init_worker(db_url, page_xml_dir):
    # called exactly once in each worker process
    global _worker
    _worker = ProcessingWorker(db_url, page_xml_dir)

def worker_process(doc_id):
    # called for every task in each worker, but uses the same _worker
    return _worker(doc_id)


def main():
    args = parse_args()

    DATABASE_URL = os.getenv("DATABASE_URL", None)  # Replace with your actual database URL
    if not DATABASE_URL:
        raise ValueError("DATABASE_URL environment variable is not set.")

    doc_to_process = []
    with open(args.input_file, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Reading input file", unit="line"):
            file_path = line.strip()
            library_id = os.path.basename(file_path).split(".")[0]
            doc_id = os.path.basename(file_path).split(".")[1]
            doc_to_process.append((doc_id, library_id))

    logging.info(f"Read {len(doc_to_process)} documents from input file.")

    logging.info(f"Documents to process: {len(doc_to_process)}")
    # filter already processed documents

    logging.info(f"Documents to process after filtering: {len(doc_to_process)}")

    if not doc_to_process:
        logging.info("No documents to process. Exiting.")
        return

    wrong_library = 0
    missing_page_xml = 0
    failed_doc_count = 0
    missing_doc_count = 0
    updated_document = 0
    updated_pages = 0

    missing_docs = defaultdict(int)
    library_mapping = defaultdict(int)

    counter = 0
    with Pool(processes=4,
                initializer=_init_worker,
                initargs=(DATABASE_URL, args.page_xml_dir, )  # Pass the database URL and page XML directory to the worker
              ) as pool:
        for result in tqdm(pool.imap(worker_process, doc_to_process), total=len(doc_to_process), desc="Processing documents"):
            if counter % 1000 == 0:
                print(f"Failed documents: {failed_doc_count} / {counter}")
                print(
                    f" missing_doc_count: {missing_doc_count}, wrong_library: {wrong_library}, missing_page_xml: {missing_page_xml}, updated_document: {updated_document}, updated_pages: {updated_pages}")

                for library, count in missing_docs.items():
                    print(f"Library {library} has {count} missing documents.")

                for library, count in library_mapping.items():
                    print(f"{library} : {count}")

            counter += 1

            if result["doc_missing"]:
                missing_docs[result["library"]] += 1
            if result["other_library"]:
                library_mapping[result["other_library"]] += 1
            missing_doc_count += result["doc_missing"]
            wrong_library += result["wrong_library"]
            missing_page_xml += result["page_xml_missing"]
            failed_doc_count += result["doc_missing"] or result["wrong_library"] or result["page_xml_missing"]
            updated_pages += result["updated_pages"]
            updated_document += result["updated_document"]



if __name__ == "__main__":
    main()