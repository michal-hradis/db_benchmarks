from torch.utils.data import Dataset

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerFast
from torch.utils.data import DataLoader
from pytorch_lightning import LightningDataModule
import pyarrow.parquet as pq
from torch.utils.data import IterableDataset, get_worker_info
import random
import os
from glob import glob


class ParquetIterableDataset(IterableDataset):
    def __init__(self, parquet_paths, tokenizer=None, max_length=256):
        # Accept a single path, a directory, or a list:
        if isinstance(parquet_paths, str):
            if os.path.isdir(parquet_paths):
                self.files = sorted(glob(os.path.join(parquet_paths, "*.parquet")))
            else:
                self.files = [parquet_paths]
        else:
            self.files = list(parquet_paths)
        assert self.files, f"No parquet files found in {parquet_paths}"

        self.tokenizer  = tokenizer
        self.max_length = max_length

        # Create list of row groups
        self.row_groups = []
        for fp in self.files:
            pf = pq.ParquetFile(fp)
            # store (file_path, num_row_groups)
            for i in range(pf.num_row_groups):
                self.row_groups.append((fp, i))

        # shuffle row groups to ensure random access - with constant seed for reproducibility
        random.seed(42)
        random.shuffle(self.row_groups)

    def __iter__(self):
        worker_info = get_worker_info()

        # Split row groups across workers
        if worker_info is None:
            row_group_ids = range(len(self.row_groups))
        else:
            num_workers = worker_info.num_workers
            worker_id = worker_info.id
            row_group_ids = list(range(len(self.row_groups)))[worker_id::num_workers]

        for global_rg_id in row_group_ids:
            pq_file, rg_id = self.row_groups[global_rg_id]
            pq_file = pq.ParquetFile(pq_file)
            row_group = pq_file.read_row_group(rg_id)
            text_array = row_group.column("text")
            embedding_array = row_group.column("embedding")

            for i in range(len(text_array)):
                text = text_array[i].as_py()
                embedding = torch.tensor(embedding_array[i].values, dtype=torch.float16)

                if self.tokenizer:
                    # Tokenize the text if a tokenizer is provided
                    tokens = self.tokenizer(text, truncation=True, padding="max_length", max_length=self.max_length,
                                            return_tensors="pt")
                    yield {
                        "text": text,
                        "embedding": embedding,
                        "input_ids": tokens["input_ids"].squeeze(0),
                        "attention_mask": tokens["attention_mask"].squeeze(0),
                    }
                else:
                    yield {
                        "text": text,
                        "embedding": embedding,
                    }


class BufferedShuffleDataset(IterableDataset):
    def __init__(self, dataset, buffer_size=10000, seed=42):
        self.dataset = dataset
        self.buffer_size = buffer_size
        self.seed = seed

    def __iter__(self):
        rng = random.Random(self.seed)
        buffer = []
        dataset_iter = iter(self.dataset)

        try:
            for _ in range(self.buffer_size):
                buffer.append(next(dataset_iter))
        except StopIteration:
            pass  # dataset has fewer elements than buffer_size

        while buffer:
            idx = rng.randint(0, len(buffer) - 1)
            yield buffer[idx]
            try:
                buffer[idx] = next(dataset_iter)
            except StopIteration:
                buffer.pop(idx)


class EmbedDistillDataModule(LightningDataModule):
    def __init__(self, trn_ds: str, val_ds: str, tokenizer: str=None,  batch_size=64, num_workers=8):
        super().__init__()
        self.trn_ds = trn_ds
        self.val_ds = val_ds
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.tokenizer = tokenizer

        self.b_tokenizer = None  # Placeholder for tokenizer, to be set in setup
        self.b_trn_ds = None  # Placeholder for training dataset, to be set in setup
        self.b_val_ds = None  # Placeholder for validation dataset, to be set in setup

    @staticmethod
    def add_argparse_args(parent_parser, prefix="data"):
        local_parser = parent_parser.add_argument_group(f"{prefix} module")
        local_parser.add_argument(
            f"--{prefix}.trn_ds", type=str, required=True,
            help="Path to the training dataset in Parquet format."
        )
        local_parser.add_argument(
            f"--{prefix}.val_ds", type=str, required=True,
            help="Path to the validation dataset in Parquet format."
        )
        local_parser.add_argument(
            f"--{prefix}.tokenizer", type=str, default="sentence-transformers/all-mpnet-base-v2",
            help="Pretrained tokenizer to use for text processing."
        )
        local_parser.add_argument(
            f"--{prefix}.batch_size", type=int, default=64,
            help="Batch size for training and validation."
        )
        local_parser.add_argument(
            f"--{prefix}.num_workers", type=int, default=4,
            help="Number of workers for data loading."
        )
        return parent_parser

    def setup(self, stage=None):
        if self.b_tokenizer is None:
            self.b_tokenizer = PreTrainedTokenizerFast.from_pretrained(self.tokenizer)
            #AutoTokenizer.from_pretrained(self.tokenizer, use_fast=True)

        if stage == "fit" or stage is None:
            self.b_trn_ds = ParquetIterableDataset(self.trn_ds, tokenizer=None, max_length=256)
            self.b_trn_ds = BufferedShuffleDataset(self.b_trn_ds, buffer_size=20_000, seed=42)
            self.b_val_ds = ParquetIterableDataset(self.val_ds, tokenizer=None, max_length=256)

        if stage == "validate" or stage is None:
            self.b_val_ds = ParquetIterableDataset(self.val_ds, tokenizer=None, max_length=256)

    def train_dataloader(self):
        return DataLoader(
            self.b_trn_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=self._collate
        )

    def val_dataloader(self):
        return DataLoader(
            self.b_val_ds, batch_size=self.batch_size,
            shuffle=False, num_workers=self.num_workers,
            collate_fn=self._collate
        )

    def _collate(self, batch: list):
        if self.tokenizer is None:
            return {
                    "embedding": torch.stack([text["embedding"] for text in batch])
                }

        texts = [text["text"] for text in batch]
        tokens = self.b_tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=256,  # Fixed length for simplicity
            return_tensors="pt"
        )
        return {
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "embedding": torch.stack([text["embedding"] for text in batch])
        }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Embedding Distillation Data Module")
    parser = EmbedDistillDataModule.add_argparse_args(parser, prefix="data")

    args = parser.parse_args()
    print(args)

    data_module = EmbedDistillDataModule(
        trn_ds=getattr(args, "data.trn_ds"),
        val_ds=getattr(args, "data.val_ds"),
        tokenizer=getattr(args, "data.tokenizer"),
        batch_size=getattr(args, "data.batch_size"),
        num_workers=6#getattr(args, "data.num_workers"),
    )

    data_module.setup()  # Setup the data module

    from tqdm import tqdm

    for batch in tqdm(data_module.train_dataloader()):
        pass
        #print(f"Batch size: {len(batch['embedding'])}, Embedding shape: {batch['embedding'].shape}")

