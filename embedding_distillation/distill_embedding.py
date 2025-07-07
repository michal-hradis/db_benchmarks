# train_distilled_embedder.py

import torch
from torch.utils.data import DataLoader
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Configuration
data_path = "my_data.parquet"        # Path to your Parquet dataset
student_model_name = "distilbert-base-multilingual-cased"
tokenizer_model = "distilbert-base-multilingual-cased"
batch_size = 32
epochs = 3
learning_rate = 2e-5
max_length = 512
shuffle_buffer_size = 100_000  # Buffer size for iterable shuffle

# 1) Load dataset in streaming (IterableDataset) mode and shuffle once
ds_iter = load_dataset(
    "parquet", data_files=data_path, split="train", streaming=True
)
# Perform streaming shuffle with fixed-size buffer (avoids random parquet seeks)
ds_iter = ds_iter.shuffle(buffer_size=shuffle_buffer_size, seed=42)

# 2) Initialize tokenizer and student model
tokenizer = AutoTokenizer.from_pretrained(tokenizer_model)
student = AutoModel.from_pretrained(student_model_name)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
student.to(device)

# 3) Define mean pooling for model outputs
def mean_pooling(token_embeds, attention_mask):
    mask = attention_mask.unsqueeze(-1).expand(token_embeds.size()).float()
    summed = torch.sum(token_embeds * mask, dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts

# 4) Custom collate to batch examples: tokenize online + load teacher embeddings
def collate_fn(examples):
    texts = [ex["text"] for ex in examples]
    embeds = [ex["embedding"] for ex in examples]
    # Tokenize on-the-fly
    toks = tokenizer(
        texts,
        padding="max_length",
        truncation=True,
        max_length=max_length,
        return_tensors="pt"
    )
    input_ids = toks.input_ids
    attention_mask = toks.attention_mask
    # Convert embeddings to float32 tensor batch
    teacher_embs = torch.stack([
        torch.tensor(vec, dtype=torch.float32) for vec in embeds
    ])
    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "teacher_emb": teacher_embs
    }

# 5) Prepare DataLoader over the IterableDataset
train_loader = DataLoader(
    ds_iter,
    batch_size=batch_size,
    collate_fn=collate_fn,
    num_workers=4,
    pin_memory=True
)

# 6) Training setup
optimizer = torch.optim.AdamW(student.parameters(), lr=learning_rate)
criterion = torch.nn.MSELoss()

# 7) Training loop
student.train()
for epoch in range(1, epochs + 1):
    total_loss = 0.0
    step = 0
    for batch in train_loader:
        step += 1
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        teacher_emb = batch["teacher_emb"].to(device)

        outputs = student(input_ids=input_ids, attention_mask=attention_mask)
        student_emb = mean_pooling(outputs.last_hidden_state, attention_mask)

        loss = criterion(student_emb, teacher_emb)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        total_loss += loss.item()
        if step % 100 == 0:
            print(f"Epoch {epoch} | Step {step} | Avg Loss {total_loss/step:.4f}")

    avg_loss = total_loss / step
    print(f"Epoch {epoch}/{epochs} completed - Avg Loss: {avg_loss:.4f}")

# 8) Save distilled model
student.save_pretrained("distilled-embedder")
