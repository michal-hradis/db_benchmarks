import torch
import os
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig, AutoTokenizer
from pytorch_lightning import LightningModule
import torch.nn as nn


class ProjectedBert(nn.Module):
    def __init__(self,
                 model_name: str,
                 proj_dim: int,
                 hidden_dim: int = 1024,
                 pooling: str = "cls"  # one of "cls", "mean", "max"
                 ):
        super().__init__()
        pooling = pooling.lower()
        assert pooling in {"cls", "mean", "max"}
        self.pooling = pooling

        # load pretrained BERT
        self.bert = AutoModel.from_pretrained(model_name)
        self.hidden_size = self.bert.config.hidden_size

        # simple linear projection head
        self.proj1 = nn.Linear(self.hidden_size, hidden_dim)
        self.proj2 = nn.Linear(hidden_dim, proj_dim)

    def forward(self, input_ids, attention_mask=None, token_type_ids=None):
        # get last hidden states
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True,
        )
        # shape: (batch, seq_len, hidden_size)
        hidden = outputs.last_hidden_state

        if self.pooling == "cls":
            # take first token ([CLS]) representation
            rep = hidden[:, 0, :]
            rep = self.proj1(rep)  # (B, H)
            rep = F.softplus(rep)

        elif self.pooling == "mean":
            hidden = self.proj1(hidden)
            hidden = F.softplus(hidden)

            # mask padding tokens out of the mean
            mask = attention_mask.unsqueeze(-1)  # (B, T, 1)
            summed = (hidden * mask).sum(dim=1)  # (B, H)
            counts = mask.sum(dim=1).clamp(min=1e-9)  # (B, 1)
            rep = summed / counts  # (B, H)

        else:  # "max"
            hidden = self.proj1(hidden)
            hidden = F.softplus(hidden)

            # mask padding to a very low value before max
            mask = attention_mask.unsqueeze(-1).bool()
            neg_inf = torch.finfo(hidden.dtype).min
            hidden_masked = hidden.masked_fill(~mask, neg_inf)  # (B, T, H)
            rep, _ = hidden_masked.max(dim=1)  # (B, H)

        # project into your target dim
        return self.proj2(rep)


class DistillEmbedder(LightningModule):
    def __init__(self,
                 dim: int,
                 hidden_dim: int,
                 student: str, aggregation: str = "mean", teacher_projector: str = None,
                 lr=3e-5,
                 emb_loss_weight=0.5, emb_loss: str = "mse", emb_loss_norm: bool = True,
                 product_loss_weight=0.5, product_loss: str= "mse"):
        """Distill embeddings from a teacher model to a student model.
        Args:
            dim (int): Dimension of the embeddings to be produced by the student model.
            student (str): Pretrained student model name.
            aggregation (str): Aggregation method for embeddings, default is "mean". Options are "mean", "max", or "cls" (cls reads the [CLS] token).
            teacher_projector (str): Path to the teacher projector model (in TorchScript format). This model is used to project the teacher embeddings to the same space as the student embeddings.
            lr (float): Learning rate for training.
            emb_loss_weight (float): Weight for embedding loss.
            emb_loss (str): Type of embedding loss to use, default is "mse". Options are "mse" or "cosine". This determines how the student embeddings are compared to the teacher embeddings.
            emb_loss_norm (bool): Whether to normalize the embeddings before computing the loss.
            product_loss_weight (float): Weight for product loss. Computes  matrix of student dot products and teacher dot products, and forces them to be similar.
            product_loss (str): Type of product loss to use, default is "mse". Options are "l2" (Mean Squared Error) or "l1" (Mean Absolute Error).
        """

        super().__init__()
        assert aggregation in {"mean", "max", "cls"}, f"Unsupported aggregation method: {aggregation}"
        assert emb_loss in {"mse", "cosine"}, f"Unsupported embedding loss: {emb_loss}"
        assert product_loss in {"mse", "l2", "l1"}, f"Unsupported product loss: {product_loss}"

        self.save_hyperparameters()

        self.teacher_projector = torch.jit.load(teacher_projector, map_location="cpu").eval() if teacher_projector else None

        self.student = ProjectedBert(
            model_name=student,
            proj_dim=dim,
            hidden_dim=hidden_dim,
            pooling=aggregation
        )
        self.tokenizer = AutoTokenizer.from_pretrained(student)
        self.lr = lr
        self.emb_loss_weight = emb_loss_weight
        self.emb_loss = emb_loss
        self.emb_loss_norm = emb_loss_norm
        self.product_loss_weight = product_loss_weight
        self.product_loss = product_loss

    #def on_save_checkpoint(self, checkpoint: dict) -> None:
    #    """This hook is called whenever Lightning writes a .ckpt file."""
    #    save_dir = os.path.splitext(self.trainer.checkpoint_callback.best_model_path)[0]
    #    # ensure the directory exists
    #    os.makedirs(save_dir, exist_ok=True)
    #    # dump the huggingface student
    #    self.student.save_pretrained(os.path.join(save_dir, "student"))
    #    # if you have a tokenizer attribute, save it too
    #    self.tokenizer.save_pretrained(os.path.join(save_dir, "student"))

    def forward(self, input_ids, attention_mask):
        return self.student(input_ids=input_ids, attention_mask=attention_mask)

    def compute_emb_loss(self, s_emb, t_emb):
        """Compute the embedding loss between student and teacher embeddings."""
        if self.emb_loss_norm:
            s_emb = F.normalize(s_emb, p=2, dim=1)
            t_emb = F.normalize(t_emb, p=2, dim=1)

        if self.emb_loss == "mse":
            loss = F.mse_loss(s_emb, t_emb)
        elif self.emb_loss == "cosine":
            loss = 1 - F.cosine_similarity(s_emb, t_emb).mean()
        else:
            raise ValueError(f"Unsupported embedding loss: {self.emb_loss}")

        return loss

    def compute_product_loss(self, s_emb, t_emb):
        """Compute the product loss between student and teacher embeddings."""
        s_emb = F.normalize(s_emb, p=2, dim=1)
        t_emb = F.normalize(t_emb, p=2, dim=1)
        stud_products = torch.matmul(s_emb, s_emb.T)
        teacher_products = torch.matmul(t_emb, t_emb.T)
        if self.product_loss == "mse" or self.product_loss == "l2":
            loss = F.mse_loss(stud_products, teacher_products)
        elif self.product_loss == "l1":
            loss = F.l1_loss(stud_products, teacher_products)
        else:
            raise ValueError(f"Unsupported product loss: {self.product_loss}")
        return loss

    def training_step(self, batch, batch_idx):
        s_emb = self(batch["input_ids"], batch["attention_mask"])
        t_emb = batch["embedding"].float()
        if self.teacher_projector:
            t_emb = self.teacher_projector(t_emb)
        emb_loss = self.compute_emb_loss(s_emb, t_emb)
        product_loss = self.compute_product_loss(s_emb, t_emb)
        loss = self.emb_loss_weight * emb_loss + self.product_loss_weight * product_loss
        self.log_dict({"loss": loss, "product_loss": product_loss, "emb_loss": emb_loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s_emb = self(batch["input_ids"], batch["attention_mask"])
        t_emb = batch["embedding"].float()
        if self.teacher_projector:
            t_emb = self.teacher_projector(t_emb)
        emb_loss = self.compute_emb_loss(s_emb, t_emb)
        product_loss = self.compute_product_loss(s_emb, t_emb)
        loss = self.emb_loss_weight * emb_loss + self.product_loss_weight * product_loss
        self.log_dict({"val_loss": loss, "val_product_loss": product_loss, "val_emb_loss": emb_loss}, prog_bar=True)
        return {"val_loss": loss}

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.student.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distill embeddings from a teacher model to a student model.")
    parser.add_argument("--teacher-dim", type=int, required=True, help="Dimension of the teacher embeddings.")
    parser.add_argument("--student", type=str, required=True, help="Pretrained student model name.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for training.")

    args = parser.parse_args()

    distill_embedder = DistillEmbedder( student=args.student, lr=args.lr)
    print(distill_embedder)  # For testing purposes
