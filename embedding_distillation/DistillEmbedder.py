import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from pytorch_lightning import LightningModule

class DistillEmbedder(LightningModule):
    def __init__(self, teacher_dim, student: str, lr=3e-5, emb_loss_weight=0.5, product_loss_weight=0.5):
        super().__init__()
        self.save_hyperparameters()
        self.student = AutoModel.from_pretrained(student)
        self.proj = torch.nn.Linear(self.student.config.hidden_size, teacher_dim, bias=False)
        self.lr = lr
        self.emb_loss_weight = emb_loss_weight
        self.product_loss_weight = product_loss_weight

    def forward(self, input_ids, attention_mask):
        out = self.student(input_ids=input_ids, attention_mask=attention_mask)
        # mean-pool
        hid = out.last_hidden_state
        mask = attention_mask.unsqueeze(-1)
        emb = (hid * mask).sum(1) / mask.sum(1)
        emb = self.proj(emb)
        return F.normalize(emb, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        s_emb = self(batch["input_ids"], batch["attention_mask"])
        t_emb = F.normalize(batch["embedding"].float(), p=2, dim=1)
        emb_loss = F.mse_loss(s_emb, t_emb)

        teacher_products = torch.matmul(t_emb, t_emb.T)
        student_products = torch.matmul(s_emb, s_emb.T)
        product_loss = F.mse_loss(student_products, teacher_products)

        loss = self.emb_loss_weight * emb_loss + self.product_loss_weight * product_loss

        self.log_dict({"loss": loss, "product_loss": product_loss, "emb_loss": emb_loss}, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr)
        return optimizer


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Distill embeddings from a teacher model to a student model.")
    parser.add_argument("--teacher-dim", type=int, required=True, help="Dimension of the teacher embeddings.")
    parser.add_argument("--student", type=str, required=True, help="Pretrained student model name.")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate for training.")

    args = parser.parse_args()

    distill_embedder = DistillEmbedder(teacher_dim=args.teacher_dim, student=args.student, lr=args.lr)
    print(distill_embedder)  # For testing purposes
