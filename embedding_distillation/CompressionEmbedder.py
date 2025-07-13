import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from pytorch_lightning import LightningModule
import os


class CompressionEmbedder(LightningModule):
    def __init__(self, teacher_dim, layers: int = 2, dim: int = 512, lr=3e-5):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.teacher_dim = teacher_dim
        self.dim = dim
        l = [torch.nn.Linear(teacher_dim, dim)]
        for i in range(layers -1):
            l.append(torch.nn.Softplus())
            l.append(torch.nn.Linear(dim, dim))
        self.model = torch.nn.Sequential(*l)

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        """This hook is called whenever Lightning writes a .ckpt file."""
        save_dir = os.path.splitext(self.trainer.checkpoint_callback.best_model_path)[0]
        os.makedirs(save_dir, exist_ok=True)
        example_input = torch.randn(1, self.teacher_dim, device=self.device)
        ts_model = torch.jit.trace(self.model, example_input)
        ts_model.save(os.path.join(save_dir, f'ts_model_{self.teacher_dim}_{self.dim}.pt'))

    def forward(self, x):
        out = self.model(x)
        return F.normalize(out, p=2, dim=1)

    def training_step(self, batch, batch_idx):
        s_emb = self(batch["embedding"])
        t_emb = F.normalize(batch["embedding"].float(), p=2, dim=1)

        teacher_products = torch.matmul(t_emb, t_emb.T)
        student_products = torch.matmul(s_emb, s_emb.T)
        #print(t_emb.shape, s_emb.shape, teacher_products.shape, student_products.shape, teacher_products.max().item(), teacher_products.min().item(), student_products.max().item(), student_products.min().item(), student_products, teacher_products, s_emb, t_emb)
        loss = F.mse_loss(student_products, teacher_products)


        self.log_dict({"loss": loss}, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        s_emb = self(batch["embedding"])
        t_emb = F.normalize(batch["embedding"].float(), p=2, dim=1)

        teacher_products = torch.matmul(t_emb, t_emb.T)
        student_products = torch.matmul(s_emb, s_emb.T)
        loss = F.mse_loss(student_products, teacher_products)

        self.log_dict({"val_loss": loss}, prog_bar=True)
        return {"val_loss": loss}

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
