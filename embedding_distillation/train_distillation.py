from pytorch_lightning.cli import LightningCLI
from DistillEmbedder import DistillEmbedder
from embedding_dataset import EmbedDistillDataModule


if __name__ == "__main__":
    # Initialize the Lightning CLI with the DistillEmbedder and EmbedDistillDataModule
    cli = LightningCLI(
        model_class=DistillEmbedder,
        datamodule_class=EmbedDistillDataModule,
        save_config_kwargs={"overwrite": True}
    )

