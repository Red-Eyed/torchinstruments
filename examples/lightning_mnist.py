"""Diagnose a real MNIST classifier through Lightning and TensorBoard."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from uuid import uuid4

import lightning as L
import torch
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

from torchinstruments import (
    CompositeSink,
    DirectorySink,
    EveryNForwardsSampler,
    MetricLoggerSink,
    inject_observer,
    remove_observer,
)


@dataclass(frozen=True)
class MnistRunConfig:
    """Define bounded data, training, sampling, and output settings for one run."""

    data_dir: Path
    telemetry_dir: Path
    log_dir: Path
    train_batches: int = 100
    validation_batches: int = 25
    epochs: int = 1
    batch_size: int = 64
    sample_every_n_forwards: int = 25

    def __post_init__(self) -> None:
        """Reject non-positive limits that would make the demonstration misleading."""
        limits = {
            "train_batches": self.train_batches,
            "validation_batches": self.validation_batches,
            "epochs": self.epochs,
            "batch_size": self.batch_size,
            "sample_every_n_forwards": self.sample_every_n_forwards,
        }
        for name, value in limits.items():
            if value <= 0:
                raise ValueError(f"{name} must be greater than zero")


class MnistClassifier(L.LightningModule):
    """Train a compact CNN while exposing its computational network for instrumentation."""

    def __init__(self) -> None:
        """Create convolutional feature extraction and a ten-class head."""
        super().__init__()
        self.network = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4, 4)),
            nn.Flatten(),
            nn.Linear(32 * 4 * 4, 10),
        )

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """Return class logits for one image batch."""
        return self.network(images)

    def training_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_index: int,
    ) -> torch.Tensor:
        """Optimize cross-entropy while logging the task-level training signals."""
        del batch_index
        loss, accuracy = self._loss_and_accuracy(batch)
        self.log("task/train_loss", loss, on_step=True, on_epoch=True, batch_size=batch[0].shape[0])
        self.log(
            "task/train_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=batch[0].shape[0],
        )
        return loss

    def validation_step(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
        batch_index: int,
    ) -> None:
        """Log validation loss and accuracy beside internal telemetry trends."""
        del batch_index
        loss, accuracy = self._loss_and_accuracy(batch)
        self.log(
            "task/validation_loss",
            loss,
            on_step=False,
            on_epoch=True,
            batch_size=batch[0].shape[0],
        )
        self.log(
            "task/validation_accuracy",
            accuracy,
            on_step=False,
            on_epoch=True,
            batch_size=batch[0].shape[0],
        )

    def configure_optimizers(self) -> torch.optim.Optimizer:
        """Configure the optimizer owned by Lightning's training loop."""
        return torch.optim.Adam(self.parameters(), lr=1e-3)

    def _loss_and_accuracy(
        self,
        batch: tuple[torch.Tensor, torch.Tensor],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute differentiable loss and batch accuracy from one labeled batch."""
        images, targets = batch
        logits = self(images)
        loss = nn.functional.cross_entropy(logits, targets)
        accuracy = (logits.argmax(dim=1) == targets).to(dtype=torch.float32).mean()
        return loss, accuracy


def build_mnist_loaders(config: MnistRunConfig) -> tuple[DataLoader, DataLoader]:
    """Download MNIST when needed and return shuffled train and stable validation loaders."""
    transform = transforms.ToTensor()
    train_dataset = MNIST(config.data_dir, train=True, download=True, transform=transform)
    validation_dataset = MNIST(config.data_dir, train=False, download=True, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True)
    validation_loader = DataLoader(validation_dataset, batch_size=config.batch_size)
    return train_loader, validation_loader


def run_training(
    config: MnistRunConfig,
    train_loader: DataLoader,
    validation_loader: DataLoader,
) -> Path:
    """Train with one logger shared by Lightning and TorchInstruments.

    The returned path contains TensorBoard event files. Full structured snapshots remain in the
    configured telemetry directory because scalar loggers are a lossy dashboard projection.
    """
    L.seed_everything(7, workers=True)
    model = MnistClassifier()
    logger = TensorBoardLogger(
        save_dir=config.log_dir,
        name="mnist-research",
        version="demo",
    )
    sink = CompositeSink(
        DirectorySink(config.telemetry_dir),
        MetricLoggerSink(logger),
    )

    # Instrument the network invoked by training_step; Lightning does not guarantee that the
    # outer LightningModule.forward method is the trainer's computational root.
    inject_observer(
        model.network,
        sampler=EveryNForwardsSampler(config.sample_every_n_forwards),
        sink=sink,
    )
    trainer = _build_trainer(config, logger)
    try:
        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=validation_loader,
        )
    finally:
        # The Trainer owns and finalizes logger; observer removal only detaches its sink view.
        remove_observer(model.network)

    return Path(logger.log_dir)


def _build_trainer(config: MnistRunConfig, logger: TensorBoardLogger) -> L.Trainer:
    """Build a quiet, bounded CPU trainer for a reproducible demonstration."""
    return L.Trainer(
        accelerator="cpu",
        devices=1,
        max_epochs=config.epochs,
        limit_train_batches=config.train_batches,
        limit_val_batches=config.validation_batches,
        num_sanity_val_steps=0,
        logger=logger,
        enable_checkpointing=False,
        enable_model_summary=False,
        enable_progress_bar=True,
        log_every_n_steps=1,
    )


def main() -> None:
    """Download MNIST and create collision-free JSON and TensorBoard outputs."""
    run_root = Path("stats") / f"lightning-mnist-{uuid4().hex[:8]}"
    config = MnistRunConfig(
        data_dir=Path("data") / "mnist",
        telemetry_dir=run_root / "telemetry",
        log_dir=run_root / "lightning",
    )
    train_loader, validation_loader = build_mnist_loaders(config)
    tensorboard_dir = run_training(config, train_loader, validation_loader)
    print(f"Structured telemetry written to {config.telemetry_dir}")
    print(f"TensorBoard events written to {tensorboard_dir}")


if __name__ == "__main__":
    main()
