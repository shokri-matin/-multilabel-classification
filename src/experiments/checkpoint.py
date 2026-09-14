# src/experiments/checkpoint.py

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

@dataclass
class EarlyStoppingState:
    best_loss: float | None = None
    num_bad_epochs: int = 0
    stopped: bool = False


class EarlyStopping:
    def __init__(
        self,
        patience: int = 2,
        min_delta: float = 0.0,
    ):
        if patience < 0:
            raise ValueError(
                "patience must be non-negative"
            )

        if min_delta < 0:
            raise ValueError(
                "min_delta must be non-negative"
            )

        self.patience = patience
        self.min_delta = min_delta

        self.state = EarlyStoppingState()

    def step(
        self,
        validation_loss: float,
    ) -> bool:

        if validation_loss < 0:
            raise ValueError(
                "validation_loss must be non-negative"
            )

        if self.state.best_loss is None:
            self.state.best_loss = validation_loss
            self.state.num_bad_epochs = 0
            self.state.stopped = False

            return False

        improvement = (
            self.state.best_loss
            - validation_loss
        )

        if improvement > self.min_delta:
            self.state.best_loss = validation_loss
            self.state.num_bad_epochs = 0
            self.state.stopped = False

            return False

        self.state.num_bad_epochs += 1

        if self.state.num_bad_epochs > self.patience:
            self.state.stopped = True

        return self.state.stopped

    @property
    def best_loss(self) -> float | None:
        return self.state.best_loss

    @property
    def num_bad_epochs(self) -> int:
        return self.state.num_bad_epochs

    @property
    def stopped(self) -> bool:
        return self.state.stopped


@dataclass(frozen=True)
class CheckpointResult:
    path: Path
    epoch: int
    validation_loss: float
    is_best: bool


class CheckpointManager:
    def __init__(
        self,
        checkpoint_dir: str | Path,
        save_best_only: bool = True,
    ):
        if not checkpoint_dir:
            raise ValueError(
                "checkpoint_dir must not be empty"
            )

        self.checkpoint_dir = Path(checkpoint_dir)
        self.save_best_only = save_best_only

        self.checkpoint_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.best_checkpoint_path = (
            self.checkpoint_dir / "best.pt"
        )

    def save(
        self,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        validation_loss: float,
        is_best: bool,
        training_state: dict[str, Any] | None = None,
    ) -> CheckpointResult:

        if epoch < 0:
            raise ValueError(
                "epoch must be non-negative"
            )

        if validation_loss < 0:
            raise ValueError(
                "validation_loss must be non-negative"
            )

        if is_best:
            path = self.best_checkpoint_path
        elif self.save_best_only:
            return CheckpointResult(
                path=self.best_checkpoint_path,
                epoch=epoch,
                validation_loss=validation_loss,
                is_best=False,
            )
        else:
            path = (
                self.checkpoint_dir
                / f"epoch_{epoch:04d}.pt"
            )

        checkpoint = {
            "epoch": epoch,
            "validation_loss": validation_loss,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "training_state": training_state or {},
        }

        torch.save(
            checkpoint,
            path,
        )

        return CheckpointResult(
            path=path,
            epoch=epoch,
            validation_loss=validation_loss,
            is_best=is_best,
        )

    def load(
        self,
        path: str | Path,
        model: torch.nn.Module,
        optimizer: torch.optim.Optimizer | None = None,
        map_location: str | torch.device = "cpu",
    ) -> dict[str, Any]:

        path = Path(path)

        if not path.exists():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}"
            )

        checkpoint = torch.load(
            path,
            map_location=map_location,
        )

        if "model_state_dict" not in checkpoint:
            raise ValueError(
                "Checkpoint does not contain "
                "'model_state_dict'"
            )

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        if (
            optimizer is not None
            and "optimizer_state_dict" in checkpoint
        ):
            optimizer.load_state_dict(
                checkpoint["optimizer_state_dict"]
            )

        return checkpoint