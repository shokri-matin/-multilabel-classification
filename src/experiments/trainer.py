from dataclasses import dataclass, field
from typing import List


@dataclass(frozen=True)
class EpochMetrics:
    epoch: int
    train_loss: float
    validation_loss: float


@dataclass
class TrainingHistory:
    epochs: List[EpochMetrics] = field(
        default_factory=list
    )

    def add(
        self,
        epoch: int,
        train_loss: float,
        validation_loss: float,
    ) -> None:
        self.epochs.append(
            EpochMetrics(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
            )
        )

    @property
    def best_epoch(self) -> int | None:
        if not self.epochs:
            return None

        best = min(
            self.epochs,
            key=lambda item: item.validation_loss,
        )

        return best.epoch

    @property
    def best_validation_loss(self) -> float | None:
        if not self.epochs:
            return None

        return min(
            item.validation_loss
            for item in self.epochs
        )

    @property
    def num_epochs(self) -> int:
        return len(self.epochs)


@dataclass(frozen=True)
class TrainingResult:
    history: TrainingHistory
    best_epoch: int | None
    best_validation_loss: float | None
    stopped_early: bool
    checkpoint_path: str | None


import random

import numpy as np
import torch
from torch import nn

from src.experiments.checkpoint import (
    CheckpointManager,
    EarlyStopping,
)
from src.experiments.training_config import TrainingConfig
from src.experiments.training_loop import TrainingLoop

class TrainingManager:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        config: TrainingConfig,
        device: torch.device | None = None,
    ):
        if device is None:
            device = self.resolve_device(
                config.device
            )

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.config = config
        self.device = device

        self.set_seed(config.seed)

        self.training_loop = TrainingLoop(
            model=model,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            gradient_accumulation_steps=(
                config.gradient_accumulation_steps
            ),
            max_grad_norm=config.max_grad_norm,
        )

        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint_dir,
            save_best_only=config.save_best_only,
        )

        self.early_stopping = EarlyStopping(
            patience=config.early_stopping_patience,
        )

    @staticmethod
    def resolve_device(
        device: str,
    ) -> torch.device:
        if not device:
            raise ValueError(
                "device must not be empty"
            )

        if device == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return torch.device(device)

    @staticmethod
    def set_seed(seed: int) -> None:
        if seed < 0:
            raise ValueError(
                "seed must be non-negative"
            )

        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(
        self,
        train_dataloader,
        validation_dataloader,
    ) -> TrainingResult:

        history = TrainingHistory()

        best_checkpoint_path = None
        stopped_early = False

        for epoch in range(1, self.config.epochs + 1):

            train_result = (
                self.training_loop.train_one_epoch(
                    train_dataloader
                )
            )

            validation_result = (
                self.training_loop.validate_one_epoch(
                    validation_dataloader
                )
            )

            train_loss = train_result.loss
            validation_loss = validation_result.loss

            history.add(
                epoch=epoch,
                train_loss=train_loss,
                validation_loss=validation_loss,
            )

            is_best = (
                self.early_stopping.best_loss is None
                or validation_loss
                < self.early_stopping.best_loss
            )

            checkpoint_result = (
                self.checkpoint_manager.save(
                    model=self.model,
                    optimizer=self.optimizer,
                    epoch=epoch,
                    validation_loss=validation_loss,
                    is_best=is_best,
                    training_state={
                        "train_loss": train_loss,
                        "validation_loss": validation_loss,
                        "early_stopping_bad_epochs": (
                            self.early_stopping.num_bad_epochs
                        ),
                    },
                )
            )

            if is_best:
                best_checkpoint_path = str(
                    checkpoint_result.path
                )

            should_stop = self.early_stopping.step(
                validation_loss
            )

            if should_stop:
                stopped_early = True
                break

        return TrainingResult(
            history=history,
            best_epoch=history.best_epoch,
            best_validation_loss=(
                history.best_validation_loss
            ),
            stopped_early=stopped_early,
            checkpoint_path=best_checkpoint_path,
        )