from dataclasses import dataclass
from pathlib import Path
from typing import Literal


OptimizerName = Literal["adamw"]


@dataclass(frozen=True)
class TrainingConfig:
    learning_rate: float = 2e-5
    weight_decay: float = 0.01
    batch_size: int = 8
    epochs: int = 5
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0

    optimizer: OptimizerName = "adamw"

    seed: int = 42

    device: str = "auto"

    checkpoint_dir: str = f"results/checkpoints"

    save_best_only: bool = True

    early_stopping_patience: int = 2

    num_workers: int = 0

    def __post_init__(self):
        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.weight_decay < 0:
            raise ValueError(
                "weight_decay must be non-negative"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be positive"
            )

        if self.gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive"
            )

        if self.max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be positive"
            )

        if self.optimizer not in {"adamw"}:
            raise ValueError(
                f"Unsupported optimizer: {self.optimizer}"
            )

        if self.seed < 0:
            raise ValueError(
                "seed must be non-negative"
            )

        if not self.device:
            raise ValueError(
                "device must not be empty"
            )

        if not self.checkpoint_dir:
            raise ValueError(
                "checkpoint_dir must not be empty"
            )

        if self.early_stopping_patience < 0:
            raise ValueError(
                "early_stopping_patience must be "
                "non-negative"
            )

        if self.num_workers < 0:
            raise ValueError(
                "num_workers must be non-negative"
            )

    @property
    def checkpoint_path(self) -> Path:
        return Path(self.checkpoint_dir)