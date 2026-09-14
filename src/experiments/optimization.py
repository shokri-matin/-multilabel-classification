from dataclasses import dataclass

import torch
from torch import nn

from src.experiments.training_config import TrainingConfig


@dataclass
class OptimizationComponents:
    criterion: nn.Module
    optimizer: torch.optim.Optimizer


class OptimizationBuilder:
    @staticmethod
    def build_criterion() -> nn.Module:
        return nn.BCEWithLogitsLoss()

    @staticmethod
    def build_optimizer(
        model: nn.Module,
        config: TrainingConfig,
    ) -> torch.optim.Optimizer:
        if not isinstance(model, nn.Module):
            raise TypeError(
                "model must be an instance of torch.nn.Module"
            )

        if config.optimizer == "adamw":
            return torch.optim.AdamW(
                model.parameters(),
                lr=config.learning_rate,
                weight_decay=config.weight_decay,
            )

        raise ValueError(
            f"Unsupported optimizer: {config.optimizer}"
        )

    @classmethod
    def build(
        cls,
        model: nn.Module,
        config: TrainingConfig,
    ) -> OptimizationComponents:
        criterion = cls.build_criterion()

        optimizer = cls.build_optimizer(
            model=model,
            config=config,
        )

        return OptimizationComponents(
            criterion=criterion,
            optimizer=optimizer,
        )