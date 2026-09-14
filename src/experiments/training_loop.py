# src/experiments/training_loop.py

from dataclasses import dataclass

import torch
from torch import nn

from src.classifier.dataloader import ClassifierBatch


@dataclass(frozen=True)
class EpochResult:
    loss: float
    steps: int
    samples: int


class TrainingLoop:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: torch.device,
        gradient_accumulation_steps: int = 1,
        max_grad_norm: float = 1.0,
    ):
        if not isinstance(model, nn.Module):
            raise TypeError(
                "model must be an instance of torch.nn.Module"
            )

        if not isinstance(criterion, nn.Module):
            raise TypeError(
                "criterion must be an instance of torch.nn.Module"
            )

        if not isinstance(optimizer, torch.optim.Optimizer):
            raise TypeError(
                "optimizer must be a torch.optim.Optimizer"
            )

        if gradient_accumulation_steps <= 0:
            raise ValueError(
                "gradient_accumulation_steps must be positive"
            )

        if max_grad_norm <= 0:
            raise ValueError(
                "max_grad_norm must be positive"
            )

        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.gradient_accumulation_steps = (
            gradient_accumulation_steps
        )
        self.max_grad_norm = max_grad_norm

        self.model.to(self.device)

    def _move_batch_to_device(
        self,
        batch: ClassifierBatch,
    ) -> ClassifierBatch:
        return ClassifierBatch(
            inputs_embeds=batch.inputs_embeds.to(self.device),
            attention_mask=batch.attention_mask.to(self.device),
            labels=batch.labels.to(self.device),
            doc_ids=batch.doc_ids,
            strategies=batch.strategies,
            modes=batch.modes,
            central_nodes=batch.central_nodes,
            best_neighbors=batch.best_neighbors,
        )

    def train_one_epoch(
        self,
        dataloader,
    ) -> EpochResult:
        self.model.train()

        self.optimizer.zero_grad(set_to_none=True)

        total_loss = 0.0
        total_samples = 0
        steps = 0

        number_of_batches = len(dataloader)

        for batch_index, batch in enumerate(dataloader):
            batch = self._move_batch_to_device(batch)

            logits = self.model(
                inputs_embeds=batch.inputs_embeds,
                attention_mask=batch.attention_mask,
            )

            loss = self.criterion(
                logits,
                batch.labels,
            )

            loss_for_backward = (
                loss / self.gradient_accumulation_steps
            )

            loss_for_backward.backward()

            is_accumulation_boundary = (
                (batch_index + 1)
                % self.gradient_accumulation_steps
                == 0
            )

            is_last_batch = (
                batch_index + 1 == number_of_batches
            )

            if is_accumulation_boundary or is_last_batch:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.max_grad_norm,
                )

                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)

            batch_size = batch.inputs_embeds.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            steps += 1

        if total_samples == 0:
            raise ValueError(
                "Dataloader produced zero samples"
            )

        return EpochResult(
            loss=total_loss / total_samples,
            steps=steps,
            samples=total_samples,
        )

    @torch.no_grad()
    def validate_one_epoch(
        self,
        dataloader,
    ) -> EpochResult:
        self.model.eval()

        total_loss = 0.0
        total_samples = 0
        steps = 0

        for batch in dataloader:
            batch = self._move_batch_to_device(batch)

            logits = self.model(
                inputs_embeds=batch.inputs_embeds,
                attention_mask=batch.attention_mask,
            )

            loss = self.criterion(
                logits,
                batch.labels,
            )

            batch_size = batch.inputs_embeds.shape[0]

            total_loss += loss.item() * batch_size
            total_samples += batch_size
            steps += 1

        if total_samples == 0:
            raise ValueError(
                "Dataloader produced zero samples"
            )

        return EpochResult(
            loss=total_loss / total_samples,
            steps=steps,
            samples=total_samples,
        )