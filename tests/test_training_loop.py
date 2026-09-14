# tests/test_training_loop.py

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.classifier.dataloader import ClassifierBatch
from src.experiments.training_loop import (
    EpochResult,
    TrainingLoop,
)


class SimpleClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 3)

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
    ):
        pooled = inputs_embeds.mean(dim=1)
        return self.linear(pooled)


def make_batch(batch_size=4):
    return ClassifierBatch(
        inputs_embeds=torch.randn(
            batch_size,
            5,
            4,
        ),
        attention_mask=torch.ones(
            batch_size,
            5,
            dtype=torch.long,
        ),
        labels=torch.randint(
            0,
            2,
            (batch_size, 3),
        ).float(),
        doc_ids=[f"doc_{i}" for i in range(batch_size)],
        strategies=["degree"] * batch_size,
        modes=["no_branch"] * batch_size,
        central_nodes=["machine"] * batch_size,
        best_neighbors=["learning"] * batch_size,
    )


def make_dataloader(
    number_of_batches=3,
    batch_size=4,
):
    batches = [
        make_batch(batch_size)
        for _ in range(number_of_batches)
    ]

    return batches


def create_training_loop(
    gradient_accumulation_steps=1,
):
    model = SimpleClassifier()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    return TrainingLoop(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        device=torch.device("cpu"),
        gradient_accumulation_steps=gradient_accumulation_steps,
        max_grad_norm=1.0,
    )


def test_epoch_result():
    result = EpochResult(
        loss=0.5,
        steps=3,
        samples=12,
    )

    assert result.loss == 0.5
    assert result.steps == 3
    assert result.samples == 12


def test_train_one_epoch_returns_result():
    trainer = create_training_loop()

    dataloader = make_dataloader(
        number_of_batches=3,
        batch_size=4,
    )

    result = trainer.train_one_epoch(dataloader)

    assert isinstance(result, EpochResult)
    assert result.steps == 3
    assert result.samples == 12
    assert result.loss > 0
    assert torch.isfinite(
        torch.tensor(result.loss)
    )


def test_validate_one_epoch_returns_result():
    trainer = create_training_loop()

    dataloader = make_dataloader(
        number_of_batches=3,
        batch_size=4,
    )

    result = trainer.validate_one_epoch(dataloader)

    assert isinstance(result, EpochResult)
    assert result.steps == 3
    assert result.samples == 12
    assert result.loss > 0
    assert torch.isfinite(
        torch.tensor(result.loss)
    )


def test_training_updates_model_parameters():
    trainer = create_training_loop()

    dataloader = make_dataloader(
        number_of_batches=2,
        batch_size=4,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    trainer.train_one_epoch(dataloader)

    changed = False

    for name, parameter in trainer.model.named_parameters():
        if not torch.equal(before[name], parameter):
            changed = True
            break

    assert changed


def test_validation_does_not_update_model_parameters():
    trainer = create_training_loop()

    dataloader = make_dataloader(
        number_of_batches=2,
        batch_size=4,
    )

    before = {
        name: parameter.detach().clone()
        for name, parameter in trainer.model.named_parameters()
    }

    trainer.validate_one_epoch(dataloader)

    for name, parameter in trainer.model.named_parameters():
        assert torch.equal(
            before[name],
            parameter,
        )


def test_gradient_accumulation():
    trainer = create_training_loop(
        gradient_accumulation_steps=2,
    )

    dataloader = make_dataloader(
        number_of_batches=4,
        batch_size=4,
    )

    result = trainer.train_one_epoch(dataloader)

    assert result.steps == 4
    assert result.samples == 16


def test_empty_training_dataloader_raises():
    trainer = create_training_loop()

    empty_dataloader = []

    try:
        trainer.train_one_epoch(empty_dataloader)
        assert False
    except ValueError as exc:
        assert "zero samples" in str(exc)


def test_empty_validation_dataloader_raises():
    trainer = create_training_loop()

    empty_dataloader = []

    try:
        trainer.validate_one_epoch(empty_dataloader)
        assert False
    except ValueError as exc:
        assert "zero samples" in str(exc)