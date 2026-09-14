import torch
from torch import nn

from src.classifier.dataloader import ClassifierBatch
from src.experiments.trainer import (
    TrainingHistory,
    TrainingManager,
    TrainingResult,
)
from src.experiments.training_config import TrainingConfig


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
        doc_ids=[
            f"doc_{i}"
            for i in range(batch_size)
        ],
        strategies=["degree"] * batch_size,
        modes=["no_branch"] * batch_size,
        central_nodes=["machine"] * batch_size,
        best_neighbors=["learning"] * batch_size,
    )


def make_dataloader(
    number_of_batches=2,
    batch_size=4,
):
    return [
        make_batch(batch_size)
        for _ in range(number_of_batches)
    ]


def create_manager(
    tmp_path,
    epochs=3,
    patience=2,
):
    model = SimpleClassifier()

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=4,
        epochs=epochs,
        early_stopping_patience=patience,
        checkpoint_dir=str(tmp_path),
        save_best_only=True,
        seed=42,
        device="cpu",
    )

    manager = TrainingManager(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        config=config,
        device=torch.device("cpu"),
    )

    return manager


def test_training_history():
    history = TrainingHistory()

    history.add(
        epoch=1,
        train_loss=0.6,
        validation_loss=0.5,
    )

    history.add(
        epoch=2,
        train_loss=0.5,
        validation_loss=0.4,
    )

    history.add(
        epoch=3,
        train_loss=0.4,
        validation_loss=0.45,
    )

    assert history.num_epochs == 3
    assert history.best_epoch == 2
    assert history.best_validation_loss == 0.4


def test_empty_training_history():
    history = TrainingHistory()

    assert history.num_epochs == 0
    assert history.best_epoch is None
    assert history.best_validation_loss is None


def test_resolve_cpu_device():
    device = TrainingManager.resolve_device(
        "cpu"
    )

    assert device == torch.device("cpu")


def test_resolve_auto_device():
    device = TrainingManager.resolve_device(
        "auto"
    )

    assert device.type in {
        "cpu",
        "cuda",
    }


def test_training_manager_fit(tmp_path):
    manager = create_manager(
        tmp_path=tmp_path,
        epochs=3,
        patience=2,
    )

    train_loader = make_dataloader(
        number_of_batches=2,
        batch_size=4,
    )

    validation_loader = make_dataloader(
        number_of_batches=2,
        batch_size=4,
    )

    result = manager.fit(
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
    )

    assert isinstance(
        result,
        TrainingResult,
    )

    assert result.history.num_epochs == 3

    assert result.best_epoch is not None

    assert result.best_validation_loss is not None

    assert result.best_validation_loss > 0

    assert result.checkpoint_path is not None


def test_best_checkpoint_exists(tmp_path):
    manager = create_manager(
        tmp_path=tmp_path,
        epochs=3,
        patience=2,
    )

    train_loader = make_dataloader()
    validation_loader = make_dataloader()

    result = manager.fit(
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
    )

    assert result.checkpoint_path is not None

    checkpoint_path = (
        tmp_path / "best.pt"
    )

    assert checkpoint_path.exists()


def test_training_history_has_matching_epochs(
    tmp_path,
):
    manager = create_manager(
        tmp_path=tmp_path,
        epochs=4,
        patience=2,
    )

    train_loader = make_dataloader()
    validation_loader = make_dataloader()

    result = manager.fit(
        train_dataloader=train_loader,
        validation_dataloader=validation_loader,
    )

    epochs = [
        item.epoch
        for item in result.history.epochs
    ]

    assert epochs == list(
        range(
            1,
            result.history.num_epochs + 1,
        )
    )


def test_model_is_on_configured_device(tmp_path):
    manager = create_manager(
        tmp_path=tmp_path,
    )

    for parameter in manager.model.parameters():
        assert parameter.device.type == "cpu"


def test_seed_reproducibility():
    TrainingManager.set_seed(42)

    first = torch.randn(5)

    TrainingManager.set_seed(42)

    second = torch.randn(5)

    assert torch.equal(
        first,
        second,
    )