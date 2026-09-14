import torch
from torch import nn

from src.experiments.checkpoint import (
    CheckpointManager,
    EarlyStopping,
    EarlyStoppingState,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(4, 2)

    def forward(self, x):
        return self.linear(x)


def create_model_and_optimizer():
    model = SimpleModel()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1e-3,
    )

    return model, optimizer


def test_early_stopping_initial_state():
    early_stopping = EarlyStopping(
        patience=2,
    )

    assert early_stopping.best_loss is None
    assert early_stopping.num_bad_epochs == 0
    assert early_stopping.stopped is False


def test_first_validation_loss_becomes_best():
    early_stopping = EarlyStopping(
        patience=2,
    )

    stopped = early_stopping.step(0.5)

    assert stopped is False
    assert early_stopping.best_loss == 0.5
    assert early_stopping.num_bad_epochs == 0


def test_improved_loss_resets_bad_epochs():
    early_stopping = EarlyStopping(
        patience=2,
    )

    early_stopping.step(0.5)
    early_stopping.step(0.6)

    assert early_stopping.num_bad_epochs == 1

    early_stopping.step(0.4)

    assert early_stopping.best_loss == 0.4
    assert early_stopping.num_bad_epochs == 0
    assert early_stopping.stopped is False


def test_patience_triggers_after_bad_epochs():
    early_stopping = EarlyStopping(
        patience=2,
    )

    assert early_stopping.step(0.5) is False
    assert early_stopping.step(0.6) is False
    assert early_stopping.step(0.7) is False

    assert early_stopping.num_bad_epochs == 2
    assert early_stopping.stopped is False

    assert early_stopping.step(0.8) is True

    assert early_stopping.num_bad_epochs == 3
    assert early_stopping.stopped is True


def test_min_delta_controls_improvement():
    early_stopping = EarlyStopping(
        patience=2,
        min_delta=0.05,
    )

    early_stopping.step(0.50)

    # Improvement is only 0.02.
    early_stopping.step(0.48)

    assert early_stopping.best_loss == 0.50
    assert early_stopping.num_bad_epochs == 1

    # Improvement is 0.10.
    early_stopping.step(0.40)

    assert early_stopping.best_loss == 0.40
    assert early_stopping.num_bad_epochs == 0


def test_checkpoint_manager_creates_directory(tmp_path):
    checkpoint_dir = tmp_path / "checkpoints"

    manager = CheckpointManager(
        checkpoint_dir=checkpoint_dir,
    )

    assert checkpoint_dir.exists()
    assert manager.best_checkpoint_path == (
        checkpoint_dir / "best.pt"
    )


def test_save_best_checkpoint(tmp_path):
    model, optimizer = create_model_and_optimizer()

    manager = CheckpointManager(
        checkpoint_dir=tmp_path,
    )

    result = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=3,
        validation_loss=0.42,
        is_best=True,
    )

    assert result.is_best is True
    assert result.epoch == 3
    assert result.validation_loss == 0.42

    assert result.path.exists()
    assert result.path == (
        tmp_path / "best.pt"
    )


def test_load_checkpoint(tmp_path):
    model, optimizer = create_model_and_optimizer()

    manager = CheckpointManager(
        checkpoint_dir=tmp_path,
    )

    original_parameters = {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
    }

    manager.save(
        model=model,
        optimizer=optimizer,
        epoch=2,
        validation_loss=0.35,
        is_best=True,
        training_state={
            "seed": 42,
        },
    )

    new_model, new_optimizer = (
        create_model_and_optimizer()
    )

    checkpoint = manager.load(
        path=tmp_path / "best.pt",
        model=new_model,
        optimizer=new_optimizer,
    )

    assert checkpoint["epoch"] == 2
    assert checkpoint["validation_loss"] == 0.35
    assert checkpoint["training_state"]["seed"] == 42

    for name, parameter in new_model.named_parameters():
        assert torch.equal(
            original_parameters[name],
            parameter,
        )


def test_save_best_only_does_not_save_regular_epoch(
    tmp_path,
):
    model, optimizer = create_model_and_optimizer()

    manager = CheckpointManager(
        checkpoint_dir=tmp_path,
        save_best_only=True,
    )

    result = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=4,
        validation_loss=0.5,
        is_best=False,
    )

    assert result.is_best is False

    assert not (
        tmp_path / "epoch_0004.pt"
    ).exists()


def test_save_all_epochs_when_disabled(tmp_path):
    model, optimizer = create_model_and_optimizer()

    manager = CheckpointManager(
        checkpoint_dir=tmp_path,
        save_best_only=False,
    )

    result = manager.save(
        model=model,
        optimizer=optimizer,
        epoch=4,
        validation_loss=0.5,
        is_best=False,
    )

    assert result.path.exists()

    assert result.path == (
        tmp_path / "epoch_0004.pt"
    )


def test_missing_checkpoint_raises(tmp_path):
    model, optimizer = create_model_and_optimizer()

    manager = CheckpointManager(
        checkpoint_dir=tmp_path,
    )

    try:
        manager.load(
            path=tmp_path / "missing.pt",
            model=model,
            optimizer=optimizer,
        )
        assert False
    except FileNotFoundError as exc:
        assert "Checkpoint not found" in str(exc)


def test_invalid_early_stopping_patience():
    try:
        EarlyStopping(patience=-1)
        assert False
    except ValueError as exc:
        assert "patience" in str(exc)


def test_invalid_checkpoint_epoch(tmp_path):
    model, optimizer = create_model_and_optimizer()

    manager = CheckpointManager(
        checkpoint_dir=tmp_path,
    )

    try:
        manager.save(
            model=model,
            optimizer=optimizer,
            epoch=-1,
            validation_loss=0.5,
            is_best=True,
        )
        assert False
    except ValueError as exc:
        assert "epoch" in str(exc)