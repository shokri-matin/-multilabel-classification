import pytest

from src.experiments.training_config import (
    TrainingConfig,
)


def test_default_training_config():
    config = TrainingConfig()

    assert config.learning_rate == 2e-5
    assert config.weight_decay == 0.01
    assert config.batch_size == 8
    assert config.epochs == 5
    assert config.gradient_accumulation_steps == 1
    assert config.max_grad_norm == 1.0
    assert config.optimizer == "adamw"
    assert config.seed == 42
    assert config.device == "auto"
    assert config.save_best_only is True
    assert config.early_stopping_patience == 2
    assert config.num_workers == 0


def test_custom_training_config():
    config = TrainingConfig(
        learning_rate=1e-4,
        weight_decay=0.05,
        batch_size=16,
        epochs=10,
        gradient_accumulation_steps=2,
        max_grad_norm=0.5,
        optimizer="adamw",
        seed=123,
        device="cuda",
        checkpoint_dir="results/checkpoints",
        save_best_only=False,
        early_stopping_patience=5,
        num_workers=2,
    )

    assert config.learning_rate == 1e-4
    assert config.weight_decay == 0.05
    assert config.batch_size == 16
    assert config.epochs == 10
    assert config.gradient_accumulation_steps == 2
    assert config.max_grad_norm == 0.5
    assert config.seed == 123
    assert config.device == "cuda"
    assert config.save_best_only is False
    assert config.early_stopping_patience == 5
    assert config.num_workers == 2


def test_checkpoint_path():
    config = TrainingConfig(
        checkpoint_dir=f"results/checkpoints"
    )

    assert str(config.checkpoint_path) == (
        f"results\\checkpoints"
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("learning_rate", 0),
        ("learning_rate", -1e-5),
        ("weight_decay", -0.01),
        ("batch_size", 0),
        ("epochs", 0),
        ("gradient_accumulation_steps", 0),
        ("max_grad_norm", 0),
        ("seed", -1),
        ("early_stopping_patience", -1),
        ("num_workers", -1),
    ],
)
def test_invalid_values(field, value):
    with pytest.raises(ValueError):
        TrainingConfig(**{field: value})


def test_empty_device():
    with pytest.raises(
        ValueError,
        match="device must not be empty",
    ):
        TrainingConfig(device="")


def test_empty_checkpoint_dir():
    with pytest.raises(
        ValueError,
        match="checkpoint_dir must not be empty",
    ):
        TrainingConfig(checkpoint_dir="")


def test_invalid_optimizer():
    with pytest.raises(
        ValueError,
        match="Unsupported optimizer",
    ):
        TrainingConfig(optimizer="sgd")


def test_config_is_immutable():
    config = TrainingConfig()

    with pytest.raises(AttributeError):
        config.learning_rate = 1e-4