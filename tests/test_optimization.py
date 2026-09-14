import pytest
import torch
from torch import nn

from src.experiments.optimization import (
    OptimizationBuilder,
    OptimizationComponents,
)
from src.experiments.training_config import (
    TrainingConfig,
)


class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.linear = nn.Linear(
            4,
            3,
        )

    def forward(self, x):
        return self.linear(x)


def test_build_criterion():
    criterion = (
        OptimizationBuilder.build_criterion()
    )

    assert isinstance(
        criterion,
        nn.BCEWithLogitsLoss,
    )


def test_bce_with_logits_loss():
    criterion = (
        OptimizationBuilder.build_criterion()
    )

    logits = torch.tensor(
        [
            [0.5, -0.5, 1.0],
            [-1.0, 2.0, -0.2],
        ]
    )

    labels = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
        ]
    )

    loss = criterion(
        logits,
        labels,
    )

    assert loss.ndim == 0
    assert torch.isfinite(loss)
    assert loss.item() > 0


def test_build_adamw():
    model = SimpleModel()

    config = TrainingConfig(
        learning_rate=1e-4,
        weight_decay=0.05,
    )

    optimizer = (
        OptimizationBuilder.build_optimizer(
            model=model,
            config=config,
        )
    )

    assert isinstance(
        optimizer,
        torch.optim.AdamW,
    )

    assert optimizer.defaults["lr"] == 1e-4
    assert optimizer.defaults["weight_decay"] == 0.05


def test_build_components():
    model = SimpleModel()

    config = TrainingConfig()

    components = OptimizationBuilder.build(
        model=model,
        config=config,
    )

    assert isinstance(
        components,
        OptimizationComponents,
    )

    assert isinstance(
        components.criterion,
        nn.BCEWithLogitsLoss,
    )

    assert isinstance(
        components.optimizer,
        torch.optim.AdamW,
    )


def test_optimizer_contains_model_parameters():
    model = SimpleModel()

    config = TrainingConfig()

    optimizer = (
        OptimizationBuilder.build_optimizer(
            model=model,
            config=config,
        )
    )

    model_parameter_ids = {
        id(parameter)
        for parameter in model.parameters()
    }

    optimizer_parameter_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }

    assert model_parameter_ids == (
        optimizer_parameter_ids
    )


def test_optimizer_learning_rate():
    model = SimpleModel()

    config = TrainingConfig(
        learning_rate=3e-5,
    )

    optimizer = (
        OptimizationBuilder.build_optimizer(
            model=model,
            config=config,
        )
    )

    assert optimizer.param_groups[0]["lr"] == 3e-5


def test_invalid_model():
    config = TrainingConfig()

    with pytest.raises(
        TypeError,
        match="model must be an instance",
    ):
        OptimizationBuilder.build_optimizer(
            model="not a model",
            config=config,
        )


def test_optimizer_updates_model():
    model = SimpleModel()

    config = TrainingConfig(
        learning_rate=1e-3,
    )

    components = OptimizationBuilder.build(
        model=model,
        config=config,
    )

    inputs = torch.randn(
        4,
        4,
    )

    labels = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    logits = model(inputs)

    loss = components.criterion(
        logits,
        labels,
    )

    loss.backward()

    before = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    components.optimizer.step()

    after = [
        parameter.detach().clone()
        for parameter in model.parameters()
    ]

    changed = any(
        not torch.equal(before_parameter, after_parameter)
        for before_parameter, after_parameter in zip(
            before,
            after,
        )
    )

    assert changed