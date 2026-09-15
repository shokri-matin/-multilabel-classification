from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.classifier.dataloader import ClassifierDataLoader
from src.classifier.dataset import (
    ClassifierDataset,
    LabelEncoder,
)
from src.embedding.base import EmbeddingOutput
from src.experiments.application import ExperimentApplication
from src.experiments.config import ExperimentConfig
from src.experiments.executor import ExperimentExecutor
from src.experiments.experiment_context import ExperimentData
from src.experiments.runner import ExperimentRunner
from src.experiments.training_config import TrainingConfig


class TinyClassifier(nn.Module):
    """
    Small classifier used only for integration testing.

    Input:
        (batch, sequence_length, input_dim)

    Output:
        (batch, num_labels)
    """

    def __init__(
        self,
        input_dim: int,
        num_labels: int,
    ):
        super().__init__()

        self.projection = nn.Linear(
            input_dim,
            8,
        )

        self.classifier = nn.Linear(
            8,
            num_labels,
        )

    def forward(
        self,
        inputs_embeds,
        attention_mask=None,
    ):
        hidden = self.projection(
            inputs_embeds
        )

        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1).to(
                hidden.dtype
            )

            hidden = (
                hidden * mask
            ).sum(dim=1)

            denominator = mask.sum(
                dim=1
            ).clamp(min=1.0)

            hidden = hidden / denominator

        else:
            hidden = hidden.mean(dim=1)

        return self.classifier(hidden)


class FakeExperimentContext:
    """
    Minimal context for integration testing.

    The real ExperimentContext is already tested separately.
    Here we want to test the execution lifecycle around it.
    """

    def __init__(
        self,
        experiment,
        experiment_data,
        document_labels,
    ):
        self.experiment = experiment

        self._experiment_data = (
            experiment_data
        )

        self.dataset_context = type(
            "FakeDatasetContext",
            (),
            {
                "document_labels": document_labels,
            },
        )()

    def build(self):
        return self._experiment_data


def create_embedding(
    doc_id: str,
    label_type: str,
    input_dim: int = 4,
    sequence_length: int = 5,
):
    rng = np.random.default_rng(
        seed=abs(hash(doc_id)) % (2**32)
    )

    embeddings = rng.normal(
        size=(
            sequence_length,
            input_dim,
        )
    ).astype(np.float32)

    words = [
        "word1",
        "word2",
        "word3",
        "word4",
        "<PAD>",
    ]

    return EmbeddingOutput(
        doc_id=doc_id,
        strategy="closeness",
        mode="no_branch",
        central_node="central",
        best_neighbor="neighbor",
        words=words,
        embeddings=embeddings,
        embedding_dim=input_dim,
        sequence_length=sequence_length,
    )


def create_dataset(
    doc_ids,
    document_labels,
    label_vocabulary,
):
    embeddings = []

    for doc_id in doc_ids:
        label = document_labels[doc_id][0]

        embeddings.append(
            create_embedding(
                doc_id=doc_id,
                label_type=label,
            )
        )

    label_encoder = LabelEncoder(
        label_vocabulary
    )

    return ClassifierDataset(
        embeddings=embeddings,
        document_labels=document_labels,
        label_encoder=label_encoder,
    )


def create_experiment():
    return ExperimentConfig(
        dataset="aapd",
        mode="no_branch",
        centrality="closeness",
        embedding="bert",
        classifier="bert",
        top_k=200,
        max_nodes=20,
        sequence_length=5,
    )


def create_experiment_data(
    experiment,
):
    label_vocabulary = [
        "label_a",
        "label_b",
    ]

    train_labels = {
        "train_1": ["label_a"],
        "train_2": ["label_b"],
        "train_3": ["label_a"],
        "train_4": ["label_b"],
    }

    validation_labels = {
        "validation_1": ["label_a"],
        "validation_2": ["label_b"],
    }

    test_labels = {
        "test_1": ["label_a"],
        "test_2": ["label_b"],
        "test_3": ["label_a"],
        "test_4": ["label_b"],
    }

    train_dataset = create_dataset(
        list(train_labels),
        train_labels,
        label_vocabulary,
    )

    validation_dataset = create_dataset(
        list(validation_labels),
        validation_labels,
        label_vocabulary,
    )

    test_dataset = create_dataset(
        list(test_labels),
        test_labels,
        label_vocabulary,
    )

    train_loader = ClassifierDataLoader(
        dataset=train_dataset,
        batch_size=2,
        shuffle=True,
    )

    validation_loader = ClassifierDataLoader(
        dataset=validation_dataset,
        batch_size=2,
        shuffle=False,
    )

    test_loader = ClassifierDataLoader(
        dataset=test_dataset,
        batch_size=2,
        shuffle=False,
    )

    return ExperimentData(
        experiment=experiment,
        train_dataset=train_dataset,
        validation_dataset=validation_dataset,
        test_dataset=test_dataset,
        train_loader=train_loader,
        validation_loader=validation_loader,
        test_loader=test_loader,
        num_labels=2,
        input_dim=4,
        sequence_length=5,
    ), {
        "train": train_labels,
        "validation": validation_labels,
        "test": test_labels,
    }


def test_full_experiment_execution(
    tmp_path,
    monkeypatch,
):
    experiment = create_experiment()

    experiment_data, document_labels = (
        create_experiment_data(
            experiment
        )
    )

    context = FakeExperimentContext(
        experiment=experiment,
        experiment_data=experiment_data,
        document_labels=document_labels,
    )

    training_config = TrainingConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        batch_size=2,
        epochs=2,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        optimizer="adamw",
        seed=42,
        device="cpu",
        checkpoint_dir=(
            tmp_path / "checkpoints"
        ),
        save_best_only=True,
        early_stopping_patience=2,
        num_workers=0,
    )

    def fake_create(
        classifier,
        input_dim,
        num_labels,
        dropout=0.1,
    ):
        assert classifier == "bert"

        return TinyClassifier(
            input_dim=input_dim,
            num_labels=num_labels,
        )

    monkeypatch.setattr(
        "src.experiments.executor.ClassifierFactory.create",
        fake_create,
    )

    executor = ExperimentExecutor(
        context=context,
        training_config=training_config,
    )

    result = executor.execute()

    assert result.status == "completed", result.error

    assert (
        result.experiment_id
        == experiment.experiment_id
    )

    assert result.training_epochs > 0

    assert result.best_epoch is not None

    assert (
        result.validation_loss is not None
    )

    assert (
        result.micro_f1 is not None
    )

    assert (
        result.macro_f1 is not None
    )

    assert (
        result.hamming_loss is not None
    )

    assert (
        result.checkpoint_path is not None
    )

    checkpoint_path = Path(
        result.checkpoint_path
    )

    assert checkpoint_path.exists()

    assert checkpoint_path.name == "best.pt"


def test_full_application_persists_result(
    tmp_path,
    monkeypatch,
):
    experiment = create_experiment()

    experiment_data, document_labels = (
        create_experiment_data(
            experiment
        )
    )

    context = FakeExperimentContext(
        experiment=experiment,
        experiment_data=experiment_data,
        document_labels=document_labels,
    )

    training_config = TrainingConfig(
        learning_rate=1e-2,
        weight_decay=0.0,
        batch_size=2,
        epochs=2,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        optimizer="adamw",
        seed=42,
        device="cpu",
        checkpoint_dir=(
            tmp_path / "checkpoints"
        ),
        save_best_only=True,
        early_stopping_patience=2,
        num_workers=0,
    )

    def fake_create(
        classifier,
        input_dim,
        num_labels,
        dropout=0.1,
    ):
        return TinyClassifier(
            input_dim=input_dim,
            num_labels=num_labels,
        )

    monkeypatch.setattr(
        "src.experiments.executor.ClassifierFactory.create",
        fake_create,
    )

    def executor_factory(config):
        return ExperimentExecutor(
            context=context,
            training_config=training_config,
        )

    runner = ExperimentRunner(
        artifact_dir=(
            tmp_path / "experiments"
        )
    )

    application = ExperimentApplication(
        runner=runner,
        executor_factory=executor_factory,
    )

    result = application.run(
        experiment=experiment,
        skip_completed=True,
    )

    assert result.status == "completed", result.error

    result_path = (
        tmp_path
        / "experiments"
        / f"{experiment.experiment_id}.json"
    )

    assert result_path.exists()

    # Verify that the persisted result can
    # be loaded again through ExperimentRunner.
    loaded = runner.load_result(
        experiment
    )

    assert (
        loaded.experiment_id
        == experiment.experiment_id
    )

    assert loaded.status == "completed", result.error

    # A second run must be skipped because
    # the completed result already exists.
    second_result = application.run(
        experiment=experiment,
        skip_completed=True,
    )

    assert (
        second_result.experiment_id
        == experiment.experiment_id
    )

    assert second_result.status == "completed", second_result.error