from dataclasses import dataclass
from typing import Dict

from src.classifier.dataset import (
    ClassifierDataset,
    LabelEncoder,
)
from src.classifier.dataloader import ClassifierDataLoader
from src.embedding.base import BaseEmbedder
from src.experiments.config import ExperimentConfig
from src.experiments.dataset_context import DatasetContext
from src.experiments.embedding_context import (
    DatasetEmbeddingContext,
)
from src.experiments.sequence_context import (
    DatasetSequenceContext,
)


@dataclass
class ExperimentData:
    """
    Prepared data required by one experiment.
    """

    experiment: ExperimentConfig

    train_dataset: ClassifierDataset
    validation_dataset: ClassifierDataset
    test_dataset: ClassifierDataset

    train_loader: ClassifierDataLoader
    validation_loader: ClassifierDataLoader
    test_loader: ClassifierDataLoader

    num_labels: int
    input_dim: int
    sequence_length: int


class ExperimentContext:
    """
    Resolved data context for one experiment.

    This class connects:

        DatasetContext
            ↓
        SequenceContext
            ↓
        EmbeddingContext
            ↓
        ClassifierDataset
            ↓
        ClassifierDataLoader
    """

    def __init__(
        self,
        dataset_context: DatasetContext,
        sequence_context: DatasetSequenceContext,
        embedding_context: DatasetEmbeddingContext,
        experiment: ExperimentConfig,
        embedder: BaseEmbedder,
        batch_size: int = 8,
        shuffle_train: bool = True,
    ):
        self.dataset_context = dataset_context
        self.sequence_context = sequence_context
        self.embedding_context = embedding_context
        self.experiment = experiment
        self.embedder = embedder
        self.batch_size = batch_size
        self.shuffle_train = shuffle_train

        self._validate()

    def _validate(self) -> None:
        experiment = self.experiment

        if self.dataset_context.dataset != experiment.dataset:
            raise ValueError(
                "dataset_context dataset does not match experiment dataset"
            )

        if self.sequence_context.dataset != experiment.dataset:
            raise ValueError(
                "sequence_context dataset does not match experiment dataset"
            )

        if self.embedding_context.dataset != experiment.dataset:
            raise ValueError(
                "embedding_context dataset does not match experiment dataset"
            )

        if self.sequence_context.train.top_k != experiment.top_k:
            raise ValueError(
                "train sequence_context top_k does not match experiment top_k"
            )

        if (
            self.sequence_context.validation.top_k
            != experiment.top_k
        ):
            raise ValueError(
                "validation sequence_context top_k does not match experiment top_k"
            )

        if self.sequence_context.test.top_k != experiment.top_k:
            raise ValueError(
                "test sequence_context top_k does not match experiment top_k"
            )

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if experiment.embedding not in {"bert", "fasttext"}:
            raise ValueError(
                f"Unsupported embedding: {experiment.embedding}"
            )

        if experiment.classifier not in {"bert", "xlnet"}:
            raise ValueError(
                f"Unsupported classifier: {experiment.classifier}"
            )

        if self.embedder.embedding_dim <= 0:
            raise ValueError(
                "embedder dimension must be positive"
            )

    def _collect_embeddings(
        self,
        split: str,
    ):
        split_context = self.embedding_context.split(
            split
        )

        document_embeddings = []

        for document in split_context.iter_documents(
            strategy=self.experiment.centrality,
            mode=self.experiment.mode,
            embedding=self.experiment.embedding,
            embedder=self.embedder,
        ):
            document_embeddings.extend(
                document.outputs
            )

        if not document_embeddings:
            raise ValueError(
                f"No embeddings generated for "
                f"split '{split}'"
            )

        return document_embeddings

    def build(
        self,
    ) -> ExperimentData:

        train_embeddings = self._collect_embeddings(
            "train"
        )

        validation_embeddings = self._collect_embeddings(
            "validation"
        )

        test_embeddings = self._collect_embeddings(
            "test"
        )

        label_encoder = LabelEncoder(
            self.dataset_context.label_vocabulary
        )

        train_dataset = ClassifierDataset(
            embeddings=train_embeddings,
            document_labels=self.dataset_context.labels_for_split("train"),
            label_encoder=label_encoder,
        )

        validation_dataset = ClassifierDataset(
            embeddings=validation_embeddings,
            document_labels=self.dataset_context.labels_for_split("validation"),
            label_encoder=label_encoder,
        )

        test_dataset = ClassifierDataset(
            embeddings=test_embeddings,
            document_labels=self.dataset_context.labels_for_split("test"),
            label_encoder=label_encoder,
        )

        train_loader = ClassifierDataLoader(
            dataset=train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle_train,
        )

        validation_loader = ClassifierDataLoader(
            dataset=validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        test_loader = ClassifierDataLoader(
            dataset=test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
        )

        if train_dataset.input_dim != self.embedder.embedding_dim:
            raise ValueError(
                "Train dataset input dimension does not "
                "match embedder dimension"
            )

        if (
            validation_dataset.input_dim
            != train_dataset.input_dim
        ):
            raise ValueError(
                "Validation input dimension does not "
                "match train input dimension"
            )

        if (
            test_dataset.input_dim
            != train_dataset.input_dim
        ):
            raise ValueError(
                "Test input dimension does not "
                "match train input dimension"
            )

        if (
            train_dataset.sequence_length
            != self.experiment.sequence_length
        ):
            raise ValueError(
                "Train sequence length does not "
                "match experiment configuration"
            )

        return ExperimentData(
            experiment=self.experiment,
            train_dataset=train_dataset,
            validation_dataset=validation_dataset,
            test_dataset=test_dataset,
            train_loader=train_loader,
            validation_loader=validation_loader,
            test_loader=test_loader,
            num_labels=label_encoder.num_labels,
            input_dim=train_dataset.input_dim,
            sequence_length=train_dataset.sequence_length,
        )

class ExperimentContextBuilder:
    """
    Build an ExperimentContext from reusable dataset,
    sequence, and embedding contexts.
    """

    def __init__(
        self,
        dataset_context: DatasetContext,
        sequence_context: DatasetSequenceContext,
        embedding_context: DatasetEmbeddingContext,
        embedder_registry: Dict[str, BaseEmbedder],
    ):
        self.dataset_context = dataset_context
        self.sequence_context = sequence_context
        self.embedding_context = embedding_context
        self.embedder_registry = embedder_registry

    def build(
        self,
        experiment: ExperimentConfig,
        batch_size: int = 8,
        shuffle_train: bool = True,
    ) -> ExperimentContext:

        if experiment.dataset != self.dataset_context.dataset:
            raise ValueError(
                "experiment dataset does not match "
                "dataset context"
            )

        if experiment.dataset != self.sequence_context.dataset:
            raise ValueError(
                "experiment dataset does not match "
                "sequence context"
            )

        if experiment.dataset != self.embedding_context.dataset:
            raise ValueError(
                "experiment dataset does not match "
                "embedding context"
            )

        if experiment.embedding not in self.embedder_registry:
            raise ValueError(
                f"No embedder registered for "
                f"'{experiment.embedding}'"
            )

        embedder = self.embedder_registry[
            experiment.embedding
        ]

        return ExperimentContext(
            dataset_context=self.dataset_context,
            sequence_context=self.sequence_context,
            embedding_context=self.embedding_context,
            experiment=experiment,
            embedder=embedder,
            batch_size=batch_size,
            shuffle_train=shuffle_train,
        )