import numpy as np
import pytest

from src.embedding.base import EmbeddingOutput, BaseEmbedder
from src.experiments.config import ExperimentConfig
from src.experiments.experiment_context import (
    ExperimentContext,
    ExperimentContextBuilder,
)


class FakeEmbedder(BaseEmbedder):
    def __init__(self, dimension: int = 4):
        self._dimension = dimension

    @property
    def embedding_dim(self) -> int:
        return self._dimension

    def embed(self, sequence):
        raise NotImplementedError


class FakeDocument:
    def __init__(self, outputs):
        self.outputs = outputs


class FakeSplitEmbeddingContext:
    def __init__(self, documents):
        self.documents = documents

    def iter_documents(
        self,
        strategy,
        mode,
        embedding,
        embedder,
    ):
        yield from self.documents


class FakeEmbeddingContext:
    def __init__(
        self,
        dataset,
        train_documents,
        validation_documents,
        test_documents,
    ):
        self.dataset = dataset

        self._splits = {
            "train": FakeSplitEmbeddingContext(train_documents),
            "validation": FakeSplitEmbeddingContext(
                validation_documents
            ),
            "test": FakeSplitEmbeddingContext(test_documents),
        }

    def split(self, split):
        return self._splits[split]


class FakeSequenceSplitContext:
    def __init__(self, top_k):
        self.top_k = top_k


class FakeSequenceContext:
    def __init__(
        self,
        dataset,
        top_k=200,
        validation_top_k=None,
        test_top_k=None,
    ):
        self.dataset = dataset

        if validation_top_k is None:
            validation_top_k = top_k

        if test_top_k is None:
            test_top_k = top_k

        self.train = FakeSequenceSplitContext(
            top_k=top_k
        )

        self.validation = FakeSequenceSplitContext(
            top_k=validation_top_k
        )

        self.test = FakeSequenceSplitContext(
            top_k=test_top_k
        )


def make_embedding(
    doc_id: str,
    strategy: str = "degree",
    mode: str = "no_branch",
    dimension: int = 4,
    sequence_length: int = 4,
    central_node: str = "traffic",
    best_neighbor: str = "signal",
):
    return EmbeddingOutput(
        doc_id=doc_id,
        strategy=strategy,
        mode=mode,
        central_node=central_node,
        best_neighbor=best_neighbor,
        words=["traffic", "signal", "control", "<PAD>"][
            :sequence_length
        ],
        embeddings=np.ones(
            (sequence_length, dimension),
            dtype=np.float32,
        ),
        embedding_dim=dimension,
        sequence_length=sequence_length,
    )


def make_experiment(
    dataset="aapd",
    mode="no_branch",
    centrality="degree",
    embedding="bert",
    classifier="bert",
    top_k=200,
    max_nodes=10,
    sequence_length=4,
):
    return ExperimentConfig(
        dataset=dataset,
        mode=mode,
        centrality=centrality,
        embedding=embedding,
        classifier=classifier,
        top_k=top_k,
        max_nodes=max_nodes,
        sequence_length=sequence_length,
    )


def make_contexts(
    *,
    dataset="aapd",
    dimension=4,
    sequence_length=4,
    top_k=200,
    validation_top_k=None,
    test_top_k=None,
):
    train_documents = [
        FakeDocument(
            [
                make_embedding(
                    "train-1",
                    dimension=dimension,
                    sequence_length=sequence_length,
                )
            ]
        ),
        FakeDocument(
            [
                make_embedding(
                    "train-2",
                    dimension=dimension,
                    sequence_length=sequence_length,
                )
            ]
        ),
    ]

    validation_documents = [
        FakeDocument(
            [
                make_embedding(
                    "validation-1",
                    dimension=dimension,
                    sequence_length=sequence_length,
                )
            ]
        )
    ]

    test_documents = [
        FakeDocument(
            [
                make_embedding(
                    "test-1",
                    dimension=dimension,
                    sequence_length=sequence_length,
                )
            ]
        )
    ]

    embedding_context = FakeEmbeddingContext(
        dataset=dataset,
        train_documents=train_documents,
        validation_documents=validation_documents,
        test_documents=test_documents,
    )

    sequence_context = FakeSequenceContext(
        dataset=dataset,
        top_k=top_k,
        validation_top_k=validation_top_k,
        test_top_k=test_top_k,
    )

    dataset_context = type(
        "FakeDatasetContext",
        (),
        {
            "dataset": dataset,
            "label_vocabulary": [
                "label_a",
                "label_b",
            ],
            "document_labels": {
                "train": {
                    "train-1": ["label_a"],
                    "train-2": ["label_b"],
                },
                "validation": {
                    "validation-1": ["label_a"],
                },
                "test": {
                    "test-1": ["label_b"],
                },
            },
        },
    )()

    return (
        dataset_context,
        sequence_context,
        embedding_context,
    )


def test_experiment_context_builds_all_datasets_and_loaders():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts()

    experiment = make_experiment()

    embedder = FakeEmbedder(dimension=4)

    context = ExperimentContext(
        dataset_context=dataset_context,
        sequence_context=sequence_context,
        embedding_context=embedding_context,
        experiment=experiment,
        embedder=embedder,
        batch_size=2,
        shuffle_train=True,
    )

    data = context.build()

    assert data.experiment == experiment

    assert len(data.train_dataset) == 2
    assert len(data.validation_dataset) == 1
    assert len(data.test_dataset) == 1

    assert data.num_labels == 2
    assert data.input_dim == 4
    assert data.sequence_length == 4

    assert data.train_loader is not None
    assert data.validation_loader is not None
    assert data.test_loader is not None


def test_experiment_context_uses_train_label_vocabulary():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts()

    experiment = make_experiment()

    context = ExperimentContext(
        dataset_context=dataset_context,
        sequence_context=sequence_context,
        embedding_context=embedding_context,
        experiment=experiment,
        embedder=FakeEmbedder(4),
    )

    data = context.build()

    assert data.num_labels == 2


def test_experiment_context_accepts_matching_top_k():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts(
        top_k=200,
    )

    experiment = make_experiment(
        top_k=200,
    )

    context = ExperimentContext(
        dataset_context=dataset_context,
        sequence_context=sequence_context,
        embedding_context=embedding_context,
        experiment=experiment,
        embedder=FakeEmbedder(4),
    )

    assert context.experiment.top_k == 200
    assert context.sequence_context.train.top_k == 200
    assert context.sequence_context.validation.top_k == 200
    assert context.sequence_context.test.top_k == 200


def test_experiment_context_rejects_train_top_k_mismatch():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts(
        top_k=100,
    )

    experiment = make_experiment(
        top_k=200,
    )

    with pytest.raises(
        ValueError,
        match="train sequence_context top_k does not match experiment top_k",
    ):
        ExperimentContext(
            dataset_context=dataset_context,
            sequence_context=sequence_context,
            embedding_context=embedding_context,
            experiment=experiment,
            embedder=FakeEmbedder(4),
        )


def test_experiment_context_rejects_validation_top_k_mismatch():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts(
        top_k=200,
        validation_top_k=100,
    )

    experiment = make_experiment(
        top_k=200,
    )

    with pytest.raises(
        ValueError,
        match="validation sequence_context top_k does not match experiment top_k",
    ):
        ExperimentContext(
            dataset_context=dataset_context,
            sequence_context=sequence_context,
            embedding_context=embedding_context,
            experiment=experiment,
            embedder=FakeEmbedder(4),
        )


def test_experiment_context_rejects_test_top_k_mismatch():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts(
        top_k=200,
        test_top_k=100,
    )

    experiment = make_experiment(
        top_k=200,
    )

    with pytest.raises(
        ValueError,
        match="test sequence_context top_k does not match experiment top_k",
    ):
        ExperimentContext(
            dataset_context=dataset_context,
            sequence_context=sequence_context,
            embedding_context=embedding_context,
            experiment=experiment,
            embedder=FakeEmbedder(4),
        )


def test_experiment_context_rejects_dataset_context_mismatch():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts(dataset="aapd")

    experiment = make_experiment(dataset="rcv1")

    with pytest.raises(
        ValueError,
        match="dataset_context dataset does not match",
    ):
        ExperimentContext(
            dataset_context=dataset_context,
            sequence_context=sequence_context,
            embedding_context=embedding_context,
            experiment=experiment,
            embedder=FakeEmbedder(4),
        )


def test_experiment_context_rejects_sequence_context_mismatch():
    (
        dataset_context,
        _,
        embedding_context,
    ) = make_contexts(dataset="aapd")

    sequence_context = FakeSequenceContext(
        dataset="rcv1",
        top_k=200,
    )

    experiment = make_experiment(dataset="aapd")

    with pytest.raises(
        ValueError,
        match="sequence_context dataset does not match",
    ):
        ExperimentContext(
            dataset_context=dataset_context,
            sequence_context=sequence_context,
            embedding_context=embedding_context,
            experiment=experiment,
            embedder=FakeEmbedder(4),
        )


def test_experiment_context_rejects_embedding_context_mismatch():
    (
        dataset_context,
        sequence_context,
        _,
    ) = make_contexts(dataset="aapd")

    embedding_context = FakeEmbeddingContext(
        dataset="rcv1",
        train_documents=[],
        validation_documents=[],
        test_documents=[],
    )

    experiment = make_experiment(dataset="aapd")

    with pytest.raises(
        ValueError,
        match="embedding_context dataset does not match",
    ):
        ExperimentContext(
            dataset_context=dataset_context,
            sequence_context=sequence_context,
            embedding_context=embedding_context,
            experiment=experiment,
            embedder=FakeEmbedder(4),
        )


def test_experiment_context_rejects_invalid_batch_size():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts()

    experiment = make_experiment()

    with pytest.raises(
        ValueError,
        match="batch_size must be positive",
    ):
        ExperimentContext(
            dataset_context=dataset_context,
            sequence_context=sequence_context,
            embedding_context=embedding_context,
            experiment=experiment,
            embedder=FakeEmbedder(4),
            batch_size=0,
        )


def test_experiment_context_rejects_invalid_embedder_dimension():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts()

    experiment = make_experiment()

    with pytest.raises(
        ValueError,
        match="embedder dimension must be positive",
    ):
        ExperimentContext(
            dataset_context=dataset_context,
            sequence_context=sequence_context,
            embedding_context=embedding_context,
            experiment=experiment,
            embedder=FakeEmbedder(0),
        )


def test_experiment_context_rejects_input_dimension_mismatch():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts(dimension=8)

    experiment = make_experiment()

    context = ExperimentContext(
        dataset_context=dataset_context,
        sequence_context=sequence_context,
        embedding_context=embedding_context,
        experiment=experiment,
        embedder=FakeEmbedder(4),
    )

    with pytest.raises(
        ValueError,
        match="Train dataset input dimension does not match",
    ):
        context.build()


def test_experiment_context_rejects_sequence_length_mismatch():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts(sequence_length=6)

    experiment = make_experiment(
        sequence_length=4
    )

    context = ExperimentContext(
        dataset_context=dataset_context,
        sequence_context=sequence_context,
        embedding_context=embedding_context,
        experiment=experiment,
        embedder=FakeEmbedder(4),
    )

    with pytest.raises(
        ValueError,
        match="Train sequence length does not match",
    ):
        context.build()


def test_experiment_context_rejects_empty_split():
    (
        dataset_context,
        sequence_context,
        _,
    ) = make_contexts()

    embedding_context = FakeEmbeddingContext(
        dataset="aapd",
        train_documents=[],
        validation_documents=[],
        test_documents=[],
    )

    experiment = make_experiment()

    context = ExperimentContext(
        dataset_context=dataset_context,
        sequence_context=sequence_context,
        embedding_context=embedding_context,
        experiment=experiment,
        embedder=FakeEmbedder(4),
    )

    with pytest.raises(
        ValueError,
        match="No embeddings generated for split 'train'",
    ):
        context.build()


def test_experiment_context_builder_selects_registered_embedder():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts()

    bert_embedder = FakeEmbedder(4)
    fasttext_embedder = FakeEmbedder(300)

    builder = ExperimentContextBuilder(
        dataset_context=dataset_context,
        sequence_context=sequence_context,
        embedding_context=embedding_context,
        embedder_registry={
            "bert": bert_embedder,
            "fasttext": fasttext_embedder,
        },
    )

    experiment = make_experiment(
        embedding="bert"
    )

    context = builder.build(
        experiment=experiment,
        batch_size=2,
    )

    assert context.embedder is bert_embedder


def test_experiment_context_builder_rejects_missing_embedder():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts()

    builder = ExperimentContextBuilder(
        dataset_context=dataset_context,
        sequence_context=sequence_context,
        embedding_context=embedding_context,
        embedder_registry={},
    )

    experiment = make_experiment(
        embedding="bert"
    )

    with pytest.raises(
        ValueError,
        match="No embedder registered for 'bert'",
    ):
        builder.build(experiment)


def test_experiment_context_builder_rejects_dataset_mismatch():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts(dataset="aapd")

    builder = ExperimentContextBuilder(
        dataset_context=dataset_context,
        sequence_context=sequence_context,
        embedding_context=embedding_context,
        embedder_registry={
            "bert": FakeEmbedder(4),
        },
    )

    experiment = make_experiment(
        dataset="rcv1"
    )

    with pytest.raises(
        ValueError,
        match="experiment dataset does not match dataset context",
    ):
        builder.build(experiment)


def test_experiment_context_supports_fasttext():
    (
        dataset_context,
        sequence_context,
        embedding_context,
    ) = make_contexts(
        dimension=300
    )

    builder = ExperimentContextBuilder(
        dataset_context=dataset_context,
        sequence_context=sequence_context,
        embedding_context=embedding_context,
        embedder_registry={
            "fasttext": FakeEmbedder(300),
        },
    )

    experiment = make_experiment(
        embedding="fasttext"
    )

    context = builder.build(
        experiment=experiment
    )

    assert context.embedder.embedding_dim == 300
    assert context.experiment.embedding == "fasttext"