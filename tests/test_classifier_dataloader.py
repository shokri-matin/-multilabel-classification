import numpy as np
import pytest
import torch

from src.classifier.dataset import (
    ClassifierDataset,
    LabelEncoder,
)
from src.classifier.dataloader import (
    ClassifierBatch,
    ClassifierDataLoader,
)
from src.embedding.base import EmbeddingOutput


def make_embedding(
    doc_id: str,
    words: list[str],
    embedding_dim: int = 4,
    strategy: str = "degree",
    mode: str = "no_branch",
) -> EmbeddingOutput:
    embeddings = np.ones(
        (len(words), embedding_dim),
        dtype=np.float32,
    )

    return EmbeddingOutput(
        doc_id=doc_id,
        strategy=strategy,
        mode=mode,
        central_node="machine",
        best_neighbor="learning",
        words=words,
        embeddings=embeddings,
        embedding_dim=embedding_dim,
        sequence_length=len(words),
    )


def make_dataset() -> ClassifierDataset:
    embeddings = [
        make_embedding(
            doc_id="doc_001",
            words=[
                "machine",
                "learning",
                "graph",
                "<PAD>",
                "<PAD>",
            ],
        ),
        make_embedding(
            doc_id="doc_001",
            words=[
                "graph",
                "model",
                "network",
                "<PAD>",
                "<PAD>",
            ],
        ),
        make_embedding(
            doc_id="doc_002",
            words=[
                "traffic",
                "signal",
                "control",
                "<PAD>",
                "<PAD>",
            ],
        ),
        make_embedding(
            doc_id="doc_003",
            words=[
                "deep",
                "learning",
                "model",
                "<PAD>",
                "<PAD>",
            ],
        ),
    ]

    encoder = LabelEncoder(
        ["CCAT", "GCAT", "MCAT"]
    )

    return ClassifierDataset(
        embeddings=embeddings,
        document_labels={
            "doc_001": ["CCAT", "MCAT"],
            "doc_002": ["GCAT"],
            "doc_003": ["CCAT", "GCAT"],
        },
        label_encoder=encoder,
    )


def test_dataloader_returns_classifier_batch():
    dataset = make_dataset()

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert isinstance(batch, ClassifierBatch)


def test_batch_embedding_shape():
    dataset = make_dataset()

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch.inputs_embeds.shape == (2, 5, 4)


def test_batch_attention_mask_shape():
    dataset = make_dataset()

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch.attention_mask.shape == (2, 5)


def test_batch_labels_shape():
    dataset = make_dataset()

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch.labels.shape == (2, 3)


def test_batch_attention_mask_values():
    dataset = make_dataset()

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    expected = torch.tensor(
        [
            [1, 1, 1, 0, 0],
            [1, 1, 1, 0, 0],
        ],
        dtype=torch.long,
    )

    assert torch.equal(
        batch.attention_mask,
        expected,
    )


def test_batch_labels_are_preserved():
    dataset = make_dataset()

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    expected = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
        ]
    )

    assert torch.equal(
        batch.labels,
        expected,
    )


def test_batch_preserves_document_ids():
    dataset = make_dataset()

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch.doc_ids == [
        "doc_001",
        "doc_001",
    ]


def test_batch_preserves_metadata():
    dataset = make_dataset()

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch.strategies == [
        "degree",
        "degree",
    ]

    assert batch.modes == [
        "no_branch",
        "no_branch",
    ]

    assert batch.central_nodes == [
        "machine",
        "machine",
    ]

    assert batch.best_neighbors == [
        "learning",
        "learning",
    ]


def test_dataloader_length():
    dataset = make_dataset()

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    assert len(dataloader) == 2


def test_dataloader_last_batch():
    dataset = make_dataset()

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=3,
        shuffle=False,
    )

    batches = list(dataloader)

    assert len(batches) == 2
    assert batches[0].inputs_embeds.shape == (3, 5, 4)
    assert batches[1].inputs_embeds.shape == (1, 5, 4)


def test_invalid_batch_size():
    dataset = make_dataset()

    with pytest.raises(
        ValueError,
        match="batch_size must be positive",
    ):
        ClassifierDataLoader(
            dataset=dataset,
            batch_size=0,
        )