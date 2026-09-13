import numpy as np
import pytest
import torch

from src.classifier.dataset import (
    ClassifierDataset,
    LabelEncoder,
)
from src.embedding.base import EmbeddingOutput


def make_embedding(
    doc_id: str,
    words: list[str],
    embedding_dim: int = 4,
) -> EmbeddingOutput:
    embeddings = np.ones(
        (len(words), embedding_dim),
        dtype=np.float32,
    )

    return EmbeddingOutput(
        doc_id=doc_id,
        strategy="degree",
        mode="no_branch",
        central_node="machine",
        best_neighbor="learning",
        words=words,
        embeddings=embeddings,
        embedding_dim=embedding_dim,
        sequence_length=len(words),
    )


def test_label_encoder_multi_hot():
    encoder = LabelEncoder(
        ["CCAT", "GCAT", "MCAT", "ECAT"]
    )

    encoded = encoder.encode(
        ["CCAT", "MCAT"]
    )

    expected = torch.tensor(
        [1.0, 0.0, 1.0, 0.0]
    )

    assert torch.equal(encoded, expected)


def test_label_encoder_decode():
    encoder = LabelEncoder(
        ["CCAT", "GCAT", "MCAT"]
    )

    encoded = torch.tensor(
        [1.0, 0.0, 1.0]
    )

    decoded = encoder.decode(encoded)

    assert decoded == ["CCAT", "MCAT"]


def test_dataset_attention_mask():
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
        )
    ]

    encoder = LabelEncoder(
        ["CCAT", "GCAT"]
    )

    dataset = ClassifierDataset(
        embeddings=embeddings,
        document_labels={
            "doc_001": ["CCAT"],
        },
        label_encoder=encoder,
    )

    sample = dataset[0]

    expected_mask = torch.tensor(
        [1, 1, 1, 0, 0],
        dtype=torch.long,
    )

    assert torch.equal(
        sample["attention_mask"],
        expected_mask,
    )


def test_dataset_shapes():
    embeddings = [
        make_embedding(
            doc_id="doc_001",
            words=[
                "machine",
                "learning",
                "graph",
                "<PAD>",
            ],
            embedding_dim=4,
        )
    ]

    encoder = LabelEncoder(
        ["CCAT", "GCAT", "MCAT"]
    )

    dataset = ClassifierDataset(
        embeddings=embeddings,
        document_labels={
            "doc_001": ["CCAT", "MCAT"],
        },
        label_encoder=encoder,
    )

    sample = dataset[0]

    assert sample["inputs_embeds"].shape == (4, 4)
    assert sample["attention_mask"].shape == (4,)
    assert sample["labels"].shape == (3,)


def test_dataset_preserves_document_id():
    embeddings = [
        make_embedding(
            doc_id="doc_123",
            words=["machine", "learning"],
        )
    ]

    encoder = LabelEncoder(
        ["CCAT"]
    )

    dataset = ClassifierDataset(
        embeddings=embeddings,
        document_labels={
            "doc_123": ["CCAT"],
        },
        label_encoder=encoder,
    )

    sample = dataset[0]

    assert sample["doc_id"] == "doc_123"


def test_multiple_sequences_keep_same_document_labels():
    embeddings = [
        make_embedding(
            doc_id="doc_001",
            words=["machine", "learning"],
        ),
        make_embedding(
            doc_id="doc_001",
            words=["graph", "model"],
        ),
    ]

    encoder = LabelEncoder(
        ["CCAT", "GCAT", "MCAT"]
    )

    dataset = ClassifierDataset(
        embeddings=embeddings,
        document_labels={
            "doc_001": ["CCAT", "MCAT"],
        },
        label_encoder=encoder,
    )

    assert len(dataset) == 2

    sample_0 = dataset[0]
    sample_1 = dataset[1]

    expected_labels = torch.tensor(
        [1.0, 0.0, 1.0]
    )

    assert sample_0["doc_id"] == "doc_001"
    assert sample_1["doc_id"] == "doc_001"

    assert torch.equal(
        sample_0["labels"],
        expected_labels,
    )

    assert torch.equal(
        sample_1["labels"],
        expected_labels,
    )


def test_unknown_document_raises_error():
    embeddings = [
        make_embedding(
            doc_id="doc_999",
            words=["machine"],
        )
    ]

    encoder = LabelEncoder(
        ["CCAT"]
    )

    with pytest.raises(ValueError, match="No labels found"):
        ClassifierDataset(
            embeddings=embeddings,
            document_labels={
                "doc_001": ["CCAT"],
            },
            label_encoder=encoder,
        )


def test_unknown_label_raises_error():
    embeddings = [
        make_embedding(
            doc_id="doc_001",
            words=["machine"],
        )
    ]

    encoder = LabelEncoder(
        ["CCAT"]
    )

    with pytest.raises(ValueError, match="Unknown label"):
        ClassifierDataset(
            embeddings=embeddings,
            document_labels={
                "doc_001": ["UNKNOWN"],
            },
            label_encoder=encoder,
        )