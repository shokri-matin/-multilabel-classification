import numpy as np
import pytest

from src.embedding.base import (
    BaseEmbedder,
    EmbeddingOutput,
)
from src.embedding.config import EmbeddingConfig


def test_embedding_config_defaults():
    config = EmbeddingConfig()

    assert config.sequence_length == 25
    assert config.normalize is False
    assert config.batch_size == 8
    assert config.seed == 42


def test_embedding_config_custom_values():
    config = EmbeddingConfig(
        sequence_length=50,
        normalize=True,
        batch_size=16,
        seed=123,
    )

    assert config.sequence_length == 50
    assert config.normalize is True
    assert config.batch_size == 16
    assert config.seed == 123


def test_embedding_config_invalid_sequence_length():
    with pytest.raises(ValueError):
        EmbeddingConfig(
            sequence_length=0
        )

    with pytest.raises(ValueError):
        EmbeddingConfig(
            sequence_length=-1
        )


def test_embedding_config_invalid_batch_size():
    with pytest.raises(ValueError):
        EmbeddingConfig(
            batch_size=0
        )

    with pytest.raises(ValueError):
        EmbeddingConfig(
            batch_size=-1
        )


def test_embedding_output_structure():
    embeddings = np.zeros(
        (25, 768),
        dtype=np.float32,
    )

    output = EmbeddingOutput(
        strategy="degree",
        mode="no_branch",
        central_node="machine",
        best_neighbor="learning",
        words=[
            "machine",
            "learning",
        ],
        embeddings=embeddings,
        embedding_dim=768,
        sequence_length=25,
    )

    assert output.strategy == "degree"
    assert output.mode == "no_branch"
    assert output.central_node == "machine"
    assert output.best_neighbor == "learning"

    assert output.embedding_dim == 768
    assert output.sequence_length == 25

    assert output.embeddings.shape == (
        25,
        768,
    )

    assert output.embeddings.dtype == np.float32


def test_base_embedder_is_abstract():
    with pytest.raises(TypeError):
        BaseEmbedder()