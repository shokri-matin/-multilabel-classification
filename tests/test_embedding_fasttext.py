import numpy as np
import pytest

from src.embedding.fasttext import (
    FastTextConfig,
    FastTextEmbedder,
)
from src.sequence.word_sequence import (
    WordSequence,
)

def test_fasttext_config_defaults():
    config = FastTextConfig(
        model_path="dummy/path"
    )

    assert config.model_path == "dummy/path"
    assert config.embedding_dim == 300
    assert config.padding_token == "<PAD>"
    assert config.sequence_length == 25

def test_fasttext_config_requires_model_path():
    with pytest.raises(ValueError):
        FastTextConfig()

def test_fasttext_config_invalid_dimension():
    with pytest.raises(ValueError):
        FastTextConfig(
            model_path="dummy/path",
            embedding_dim=0,
        )

class FakeFastTextModel:
    
    def get_dimension(self):
        return 4

    def get_word_vector(self, word):
        if word == "machine":
            return np.array(
                [1.0, 2.0, 3.0, 4.0],
                dtype=np.float32,
            )

        if word == "learning":
            return np.array(
                [2.0, 3.0, 4.0, 5.0],
                dtype=np.float32,
            )

        if word == "graph":
            return np.array(
                [3.0, 4.0, 5.0, 6.0],
                dtype=np.float32,
            )

        return np.ones(
            4,
            dtype=np.float32,
        )

def make_sequence(words):

    return WordSequence(
        doc_id="1",
        strategy="degree",
        mode="no_branch",
        central_node=words[0],
        best_neighbor=(
            words[1]
            if len(words) > 1
            else words[0]
        ),
        words=words,
        positions=list(range(len(words))),
        original_length=len(words),
    )

def test_fasttext_embedding_shape(monkeypatch):

    monkeypatch.setattr(
        FastTextEmbedder,
        "_load_model",
        lambda self, path: FakeFastTextModel(),
    )

    config = FastTextConfig(
        model_path="dummy/path",
        embedding_dim=4,
        sequence_length=5,
    )

    embedder = FastTextEmbedder(config)

    sequence = make_sequence(
        [
            "machine",
            "learning",
            "graph",
            "<PAD>",
            "<PAD>",
        ]
    )

    output = embedder.embed(sequence)

    assert output.embeddings.shape == (
        5,
        4,
    )

def test_fasttext_padding_is_zero(monkeypatch):

    monkeypatch.setattr(
        FastTextEmbedder,
        "_load_model",
        lambda self, path: FakeFastTextModel(),
    )

    config = FastTextConfig(
        model_path="dummy/path",
        embedding_dim=4,
    )

    embedder = FastTextEmbedder(config)

    sequence = make_sequence(
        [
            "machine",
            "learning",
            "<PAD>",
            "<PAD>",
        ]
    )

    output = embedder.embed(sequence)

    assert np.allclose(
        output.embeddings[2],
        0.0,
    )

    assert np.allclose(
        output.embeddings[3],
        0.0,
    )


def test_fasttext_output_metadata(monkeypatch):

    monkeypatch.setattr(
        FastTextEmbedder,
        "_load_model",
        lambda self, path: FakeFastTextModel(),
    )

    config = FastTextConfig(
        model_path="dummy/path",
        embedding_dim=4,
    )

    embedder = FastTextEmbedder(config)

    sequence = make_sequence(
        [
            "machine",
            "learning",
            "graph",
        ]
    )

    output = embedder.embed(sequence)

    assert output.strategy == "degree"
    assert output.mode == "no_branch"
    assert output.central_node == "machine"
    assert output.best_neighbor == "learning"

    assert output.words == [
        "machine",
        "learning",
        "graph",
    ]

    assert output.embedding_dim == 4
    assert output.sequence_length == 3