import numpy as np

from src.embedding.bert import (
    BERTConfig,
    BERTEmbedder,
)
from src.sequence.word_sequence import (
    WordSequence,
)


def make_sequence(words):
    return WordSequence(
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


def test_bert_config_defaults():
    config = BERTConfig()

    assert config.model_name == "bert-base-uncased"
    assert config.max_length == 128
    assert config.layer == -1
    assert config.sequence_length == 25


def test_bert_embedder_dimension():
    embedder = BERTEmbedder(
        BERTConfig(
            sequence_length=5,
        )
    )

    assert embedder.embedding_dim == 768


def test_bert_embedding_shape():
    embedder = BERTEmbedder(
        BERTConfig(
            sequence_length=5,
        )
    )

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
        768,
    )

    assert output.embedding_dim == 768
    assert output.sequence_length == 5


def test_bert_padding_is_zero():
    embedder = BERTEmbedder(
        BERTConfig(
            sequence_length=5,
        )
    )

    sequence = make_sequence(
        [
            "machine",
            "learning",
            "<PAD>",
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

    assert np.allclose(
        output.embeddings[4],
        0.0,
    )


def test_bert_real_words_are_not_zero():
    embedder = BERTEmbedder(
        BERTConfig(
            sequence_length=3,
        )
    )

    sequence = make_sequence(
        [
            "machine",
            "learning",
            "graph",
        ]
    )

    output = embedder.embed(sequence)

    assert not np.allclose(
        output.embeddings[0],
        0.0,
    )

    assert not np.allclose(
        output.embeddings[1],
        0.0,
    )

    assert not np.allclose(
        output.embeddings[2],
        0.0,
    )


def test_bert_output_metadata():
    embedder = BERTEmbedder(
        BERTConfig(
            sequence_length=3,
        )
    )

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


def test_bert_embedding_dtype():
    embedder = BERTEmbedder(
        BERTConfig(
            sequence_length=3,
        )
    )

    sequence = make_sequence(
        [
            "machine",
            "learning",
            "graph",
        ]
    )

    output = embedder.embed(sequence)

    assert output.embeddings.dtype == np.float32