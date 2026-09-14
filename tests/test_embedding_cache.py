from pathlib import Path

import numpy as np
import pytest

from src.embedding.base import EmbeddingOutput
from src.experiments.embedding_cache import (
    EmbeddingArtifactCache,
)


def make_output() -> EmbeddingOutput:
    return EmbeddingOutput(
        doc_id="doc_001",
        strategy="closeness",
        mode="no_branch",
        central_node="machine",
        best_neighbor="learning",
        words=[
            "machine",
            "learning",
            "<PAD>",
        ],
        embeddings=np.array(
            [
                [0.1, 0.2, 0.3],
                [0.4, 0.5, 0.6],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
        embedding_dim=3,
        sequence_length=3,
    )


def test_cache_save_and_load(
    tmp_path: Path,
):
    cache = EmbeddingArtifactCache(
        tmp_path
    )

    output = make_output()

    path = cache.save(
        dataset="aapd",
        doc_id="doc_001",
        centrality="closeness",
        mode="no_branch",
        embedding="fasttext",
        outputs=[output],
    )

    assert path.exists()

    loaded = cache.load(
        dataset="aapd",
        doc_id="doc_001",
        centrality="closeness",
        mode="no_branch",
        embedding="fasttext",
    )

    assert len(loaded) == 1

    assert loaded[0].doc_id == "doc_001"
    assert loaded[0].strategy == "closeness"
    assert loaded[0].mode == "no_branch"
    assert loaded[0].words == output.words
    assert loaded[0].embedding_dim == 3
    assert loaded[0].sequence_length == 3

    np.testing.assert_allclose(
        loaded[0].embeddings,
        output.embeddings,
    )


def test_cache_supports_multiple_outputs(
    tmp_path: Path,
):
    cache = EmbeddingArtifactCache(
        tmp_path
    )

    output1 = make_output()

    output2 = EmbeddingOutput(
        doc_id="doc_001",
        strategy="closeness",
        mode="no_branch",
        central_node="model",
        best_neighbor="learning",
        words=[
            "model",
            "learning",
            "<PAD>",
        ],
        embeddings=np.ones(
            (3, 3),
            dtype=np.float32,
        ),
        embedding_dim=3,
        sequence_length=3,
    )

    cache.save(
        dataset="aapd",
        doc_id="doc_001",
        centrality="closeness",
        mode="no_branch",
        embedding="bert",
        outputs=[
            output1,
            output2,
        ],
    )

    loaded = cache.load(
        dataset="aapd",
        doc_id="doc_001",
        centrality="closeness",
        mode="no_branch",
        embedding="bert",
    )

    assert len(loaded) == 2

    np.testing.assert_allclose(
        loaded[1].embeddings,
        output2.embeddings,
    )


def test_cache_exists(
    tmp_path: Path,
):
    cache = EmbeddingArtifactCache(
        tmp_path
    )

    assert not cache.exists(
        "aapd",
        "doc_001",
        "degree",
        "branch",
        "bert",
    )

    cache.save(
        dataset="aapd",
        doc_id="doc_001",
        centrality="degree",
        mode="branch",
        embedding="bert",
        outputs=[make_output()],
    )

    assert cache.exists(
        "aapd",
        "doc_001",
        "degree",
        "branch",
        "bert",
    )


def test_composite_centrality_path(
    tmp_path: Path,
):
    cache = EmbeddingArtifactCache(
        tmp_path
    )

    path = cache.save(
        dataset="rcv1",
        doc_id="doc_001",
        centrality=(
            "closeness*degree*clustering"
        ),
        mode="branch",
        embedding="fasttext",
        outputs=[make_output()],
    )

    assert path.name == (
        "closeness_degree_clustering"
        "__branch"
        "__fasttext.npz"
    )


def test_cache_delete(
    tmp_path: Path,
):
    cache = EmbeddingArtifactCache(
        tmp_path
    )

    cache.save(
        dataset="aapd",
        doc_id="doc_001",
        centrality="pagerank",
        mode="branch",
        embedding="fasttext",
        outputs=[make_output()],
    )

    assert cache.exists(
        "aapd",
        "doc_001",
        "pagerank",
        "branch",
        "fasttext",
    )

    cache.delete(
        "aapd",
        "doc_001",
        "pagerank",
        "branch",
        "fasttext",
    )

    assert not cache.exists(
        "aapd",
        "doc_001",
        "pagerank",
        "branch",
        "fasttext",
    )


def test_missing_artifact(
    tmp_path: Path,
):
    cache = EmbeddingArtifactCache(
        tmp_path
    )

    with pytest.raises(FileNotFoundError):
        cache.load(
            "aapd",
            "missing",
            "degree",
            "branch",
            "bert",
        )


def test_rejects_invalid_embedding(
    tmp_path: Path,
):
    cache = EmbeddingArtifactCache(
        tmp_path
    )

    with pytest.raises(ValueError):
        cache.exists(
            "aapd",
            "doc_001",
            "degree",
            "branch",
            "word2vec",
        )


def test_rejects_invalid_doc_id(
    tmp_path: Path,
):
    cache = EmbeddingArtifactCache(
        tmp_path
    )

    with pytest.raises(ValueError):
        cache.exists(
            "aapd",
            "folder/doc",
            "degree",
            "branch",
            "bert",
        )


def test_rejects_invalid_outputs(
    tmp_path: Path,
):
    cache = EmbeddingArtifactCache(
        tmp_path
    )

    with pytest.raises(TypeError):
        cache.save(
            dataset="aapd",
            doc_id="doc_001",
            centrality="degree",
            mode="branch",
            embedding="bert",
            outputs=["invalid"],
        )


def test_empty_outputs(
    tmp_path: Path,
):
    cache = EmbeddingArtifactCache(
        tmp_path
    )

    cache.save(
        dataset="aapd",
        doc_id="doc_empty",
        centrality="degree",
        mode="branch",
        embedding="bert",
        outputs=[],
    )

    loaded = cache.load(
        dataset="aapd",
        doc_id="doc_empty",
        centrality="degree",
        mode="branch",
        embedding="bert",
    )

    assert loaded == []