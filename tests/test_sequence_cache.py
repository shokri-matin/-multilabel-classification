from pathlib import Path

import pytest

from src.experiments.sequence_cache import (
    SequenceArtifactCache,
)
from src.graph.processor import SubgraphResult
from src.sequence.word_sequence import (
    SequenceConfig,
    WordSequence,
    WordSequenceBuilder,
)


def make_sequence() -> WordSequence:
    graph = __import__(
        "networkx"
    ).Graph()

    graph.add_node(
        "machine",
        positions=[0],
    )

    graph.add_node(
        "learning",
        positions=[1],
    )

    graph.add_node(
        "model",
        positions=[2],
    )

    graph.add_edge(
        "machine",
        "learning",
        weight=1,
    )

    graph.add_edge(
        "learning",
        "model",
        weight=1,
    )

    subgraph = SubgraphResult(
        doc_id="doc_001",
        strategy="closeness",
        central_node="machine",
        central_score=1.0,
        best_neighbor="learning",
        nodes=[
            "machine",
            "learning",
            "model",
        ],
        graph=graph,
    )

    builder = WordSequenceBuilder(
        SequenceConfig(
            sequence_length=5,
        )
    )

    return builder.build(
        subgraph=subgraph,
        mode="no_branch",
    )


def test_cache_save_and_load(tmp_path: Path):
    cache = SequenceArtifactCache(
        tmp_path
    )

    sequence = make_sequence()

    path = cache.save(
        dataset="aapd",
        doc_id="doc_001",
        centrality="closeness",
        mode="no_branch",
        sequences=[sequence],
    )

    assert path.exists()

    assert cache.exists(
        "aapd",
        "doc_001",
        "closeness",
        "no_branch",
    )

    loaded = cache.load(
        "aapd",
        "doc_001",
        "closeness",
        "no_branch",
    )

    assert len(loaded) == 1
    assert loaded[0].words == sequence.words
    assert loaded[0].positions == sequence.positions
    assert loaded[0].doc_id == "doc_001"


def test_cache_preserves_metadata(
    tmp_path: Path,
):
    cache = SequenceArtifactCache(
        tmp_path
    )

    sequence = make_sequence()

    cache.save(
        dataset="aapd",
        doc_id="doc_002",
        centrality="degree",
        mode="branch",
        sequences=[sequence],
    )

    loaded = cache.load(
        "aapd",
        "doc_002",
        "degree",
        "branch",
    )

    assert loaded[0].strategy == "closeness"
    assert loaded[0].mode == "no_branch"
    assert loaded[0].central_node == "machine"
    assert loaded[0].best_neighbor == "learning"


def test_composite_centrality_path(
    tmp_path: Path,
):
    cache = SequenceArtifactCache(
        tmp_path
    )

    sequence = make_sequence()

    path = cache.save(
        dataset="aapd",
        doc_id="doc_003",
        centrality=(
            "closeness*degree*clustering"
        ),
        mode="no_branch",
        sequences=[sequence],
    )

    assert path.name == (
        "closeness_degree_clustering"
        "__no_branch.pkl"
    )

    assert cache.exists(
        "aapd",
        "doc_003",
        "closeness*degree*clustering",
        "no_branch",
    )


def test_cache_delete(tmp_path: Path):
    cache = SequenceArtifactCache(
        tmp_path
    )

    sequence = make_sequence()

    cache.save(
        dataset="aapd",
        doc_id="doc_004",
        centrality="pagerank",
        mode="branch",
        sequences=[sequence],
    )

    assert cache.exists(
        "aapd",
        "doc_004",
        "pagerank",
        "branch",
    )

    cache.delete(
        "aapd",
        "doc_004",
        "pagerank",
        "branch",
    )

    assert not cache.exists(
        "aapd",
        "doc_004",
        "pagerank",
        "branch",
    )


def test_missing_artifact(
    tmp_path: Path,
):
    cache = SequenceArtifactCache(
        tmp_path
    )

    with pytest.raises(FileNotFoundError):
        cache.load(
            "aapd",
            "missing",
            "degree",
            "branch",
        )


def test_rejects_invalid_dataset(
    tmp_path: Path,
):
    cache = SequenceArtifactCache(
        tmp_path
    )

    with pytest.raises(ValueError):
        cache.exists(
            "",
            "doc_001",
            "degree",
            "branch",
        )


def test_rejects_invalid_doc_id(
    tmp_path: Path,
):
    cache = SequenceArtifactCache(
        tmp_path
    )

    with pytest.raises(ValueError):
        cache.exists(
            "aapd",
            "folder/doc",
            "degree",
            "branch",
        )


def test_rejects_invalid_centrality(
    tmp_path: Path,
):
    cache = SequenceArtifactCache(
        tmp_path
    )

    with pytest.raises(ValueError):
        cache.exists(
            "aapd",
            "doc_001",
            "",
            "branch",
        )


def test_rejects_invalid_mode(
    tmp_path: Path,
):
    cache = SequenceArtifactCache(
        tmp_path
    )

    with pytest.raises(ValueError):
        cache.exists(
            "aapd",
            "doc_001",
            "degree",
            "",
        )


def test_rejects_invalid_sequences(
    tmp_path: Path,
):
    cache = SequenceArtifactCache(
        tmp_path
    )

    with pytest.raises(TypeError):
        cache.save(
            dataset="aapd",
            doc_id="doc_001",
            centrality="degree",
            mode="branch",
            sequences=["invalid"],
        )