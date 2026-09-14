from pathlib import Path

import networkx as nx
import pytest

from src.experiments.graph_cache import GraphArtifactCache
from src.graph.centrality import (
    CentralityCalculator,
)


def build_graph():
    graph = nx.Graph()

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
        weight=2,
    )

    graph.add_edge(
        "learning",
        "model",
        weight=1,
    )

    return graph


def build_centralities(graph):
    calculator = CentralityCalculator()

    return calculator.rank(
        graph,
        top_k=2,
    )


def test_cache_save_and_load(tmp_path: Path):
    cache = GraphArtifactCache(tmp_path)

    graph = build_graph()
    centralities = build_centralities(graph)

    path = cache.save(
        doc_id="doc_001",
        graph=graph,
        centralities=centralities,
    )

    assert path.exists()
    assert cache.exists("doc_001")

    loaded_graph, loaded_centralities = cache.load(
        "doc_001"
    )

    assert loaded_graph.number_of_nodes() == 3
    assert loaded_graph.number_of_edges() == 2

    assert set(loaded_graph.nodes) == {
        "machine",
        "learning",
        "model",
    }

    assert set(loaded_centralities.keys()) == {
        "closeness",
        "degree",
        "betweenness",
        "pagerank",
        "clustering",
        "closeness*degree*clustering",
    }


def test_cache_preserves_node_positions(tmp_path: Path):
    cache = GraphArtifactCache(tmp_path)

    graph = build_graph()
    centralities = build_centralities(graph)

    cache.save(
        doc_id="doc_002",
        graph=graph,
        centralities=centralities,
    )

    loaded_graph, _ = cache.load("doc_002")

    assert loaded_graph.nodes["machine"]["positions"] == [0]
    assert loaded_graph.nodes["learning"]["positions"] == [1]
    assert loaded_graph.nodes["model"]["positions"] == [2]


def test_cache_preserves_edge_weights(tmp_path: Path):
    cache = GraphArtifactCache(tmp_path)

    graph = build_graph()
    centralities = build_centralities(graph)

    cache.save(
        doc_id="doc_003",
        graph=graph,
        centralities=centralities,
    )

    loaded_graph, _ = cache.load("doc_003")

    assert loaded_graph["machine"]["learning"]["weight"] == 2
    assert loaded_graph["learning"]["model"]["weight"] == 1


def test_cache_preserves_centrality_scores(tmp_path: Path):
    cache = GraphArtifactCache(tmp_path)

    graph = build_graph()
    centralities = build_centralities(graph)

    cache.save(
        doc_id="doc_004",
        graph=graph,
        centralities=centralities,
    )

    _, loaded_centralities = cache.load("doc_004")

    original = centralities["degree"]
    loaded = loaded_centralities["degree"]

    assert original.scores == loaded.scores
    assert original.ranked_nodes == loaded.ranked_nodes
    assert original.top_k_nodes == loaded.top_k_nodes


def test_cache_exists(tmp_path: Path):
    cache = GraphArtifactCache(tmp_path)

    assert not cache.exists("doc_005")

    graph = build_graph()
    centralities = build_centralities(graph)

    cache.save(
        doc_id="doc_005",
        graph=graph,
        centralities=centralities,
    )

    assert cache.exists("doc_005")


def test_cache_delete(tmp_path: Path):
    cache = GraphArtifactCache(tmp_path)

    graph = build_graph()
    centralities = build_centralities(graph)

    cache.save(
        doc_id="doc_006",
        graph=graph,
        centralities=centralities,
    )

    assert cache.exists("doc_006")

    cache.delete("doc_006")

    assert not cache.exists("doc_006")


def test_cache_missing_document(tmp_path: Path):
    cache = GraphArtifactCache(tmp_path)

    with pytest.raises(FileNotFoundError):
        cache.load("missing_doc")


def test_cache_rejects_invalid_doc_id(tmp_path: Path):
    cache = GraphArtifactCache(tmp_path)

    with pytest.raises(ValueError):
        cache.exists("")

    with pytest.raises(ValueError):
        cache.exists("folder/doc")

    with pytest.raises(ValueError):
        cache.exists("folder\\doc")


def test_cache_rejects_invalid_graph(tmp_path: Path):
    cache = GraphArtifactCache(tmp_path)

    calculator = CentralityCalculator()

    graph = build_graph()
    centralities = calculator.rank(
        graph,
        top_k=2,
    )

    with pytest.raises(TypeError):
        cache.save(
            doc_id="doc_007",
            graph="not a graph",
            centralities=centralities,
        )