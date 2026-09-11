# tests/test_word_sequence.py

import networkx as nx
import pytest

from src.graph.processor import SubgraphResult
from src.sequence.word_sequence import (
    SequenceConfig,
    WordSequenceBuilder,
)


def build_subgraph(
    nodes_with_positions,
    strategy="degree",
    mode="no_branch",
):
    graph = nx.Graph()

    for node, positions in nodes_with_positions.items():
        graph.add_node(
            node,
            positions=positions,
        )

    nodes = list(nodes_with_positions.keys())

    for index in range(len(nodes) - 1):
        graph.add_edge(
            nodes[index],
            nodes[index + 1],
        )

    return SubgraphResult(
        strategy=strategy,
        central_node=nodes[0],
        central_score=1.0,
        best_neighbor=(
            nodes[1]
            if len(nodes) > 1
            else nodes[0]
        ),
        nodes=nodes,
        graph=graph,
    )

def test_nodes_are_sorted_by_original_position():
    subgraph = build_subgraph(
        {
            "graph": [4],
            "machine": [0],
            "model": [7],
            "learning": [1],
        }
    )

    builder = WordSequenceBuilder(
        SequenceConfig(sequence_length=10)
    )

    result = builder.build(
        subgraph,
        mode="no_branch",
    )

    assert result.words[:4] == [
        "machine",
        "learning",
        "graph",
        "model",
    ]

    assert result.positions[:4] == [
        0,
        1,
        4,
        7,
    ]


def test_earliest_position_is_used_for_repeated_word():
    subgraph = build_subgraph(
        {
            "machine": [0, 5, 10],
            "learning": [1],
            "model": [3],
        }
    )

    builder = WordSequenceBuilder(
        SequenceConfig(sequence_length=10)
    )

    result = builder.build(
        subgraph,
        mode="no_branch",
    )

    assert result.words[:3] == [
        "machine",
        "learning",
        "model",
    ]

    assert result.positions[:3] == [
        0,
        1,
        3,
    ]


def test_sequence_is_padded():
    subgraph = build_subgraph(
        {
            "machine": [0],
            "learning": [1],
            "model": [2],
        }
    )

    builder = WordSequenceBuilder(
        SequenceConfig(
            sequence_length=6,
            padding_token="<PAD>",
        )
    )

    result = builder.build(
        subgraph,
        mode="no_branch",
    )

    assert result.words == [
        "machine",
        "learning",
        "model",
        "<PAD>",
        "<PAD>",
        "<PAD>",
    ]

    assert result.positions == [
        0,
        1,
        2,
        -1,
        -1,
        -1,
    ]

    assert result.length == 6
    assert result.original_length == 3
    assert result.is_padded is True
    assert result.is_truncated is False


def test_sequence_is_truncated():
    subgraph = build_subgraph(
        {
            "one": [0],
            "two": [1],
            "three": [2],
            "four": [3],
            "five": [4],
        }
    )

    builder = WordSequenceBuilder(
        SequenceConfig(sequence_length=3)
    )

    result = builder.build(
        subgraph,
        mode="no_branch",
    )

    assert result.words == [
        "one",
        "two",
        "three",
    ]

    assert result.positions == [
        0,
        1,
        2,
    ]

    assert result.length == 3
    assert result.original_length == 5
    assert result.is_padded is False
    assert result.is_truncated is True


def test_exact_length_sequence_has_no_padding_or_truncation():
    subgraph = build_subgraph(
        {
            "one": [0],
            "two": [1],
            "three": [2],
        }
    )

    builder = WordSequenceBuilder(
        SequenceConfig(sequence_length=3)
    )

    result = builder.build(
        subgraph,
        mode="no_branch",
    )

    assert result.words == [
        "one",
        "two",
        "three",
    ]

    assert result.positions == [
        0,
        1,
        2,
    ]

    assert result.original_length == 3
    assert result.is_padded is False
    assert result.is_truncated is False


def test_metadata_is_preserved():
    subgraph = build_subgraph(
        {
            "machine": [0],
            "learning": [1],
        },
        strategy="pagerank",
    )

    subgraph.central_node = "machine"
    subgraph.best_neighbor = "learning"

    builder = WordSequenceBuilder(
        SequenceConfig(sequence_length=5)
    )

    result = builder.build(
        subgraph,
        mode="branch",
    )

    assert result.strategy == "pagerank"
    assert result.mode == "branch"
    assert result.central_node == "machine"
    assert result.best_neighbor == "learning"


def test_branch_mode_is_preserved():
    subgraph = build_subgraph(
        {
            "machine": [0],
            "learning": [1],
        }
    )

    builder = WordSequenceBuilder(
        SequenceConfig(sequence_length=5)
    )

    result = builder.build(
        subgraph,
        mode="branch",
    )

    assert result.mode == "branch"


def test_no_branch_mode_is_preserved():
    subgraph = build_subgraph(
        {
            "machine": [0],
            "learning": [1],
        }
    )

    builder = WordSequenceBuilder(
        SequenceConfig(sequence_length=5)
    )

    result = builder.build(
        subgraph,
        mode="no_branch",
    )

    assert result.mode == "no_branch"


def test_invalid_sequence_length():
    with pytest.raises(ValueError):
        SequenceConfig(sequence_length=0)

    with pytest.raises(ValueError):
        SequenceConfig(sequence_length=-1)


def test_invalid_padding_token():
    with pytest.raises(ValueError):
        SequenceConfig(
            sequence_length=10,
            padding_token="",
        )


def test_invalid_mode():
    subgraph = build_subgraph(
        {
            "machine": [0],
        }
    )

    builder = WordSequenceBuilder(
        SequenceConfig(sequence_length=5)
    )

    with pytest.raises(ValueError):
        builder.build(
            subgraph,
            mode="invalid",
        )


def test_missing_positions_raise_error():
    graph = nx.Graph()

    graph.add_node(
        "machine",
        positions=[0],
    )

    graph.add_node(
        "learning",
    )

    graph.add_edge(
        "machine",
        "learning",
    )

    subgraph = SubgraphResult(
        strategy="degree",
        central_node="machine",
        central_score=1.0,
        best_neighbor="learning",
        nodes=["machine", "learning"],
        graph=graph,
    )

    builder = WordSequenceBuilder(
        SequenceConfig(sequence_length=5)
    )

    with pytest.raises(ValueError):
        builder.build(
            subgraph,
            mode="no_branch",
        )