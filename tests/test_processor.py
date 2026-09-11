import networkx as nx
import pytest

from src.graph.centrality import StrategyResult
from src.graph.processor import GraphProcessor


def build_branching_graph() -> nx.Graph:
    """
    Build a deterministic graph:

                    C
                    |
                    B
                  / | \
                 A  D  E
                / \    |
               A1 A2   E1

    C = central node
    B = best first-order neighbor

    B has three DFS branches:
        A -> A1, A2
        D
        E -> E1
    """

    graph = nx.Graph()

    graph.add_edges_from([
        ("C", "B"),

        ("B", "A"),
        ("B", "D"),
        ("B", "E"),

        ("A", "A1"),
        ("A", "A2"),

        ("E", "E1"),
    ])

    return graph


def build_strategy_result(
    graph: nx.Graph,
    top_k_nodes=None,
) -> StrategyResult:
    """
    Create a deterministic StrategyResult for testing.

    B has the highest centrality among C's neighbors.
    """

    scores = {
        "C": 0.50,
        "B": 0.90,
        "A": 0.70,
        "D": 0.60,
        "E": 0.80,
        "A1": 0.30,
        "A2": 0.20,
        "E1": 0.10,
    }

    ranked_nodes = sorted(
        graph.nodes,
        key=lambda node: (
            -scores[node],
            node,
        ),
    )

    if top_k_nodes is None:
        top_k_nodes = ["C"]

    return StrategyResult(
        strategy="degree",
        scores=scores,
        ranked_nodes=ranked_nodes,
        top_k_nodes=top_k_nodes,
    )


def test_best_neighbor_is_selected_by_centrality():
    graph = build_branching_graph()

    scores = {
        "C": 0.50,
        "B": 0.90,
        "A": 0.70,
        "D": 0.60,
        "E": 0.80,
        "A1": 0.30,
        "A2": 0.20,
        "E1": 0.10,
    }

    best_neighbor = GraphProcessor._select_best_neighbor(
        graph=graph,
        central_node="C",
        scores=scores,
    )

    assert best_neighbor == "B"


def test_best_neighbor_tie_is_lexical():
    graph = nx.Graph()

    graph.add_edges_from([
        ("C", "zebra"),
        ("C", "apple"),
        ("C", "banana"),
    ])

    scores = {
        "C": 1.0,
        "zebra": 0.5,
        "apple": 0.5,
        "banana": 0.5,
    }

    best_neighbor = GraphProcessor._select_best_neighbor(
        graph=graph,
        central_node="C",
        scores=scores,
    )

    assert best_neighbor == "apple"


def test_no_branch_creates_one_combined_subgraph():
    graph = build_branching_graph()

    strategy_result = build_strategy_result(graph)

    processor = GraphProcessor(
        max_nodes=20,
        mode="no_branch",
    )

    result = processor.process_strategy(
        graph=graph,
        strategy_result=strategy_result,
    )

    assert result.strategy == "degree"
    assert result.mode == "no_branch"

    # One central node -> exactly one subgraph.
    assert len(result.subgraphs) == 1

    subgraph = result.subgraphs[0]

    assert subgraph.central_node == "C"
    assert subgraph.best_neighbor == "B"

    expected_nodes = {
        "C",
        "B",
        "A",
        "D",
        "E",
        "A1",
        "A2",
        "E1",
    }

    assert set(subgraph.nodes) == expected_nodes

    assert set(subgraph.graph.nodes) == expected_nodes


def test_branch_creates_separate_subgraphs():
    graph = build_branching_graph()

    strategy_result = build_strategy_result(graph)

    processor = GraphProcessor(
        max_nodes=20,
        mode="branch",
    )

    result = processor.process_strategy(
        graph=graph,
        strategy_result=strategy_result,
    )

    assert result.strategy == "degree"
    assert result.mode == "branch"

    # B has three direct DFS children:
    #
    # A
    # D
    # E
    #
    # Therefore we expect three branches.
    assert len(result.subgraphs) == 3

    branch_node_sets = [
        set(subgraph.nodes)
        for subgraph in result.subgraphs
    ]

    expected_branches = [
        {
            "C",
            "B",
            "A",
            "A1",
            "A2",
        },
        {
            "C",
            "B",
            "D",
        },
        {
            "C",
            "B",
            "E",
            "E1",
        },
    ]

    assert all(
        expected in branch_node_sets
        for expected in expected_branches
    )


def test_central_and_best_neighbor_are_always_included():
    graph = build_branching_graph()

    strategy_result = build_strategy_result(graph)

    for mode in ("branch", "no_branch"):

        processor = GraphProcessor(
            max_nodes=20,
            mode=mode,
        )

        result = processor.process_strategy(
            graph=graph,
            strategy_result=strategy_result,
        )

        for subgraph in result.subgraphs:

            assert "C" in subgraph.nodes
            assert "B" in subgraph.nodes

            assert (
                subgraph.central_node == "C"
            )

            assert (
                subgraph.best_neighbor == "B"
            )


def test_no_branch_respects_max_nodes():
    graph = build_branching_graph()

    strategy_result = build_strategy_result(graph)

    processor = GraphProcessor(
        max_nodes=5,
        mode="no_branch",
    )

    result = processor.process_strategy(
        graph=graph,
        strategy_result=strategy_result,
    )

    assert len(result.subgraphs) == 1

    subgraph = result.subgraphs[0]

    assert len(subgraph.nodes) <= 5


def test_branch_respects_max_nodes_per_branch():
    graph = build_branching_graph()

    strategy_result = build_strategy_result(graph)

    processor = GraphProcessor(
        max_nodes=4,
        mode="branch",
    )

    result = processor.process_strategy(
        graph=graph,
        strategy_result=strategy_result,
    )

    assert len(result.subgraphs) == 3

    for subgraph in result.subgraphs:
        assert len(subgraph.nodes) <= 4


def test_isolated_central_node():
    graph = nx.Graph()

    graph.add_node("isolated")

    scores = {
        "isolated": 1.0,
    }

    strategy_result = StrategyResult(
        strategy="degree",
        scores=scores,
        ranked_nodes=["isolated"],
        top_k_nodes=["isolated"],
    )

    processor = GraphProcessor(
        max_nodes=10,
        mode="no_branch",
    )

    result = processor.process_strategy(
        graph=graph,
        strategy_result=strategy_result,
    )

    assert len(result.subgraphs) == 1

    subgraph = result.subgraphs[0]

    assert subgraph.central_node == "isolated"
    assert subgraph.best_neighbor == "isolated"
    assert subgraph.nodes == ["isolated"]
    assert list(subgraph.graph.nodes) == ["isolated"]


def test_top_w_central_nodes_only():
    graph = build_branching_graph()

    scores = {
        "C": 0.90,
        "B": 0.80,
        "A": 0.70,
        "D": 0.60,
        "E": 0.50,
        "A1": 0.40,
        "A2": 0.30,
        "E1": 0.20,
    }

    strategy_result = StrategyResult(
        strategy="degree",
        scores=scores,
        ranked_nodes=[
            "C",
            "B",
            "A",
            "D",
            "E",
            "A1",
            "A2",
            "E1",
        ],
        top_k_nodes=["C", "B"],
    )

    processor = GraphProcessor(
        max_nodes=20,
        mode="no_branch",
    )

    result = processor.process_strategy(
        graph=graph,
        strategy_result=strategy_result,
    )

    # Only C and B are Top-W.
    central_nodes = {
        subgraph.central_node
        for subgraph in result.subgraphs
    }

    assert central_nodes == {
        "C",
        "B",
    }


def test_invalid_max_nodes():
    with pytest.raises(ValueError):
        GraphProcessor(
            max_nodes=0,
            mode="branch",
        )


def test_invalid_mode():
    with pytest.raises(ValueError):
        GraphProcessor(
            max_nodes=10,
            mode="invalid",
        )
