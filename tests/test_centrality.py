
import networkx as nx
import pytest
import math
from src.graph.centrality import (
    CENTRALITY_STRATEGIES,
    CentralityCalculator,
)


def build_test_graph() -> nx.Graph:
    """
    Create a deterministic weighted graph.

             A
            / \
           B---C
           |
           D
    """
    graph = nx.Graph()

    graph.add_edge("A", "B", weight=3)
    graph.add_edge("A", "C", weight=2)
    graph.add_edge("B", "C", weight=4)
    graph.add_edge("B", "D", weight=1)

    return graph


def test_all_strategies_are_available():
    calculator = CentralityCalculator()

    assert calculator.strategies == CENTRALITY_STRATEGIES


def test_centrality_calculation():
    graph = build_test_graph()

    calculator = CentralityCalculator()

    results = calculator.calculate(graph)

    assert set(results.keys()) == set(
        CENTRALITY_STRATEGIES
    )

    for strategy in CENTRALITY_STRATEGIES:
        scores = results[strategy]

        assert set(scores.keys()) == set(graph.nodes)

        for score in scores.values():
            assert isinstance(score, (int, float))
            assert math.isfinite(score)
            assert score >= 0.0


def test_composite_centrality_is_in_zero_one():
    graph = build_test_graph()

    calculator = CentralityCalculator()

    results = calculator.calculate(graph)

    composite = results[
        "closeness*degree*clustering"
    ]

    assert all(
        0.0 <= score <= 1.0
        for score in composite.values()
    )


def test_ranking_is_descending():
    graph = build_test_graph()

    calculator = CentralityCalculator()

    results = calculator.rank(graph)

    for strategy, result in results.items():
        scores = result.scores
        ranked_nodes = result.ranked_nodes

        for first, second in zip(
            ranked_nodes,
            ranked_nodes[1:],
        ):
            assert scores[first] >= scores[second]


def test_ranking_tie_break_is_lexical():
    calculator = CentralityCalculator()

    scores = {
        "banana": 0.5,
        "apple": 0.5,
        "cherry": 0.8,
    }

    ranked = calculator._rank_nodes(scores)

    assert ranked == [
        "cherry",
        "apple",
        "banana",
    ]


def test_top_k():
    graph = build_test_graph()

    calculator = CentralityCalculator()

    results = calculator.rank(
        graph,
        top_k=2,
    )

    for result in results.values():
        assert len(result.top_k_nodes) == 2
        assert result.top_k_nodes == result.ranked_nodes[:2]


def test_top_k_cannot_be_zero():
    graph = build_test_graph()

    calculator = CentralityCalculator()

    with pytest.raises(ValueError):
        calculator.rank(
            graph,
            top_k=0,
        )


def test_empty_graph():
    graph = nx.Graph()

    calculator = CentralityCalculator()

    results = calculator.calculate(graph)

    for strategy in CENTRALITY_STRATEGIES:
        assert results[strategy] == {}


def test_invalid_strategy():
    with pytest.raises(ValueError):
        CentralityCalculator(
            strategies=["invalid_strategy"]
        )

