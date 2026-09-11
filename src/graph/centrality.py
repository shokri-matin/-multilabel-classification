from dataclasses import dataclass
from typing import Dict, List

import networkx as nx


CENTRALITY_STRATEGIES = [
    "closeness",
    "degree",
    "betweenness",
    "pagerank",
    "clustering",
    "closeness*degree*clustering",
]


@dataclass
class StrategyResult:
    strategy: str
    scores: Dict[str, float]
    ranked_nodes: List[str]
    top_k_nodes: List[str]


class CentralityCalculator:
    """
    Calculate centrality scores and select the top-k nodes.

    Notes
    -----
    - Closeness is calculated without edge weights.
    - Betweenness is calculated without edge weights.
    - Degree centrality is based on graph topology.
    - PageRank uses edge weights.
    - Clustering uses edge weights.
    - The composite strategy uses min-max normalized
      closeness * degree * clustering.
    """

    def __init__(self, strategies: List[str] | None = None):
        self.strategies = strategies or CENTRALITY_STRATEGIES.copy()

        unknown = set(self.strategies) - set(CENTRALITY_STRATEGIES)

        if unknown:
            raise ValueError(
                f"Unknown centrality strategies: {sorted(unknown)}"
            )

    @staticmethod
    def _min_max_normalize(
        scores: Dict[str, float],
    ) -> Dict[str, float]:
        """
        Min-max normalize scores into [0, 1].
        """
        if not scores:
            return {}

        values = list(scores.values())

        min_value = min(values)
        max_value = max(values)

        if max_value == min_value:
            return {
                node: 0.0
                for node in scores
            }

        return {
            node: (score - min_value) / (max_value - min_value)
            for node, score in scores.items()
        }

    @staticmethod
    def _rank_nodes(
        scores: Dict[str, float],
    ) -> List[str]:
        """
        Rank nodes by:

        1. Higher centrality score first.
        2. Lexicographical node order for ties.

        The second rule makes the ranking deterministic.
        """
        return sorted(
            scores,
            key=lambda node: (-scores[node], node),
        )

    def calculate(
        self,
        graph: nx.Graph,
    ) -> Dict[str, Dict[str, float]]:
        """
        Calculate all requested centrality strategies.

        Returns
        -------
        Dict[str, Dict[str, float]]
            Mapping:

            {
                "closeness": {
                    "word1": 0.5,
                    ...
                },
                ...
            }
        """
        if not isinstance(graph, nx.Graph):
            raise TypeError("graph must be a networkx.Graph")

        results: Dict[str, Dict[str, float]] = {}

        if "closeness" in self.strategies:
            results["closeness"] = (
                nx.closeness_centrality(graph)
            )

        if "degree" in self.strategies:
            results["degree"] = (
                nx.degree_centrality(graph)
            )

        if "betweenness" in self.strategies:
            results["betweenness"] = (
                nx.betweenness_centrality(
                    graph,
                    normalized=True,
                )
            )

        if "pagerank" in self.strategies:
            results["pagerank"] = (
                nx.pagerank(
                    graph,
                    weight="weight",
                )
            )

        if "clustering" in self.strategies:
            results["clustering"] = (
                nx.clustering(
                    graph,
                    weight="weight",
                )
            )

        if "closeness*degree*clustering" in self.strategies:
            closeness = self._min_max_normalize(
                nx.closeness_centrality(graph)
            )

            degree = self._min_max_normalize(
                nx.degree_centrality(graph)
            )

            clustering = self._min_max_normalize(
                nx.clustering(
                    graph,
                    weight="weight",
                )
            )

            results["closeness*degree*clustering"] = {
                node: (
                    closeness[node]
                    * degree[node]
                    * clustering[node]
                )
                for node in graph.nodes
            }

        return results

    def rank(
        self,
        graph: nx.Graph,
        top_k: int | None = None,
    ) -> Dict[str, StrategyResult]:
        """
        Calculate, rank and optionally select top-k nodes
        for every strategy.
        """
        if top_k is not None and top_k <= 0:
            raise ValueError("top_k must be positive")

        centralities = self.calculate(graph)

        results: Dict[str, StrategyResult] = {}

        for strategy, scores in centralities.items():
            ranked_nodes = self._rank_nodes(scores)

            if top_k is None:
                top_k_nodes = ranked_nodes.copy()
            else:
                top_k_nodes = ranked_nodes[:top_k]

            results[strategy] = StrategyResult(
                strategy=strategy,
                scores=scores,
                ranked_nodes=ranked_nodes,
                top_k_nodes=top_k_nodes,
            )

        return results

