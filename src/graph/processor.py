from dataclasses import dataclass
from typing import Dict, List, Literal

import networkx as nx

from src.graph.centrality import StrategyResult


ExtractionMode = Literal["branch", "no_branch"]


@dataclass
class SubgraphResult:
    """
    Result of extracting one subgraph from one central node.
    """

    strategy: str
    central_node: str
    central_score: float
    best_neighbor: str
    nodes: List[str]
    graph: nx.Graph


@dataclass
class StrategySubgraphResult:
    """
    All extracted subgraphs for one centrality strategy.
    """

    strategy: str
    mode: ExtractionMode
    subgraphs: List[SubgraphResult]


class GraphProcessor:
    """
    Extract Branch or No-Branch subgraphs from word graphs.

    Pipeline:

        Top-W central nodes
                ↓
        first-order neighbors
                ↓
        best neighbor by centrality
                ↓
        DFS
                ↓
        Branch / No-Branch
    """

    def __init__(
        self,
        max_nodes: int,
        mode: ExtractionMode,
    ):
        if max_nodes <= 0:
            raise ValueError(
                "max_nodes must be positive"
            )

        if mode not in {"branch", "no_branch"}:
            raise ValueError(
                "mode must be 'branch' or 'no_branch'"
            )

        self.max_nodes = max_nodes
        self.mode = mode

    @staticmethod
    def _select_best_neighbor(
        graph: nx.Graph,
        central_node: str,
        scores: Dict[str, float],
    ) -> str | None:
        """
        Select the highest-centrality first-order neighbor.

        Ties are resolved lexicographically to guarantee
        deterministic behavior.
        """
        neighbors = list(
            graph.neighbors(central_node)
        )

        if not neighbors:
            return None

        return min(
            neighbors,
            key=lambda node: (
                -scores[node],
                node,
            ),
        )

    @staticmethod
    def _build_subgraph(
        graph: nx.Graph,
        nodes: List[str],
    ) -> nx.Graph:
        """
        Create an independent copy of the induced subgraph.
        """
        return graph.subgraph(nodes).copy()

    def _extract_no_branch(
        self,
        graph: nx.Graph,
        central_node: str,
        best_neighbor: str,
    ) -> List[str]:
        """
        Extract one combined DFS subgraph.

        The central node and best neighbor are always included.

        All nodes visited by DFS from the best neighbor are
        combined into a single subgraph.

        The total number of nodes cannot exceed max_nodes.
        """

        selected = [
            central_node,
            best_neighbor,
        ]

        if self.max_nodes <= 2:
            return selected[:self.max_nodes]

        visited = set(selected)

        for node in nx.dfs_preorder_nodes(
            graph,
            source=best_neighbor,
        ):
            if node in visited:
                continue

            if len(selected) >= self.max_nodes:
                break

            selected.append(node)
            visited.add(node)

        return selected

    def _extract_branch(
        self,
        graph: nx.Graph,
        central_node: str,
        best_neighbor: str,
    ) -> List[List[str]]:
        base_nodes = [central_node, best_neighbor]

        if self.max_nodes <= 2:
            return [base_nodes[:self.max_nodes]]

        dfs_tree = nx.dfs_tree(
            graph,
            source=best_neighbor,
        )

        children = sorted(
            child
            for child in dfs_tree.successors(best_neighbor)
            if child != central_node
        )

        if not children:
            return [base_nodes]

        branches = []

        for child in children:
            branch_nodes = list(
                nx.dfs_preorder_nodes(
                    dfs_tree,
                    source=child,
                )
            )

            selected = base_nodes.copy()

            for node in branch_nodes:
                if len(selected) >= self.max_nodes:
                    break

                if node not in selected:
                    selected.append(node)

            branches.append(selected)

        return branches
    
    def process_strategy(
        self,
        graph: nx.Graph,
        strategy_result: StrategyResult,
    ) -> StrategySubgraphResult:
        """
        Extract subgraphs for all Top-W nodes of one strategy.
        """

        scores = strategy_result.scores

        subgraphs: List[SubgraphResult] = []

        for central_node in strategy_result.top_k_nodes:

            central_score = scores[central_node]

            best_neighbor = self._select_best_neighbor(
                graph=graph,
                central_node=central_node,
                scores=scores,
            )

            # Isolated node.
            if best_neighbor is None:

                subgraph_graph = self._build_subgraph(
                    graph,
                    [central_node],
                )

                subgraphs.append(
                    SubgraphResult(
                        strategy=strategy_result.strategy,
                        central_node=central_node,
                        central_score=central_score,
                        best_neighbor=central_node,
                        nodes=[central_node],
                        graph=subgraph_graph,
                    )
                )

                continue

            if self.mode == "no_branch":

                node_groups = [
                    self._extract_no_branch(
                        graph=graph,
                        central_node=central_node,
                        best_neighbor=best_neighbor,
                    )
                ]

            else:

                node_groups = self._extract_branch(
                    graph=graph,
                    central_node=central_node,
                    best_neighbor=best_neighbor,
                )

            for nodes in node_groups:

                subgraph_graph = self._build_subgraph(
                    graph,
                    nodes,
                )

                subgraphs.append(
                    SubgraphResult(
                        strategy=strategy_result.strategy,
                        central_node=central_node,
                        central_score=central_score,
                        best_neighbor=best_neighbor,
                        nodes=nodes,
                        graph=subgraph_graph,
                    )
                )

        return StrategySubgraphResult(
            strategy=strategy_result.strategy,
            mode=self.mode,
            subgraphs=subgraphs,
        )

    def process_all(
        self,
        graph: nx.Graph,
        strategy_results: Dict[str, StrategyResult],
    ) -> Dict[str, StrategySubgraphResult]:
        """
        Extract subgraphs for all centrality strategies.
        """

        return {
            strategy: self.process_strategy(
                graph=graph,
                strategy_result=result,
            )
            for strategy, result in strategy_results.items()
        }

