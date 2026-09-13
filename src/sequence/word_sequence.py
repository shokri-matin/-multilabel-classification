from dataclasses import dataclass
from typing import List

from src.graph.processor import SubgraphResult


@dataclass(frozen=True)
class SequenceConfig:
    """
    Configuration for converting a subgraph into a fixed-length
    word sequence.
    """

    sequence_length: int
    padding_token: str = "<PAD>"

    def __post_init__(self):
        if self.sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        if not self.padding_token:
            raise ValueError(
                "padding_token must not be empty"
            )


@dataclass
class WordSequence:
    """
    Fixed-length word sequence extracted from a subgraph.
    """
    doc_id: str
    strategy: str
    mode: str

    central_node: str
    best_neighbor: str

    words: List[str]
    positions: List[int]

    original_length: int

    @property
    def length(self) -> int:
        return len(self.words)

    @property
    def is_padded(self) -> bool:
        return self.original_length < len(self.words)

    @property
    def is_truncated(self) -> bool:
        return self.original_length > len(self.words)


class WordSequenceBuilder:
    """
    Convert a SubgraphResult into a fixed-length word sequence.

    Node ordering is determined by the earliest position of each
    word in the original document.
    """

    def __init__(self, config: SequenceConfig):
        self.config = config

    @staticmethod
    def _get_node_position(
        graph,
        node: str,
    ) -> int:
        """
        Return the earliest original document position of a node.
        """

        positions = graph.nodes[node].get(
            "positions",
            [],
        )

        if not positions:
            raise ValueError(
                f"Node '{node}' does not have positions"
            )

        return min(positions)

    def _sort_nodes_by_position(
        self,
        subgraph: SubgraphResult,
    ) -> List[tuple[str, int]]:
        """
        Sort subgraph nodes according to their original
        document positions.

        Lexical ordering is used as a deterministic tie-breaker.
        """

        graph = subgraph.graph

        node_positions = []

        for node in graph.nodes:

            position = self._get_node_position(
                graph,
                node,
            )

            node_positions.append(
                (node, position)
            )

        node_positions.sort(
            key=lambda item: (
                item[1],
                item[0],
            )
        )

        return node_positions

    def build(
        self,
        subgraph: SubgraphResult,
        mode: str,
    ) -> WordSequence:
        """
        Convert one SubgraphResult into one fixed-length
        WordSequence.
        """

        if mode not in {"branch", "no_branch"}:
            raise ValueError(
                "mode must be 'branch' or 'no_branch'"
            )

        sorted_nodes = self._sort_nodes_by_position(
            subgraph
        )

        original_words = [
            node
            for node, _ in sorted_nodes
        ]

        original_positions = [
            position
            for _, position in sorted_nodes
        ]

        original_length = len(original_words)

        target_length = self.config.sequence_length

        words = original_words[:target_length]
        positions = original_positions[:target_length]

        padding_count = target_length - len(words)

        if padding_count > 0:
            words.extend(
                [self.config.padding_token]
                * padding_count
            )

            positions.extend(
                [-1] * padding_count
            )

        return WordSequence(
            doc_id=subgraph.doc_id,
            strategy=subgraph.strategy,
            mode=mode,
            central_node=subgraph.central_node,
            best_neighbor=subgraph.best_neighbor,
            words=words,
            positions=positions,
            original_length=original_length,
        )