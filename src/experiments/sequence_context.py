from dataclasses import dataclass
from typing import Iterator, List
from dataclasses import replace

from src.experiments.graph_context import DocumentGraphContext, DatasetGraphContext
from src.experiments.sequence_cache import SequenceArtifactCache
from src.graph.processor import GraphProcessor
from src.sequence.word_sequence import SequenceConfig, WordSequence, WordSequenceBuilder


@dataclass(frozen=True)
class DocumentSequenceContext:
    """
    Sequence artifacts for one document and one experiment configuration.
    """

    doc_id: str
    strategy: str
    mode: str
    sequences: List[WordSequence]


@dataclass
class SplitSequenceContext:
    """
    Lazy sequence preparation context for one corpus split.

    Sequences are generated document-by-document and cached.
    """

    dataset: str
    split: str
    graph_context: object
    sequence_cache: SequenceArtifactCache
    top_k: int
    max_nodes: int
    sequence_length: int

    def iter_documents(
        self,
        strategy: str,
        mode: str,
    ) -> Iterator[DocumentSequenceContext]:

        self._validate_mode(mode)

        for graph_document in self.graph_context.iter_documents():
            yield self._get_or_build(
                graph_document=graph_document,
                strategy=strategy,
                mode=mode,
            )

    def get_document(
        self,
        doc_id: str,
        strategy: str,
        mode: str,
    ) -> DocumentSequenceContext:

        self._validate_mode(mode)

        graph_document = self.graph_context.get_document(doc_id)

        return self._get_or_build(
            graph_document=graph_document,
            strategy=strategy,
            mode=mode,
        )

    def _get_or_build(
        self,
        graph_document: DocumentGraphContext,
        strategy: str,
        mode: str,
    ) -> DocumentSequenceContext:

        doc_id = graph_document.doc_id

        if mode not in {"branch", "no_branch"}:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )

        if strategy not in graph_document.centralities:
            raise ValueError(
                f"Unknown centrality strategy: {strategy}"
            )

        if self.sequence_cache.exists(
            dataset=self.dataset,
            doc_id=doc_id,
            centrality=strategy,
            mode=mode,
        ):
            sequences = self.sequence_cache.load(
                dataset=self.dataset,
                doc_id=doc_id,
                centrality=strategy,
                mode=mode,
            )

            return DocumentSequenceContext(
                doc_id=doc_id,
                strategy=strategy,
                mode=mode,
                sequences=sequences,
            )



        centrality_result = graph_document.centralities[strategy]

        centrality_result = replace(
            centrality_result,
            ranked_nodes=centrality_result.ranked_nodes[
                :self.top_k
            ],
            top_k_nodes=centrality_result.top_k_nodes[
                :self.top_k
            ],
        )

        processor = GraphProcessor(
            max_nodes=self.max_nodes,
            mode=mode,
        )

        strategy_subgraph_result = processor.process_strategy(
            graph=graph_document.graph,
            strategy_result=centrality_result,
            doc_id=doc_id,
        )

        sequence_builder = WordSequenceBuilder(
            SequenceConfig(
                sequence_length=self.sequence_length,
            )
        )

        sequences = [
            sequence_builder.build(
                subgraph=subgraph,
                mode=mode,
            )
            for subgraph in strategy_subgraph_result.subgraphs
        ]

        self.sequence_cache.save(
            dataset=self.dataset,
            doc_id=doc_id,
            centrality=strategy,
            mode=mode,
            sequences=sequences,
        )

        return DocumentSequenceContext(
            doc_id=doc_id,
            strategy=strategy,
            mode=mode,
            sequences=sequences,
        )

    @staticmethod
    def _validate_mode(mode: str) -> None:
        if mode not in {"branch", "no_branch"}:
            raise ValueError(
                f"Unsupported mode: {mode}"
            )


@dataclass
class DatasetSequenceContext:
    """
    Lazy sequence preparation context for one dataset.
    """

    dataset: str
    train: SplitSequenceContext
    validation: SplitSequenceContext
    test: SplitSequenceContext

    def split(self, split_name: str) -> SplitSequenceContext:
        if split_name == "train":
            return self.train

        if split_name == "validation":
            return self.validation

        if split_name == "test":
            return self.test

        raise ValueError(
            f"Unsupported split: {split_name}"
        )


class DatasetSequenceContextBuilder:
    """
    Build reusable sequence contexts on top of graph artifacts.
    """

    SUPPORTED_DATASETS = {"aapd", "rcv1"}

    def __init__(
        self,
        graph_context: DatasetGraphContext,
        sequence_cache: SequenceArtifactCache,
        top_k: int,
        max_nodes: int,
        sequence_length: int,
    ):
        if top_k <= 0:
            raise ValueError(
                "top_k must be greater than zero"
            )

        if max_nodes <= 0:
            raise ValueError(
                "max_nodes must be greater than zero"
            )

        if sequence_length <= 0:
            raise ValueError(
                "sequence_length must be greater than zero"
            )

        self.graph_context = graph_context
        self.sequence_cache = sequence_cache
        self.top_k = top_k
        self.max_nodes = max_nodes
        self.sequence_length = sequence_length

    def build(
        self,
        dataset: str,
    ) -> DatasetSequenceContext:

        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset}"
            )

        if self.graph_context.dataset != dataset:
            raise ValueError(
                "graph_context dataset does not match dataset"
            )

        return DatasetSequenceContext(
            dataset=dataset,
            train=self._build_split("train"),
            validation=self._build_split("validation"),
            test=self._build_split("test"),
        )

    def _build_split(
        self,
        split: str,
    ) -> SplitSequenceContext:

        return SplitSequenceContext(
            dataset=self.graph_context.dataset,
            split=split,
            graph_context=self.graph_context.split(split),
            sequence_cache=self.sequence_cache,
            top_k=self.top_k,
            max_nodes=self.max_nodes,
            sequence_length=self.sequence_length,
        )