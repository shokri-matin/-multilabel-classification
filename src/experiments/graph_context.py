from dataclasses import dataclass
from typing import Dict, Iterable, Iterator

from src.corpus.contracts import Corpus
from src.graph.centrality import CentralityCalculator, StrategyResult
from src.graph.word_graph import WordGraphBuilder, WordGraph
from src.experiments.graph_cache import GraphArtifactCache

@dataclass(frozen=True)
class DocumentGraphContext:
    """
    Graph and centrality artifacts for a single document.
    """

    doc_id: str
    graph: object
    centralities: Dict[str, StrategyResult]


@dataclass
class SplitGraphContext:
    """
    Lazy graph context for one corpus split.

    Documents are processed incrementally to avoid keeping the
    entire dataset in memory.
    """

    dataset: str
    split: str
    corpus: Corpus
    cache: GraphArtifactCache
    graph_builder: WordGraphBuilder
    centrality_calculator: CentralityCalculator

    def iter_documents(self) -> Iterator[DocumentGraphContext]:
        """
        Yield graph contexts one document at a time.

        Cached artifacts are reused whenever available.
        """
        for document in self.corpus.documents:
            yield self._get_or_build(document)

    def get_document(self, doc_id: str) -> DocumentGraphContext:
        """
        Load or build graph artifacts for one document.
        """
        for document in self.corpus.documents:
            if document.doc_id == doc_id:
                return self._get_or_build(document)

        raise KeyError(
            f"Document '{doc_id}' was not found in split '{self.split}'"
        )

    def _get_or_build(self, document) -> DocumentGraphContext:
        """
        Load cached graph artifacts or build them if missing.
        """
        if self.cache.exists(document.doc_id):
            graph, centralities = self.cache.load(document.doc_id)

            return DocumentGraphContext(
                doc_id=document.doc_id,
                graph=graph,
                centralities=centralities,
            )

        word_graph = self.graph_builder.build(document)

        if not isinstance(word_graph, WordGraph):
            raise TypeError(
                "graph_builder.build() must return a WordGraph"
            )

        graph = word_graph.graph

        centralities = self.centrality_calculator.rank(graph)

        self.cache.save(
            document.doc_id,
            graph,
            centralities,
        )

        return DocumentGraphContext(
            doc_id=document.doc_id,
            graph=graph,
            centralities=centralities,
        )


@dataclass
class DatasetGraphContext:
    """
    Reusable graph preparation context for one dataset.

    The context contains lazy split-level graph processors.
    """

    dataset: str
    train: SplitGraphContext
    validation: SplitGraphContext
    test: SplitGraphContext

    def split(self, split_name: str) -> SplitGraphContext:
        if split_name == "train":
            return self.train

        if split_name == "validation":
            return self.validation

        if split_name == "test":
            return self.test

        raise ValueError(
            f"Unsupported split: {split_name}"
        )


class DatasetGraphContextBuilder:
    """
    Build graph preparation contexts for train/validation/test.

    Graphs and all centrality strategies are cached per document.
    """

    SUPPORTED_DATASETS = {"aapd", "rcv1"}

    def __init__(
        self,
        cache: GraphArtifactCache,
        graph_builder: WordGraphBuilder,
        centrality_calculator: CentralityCalculator,
    ):
        self.cache = cache
        self.graph_builder = graph_builder
        self.centrality_calculator = centrality_calculator

    def build(
        self,
        dataset: str,
        train_corpus: Corpus,
        validation_corpus: Corpus,
        test_corpus: Corpus,
    ) -> DatasetGraphContext:

        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset}"
            )

        self._validate_split(train_corpus, "train")
        self._validate_split(validation_corpus, "validation")
        self._validate_split(test_corpus, "test")

        return DatasetGraphContext(
            dataset=dataset,
            train=SplitGraphContext(
                dataset=dataset,
                split="train",
                corpus=train_corpus,
                cache=self.cache,
                graph_builder=self.graph_builder,
                centrality_calculator=self.centrality_calculator,
            ),
            validation=SplitGraphContext(
                dataset=dataset,
                split="validation",
                corpus=validation_corpus,
                cache=self.cache,
                graph_builder=self.graph_builder,
                centrality_calculator=self.centrality_calculator,
            ),
            test=SplitGraphContext(
                dataset=dataset,
                split="test",
                corpus=test_corpus,
                cache=self.cache,
                graph_builder=self.graph_builder,
                centrality_calculator=self.centrality_calculator,
            ),
        )

    @staticmethod
    def _validate_split(
        corpus: Corpus,
        expected_split: str,
    ) -> None:

        if not isinstance(corpus, Corpus):
            raise TypeError("corpus must be a Corpus")

        if corpus.split != expected_split:
            raise ValueError(
                f"Expected corpus split '{expected_split}', "
                f"got '{corpus.split}'"
            )