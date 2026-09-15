import tempfile

import pytest

from src.corpus.contracts import Corpus
from src.experiments.graph_cache import GraphArtifactCache
from src.experiments.graph_context import DatasetGraphContextBuilder
from src.experiments.sequence_cache import SequenceArtifactCache
from src.experiments.sequence_context import (
    DatasetSequenceContextBuilder,
    DocumentSequenceContext,
    SplitSequenceContext,
)
from src.graph.centrality import CentralityCalculator
from src.graph.word_graph import WordGraphBuilder
from src.preprocessing.contracts import ProcessedDocument, ProcessedToken


def make_document(doc_id):
    return ProcessedDocument(
        doc_id=doc_id,
        text="alpha beta gamma delta",
        labels=["L1"],
        tokens=[
            ProcessedToken(text="alpha", position=0),
            ProcessedToken(text="beta", position=1),
            ProcessedToken(text="gamma", position=2),
            ProcessedToken(text="delta", position=3),
        ],
    )


def make_corpus(split, doc_ids):
    return Corpus(
        name="aapd",
        split=split,
        documents=[
            make_document(doc_id)
            for doc_id in doc_ids
        ],
    )


@pytest.fixture
def contexts():
    root = tempfile.mkdtemp()

    graph_cache = GraphArtifactCache(
        root_dir=f"{root}/graphs"
    )

    sequence_cache = SequenceArtifactCache(
        root_dir=f"{root}/sequences"
    )

    graph_builder = WordGraphBuilder(
        window_size=3
    )

    centrality_calculator = CentralityCalculator()

    graph_context_builder = DatasetGraphContextBuilder(
        cache=graph_cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    graph_context = graph_context_builder.build(
        dataset="aapd",
        train_corpus=make_corpus(
            "train",
            ["train_1", "train_2"],
        ),
        validation_corpus=make_corpus(
            "validation",
            ["validation_1"],
        ),
        test_corpus=make_corpus(
            "test",
            ["test_1"],
        ),
    )

    sequence_context_builder = DatasetSequenceContextBuilder(
        graph_context=graph_context,
        sequence_cache=sequence_cache,
        top_k=3,
        max_nodes=10,
        sequence_length=5,
    )

    sequence_context = sequence_context_builder.build(
        dataset="aapd"
    )

    return sequence_context


def test_build_dataset_sequence_context(contexts):
    assert contexts.dataset == "aapd"

    assert isinstance(
        contexts.train,
        SplitSequenceContext,
    )

    assert isinstance(
        contexts.validation,
        SplitSequenceContext,
    )

    assert isinstance(
        contexts.test,
        SplitSequenceContext,
    )


def test_split_contexts_have_correct_split(contexts):
    assert contexts.train.split == "train"
    assert contexts.validation.split == "validation"
    assert contexts.test.split == "test"


def test_top_k_is_propagated(contexts):
    assert contexts.train.top_k == 3
    assert contexts.validation.top_k == 3
    assert contexts.test.top_k == 3


def test_build_no_branch_sequences(contexts):
    result = next(
        contexts.train.iter_documents(
            strategy="closeness",
            mode="no_branch",
        )
    )

    assert isinstance(
        result,
        DocumentSequenceContext,
    )

    assert result.doc_id == "train_1"
    assert result.strategy == "closeness"
    assert result.mode == "no_branch"

    assert result.sequences


def test_build_branch_sequences(contexts):
    result = next(
        contexts.train.iter_documents(
            strategy="closeness",
            mode="branch",
        )
    )

    assert result.doc_id == "train_1"
    assert result.strategy == "closeness"
    assert result.mode == "branch"

    assert result.sequences


def test_sequence_length_is_applied(contexts):
    result = next(
        contexts.train.iter_documents(
            strategy="degree",
            mode="no_branch",
        )
    )

    for sequence in result.sequences:
        assert len(sequence.words) == 5
        assert len(sequence.positions) == 5


def test_all_centrality_strategies_work(contexts):
    strategies = [
        "closeness",
        "degree",
        "betweenness",
        "pagerank",
        "clustering",
        "closeness*degree*clustering",
    ]

    for strategy in strategies:
        result = next(
            contexts.train.iter_documents(
                strategy=strategy,
                mode="no_branch",
            )
        )

        assert result.strategy == strategy
        assert result.sequences


def test_cached_sequences_are_reused(contexts):
    first = next(
        contexts.train.iter_documents(
            strategy="closeness",
            mode="no_branch",
        )
    )

    second = next(
        contexts.train.iter_documents(
            strategy="closeness",
            mode="no_branch",
        )
    )

    assert first.doc_id == second.doc_id
    assert len(first.sequences) == len(second.sequences)


def test_get_document(contexts):
    result = contexts.train.get_document(
        doc_id="train_2",
        strategy="pagerank",
        mode="no_branch",
    )

    assert result.doc_id == "train_2"
    assert result.strategy == "pagerank"
    assert result.mode == "no_branch"
    assert result.sequences


def test_unknown_document_raises(contexts):
    with pytest.raises(KeyError):
        contexts.train.get_document(
            doc_id="unknown",
            strategy="degree",
            mode="no_branch",
        )


def test_invalid_mode_raises(contexts):
    with pytest.raises(ValueError):
        next(
            contexts.train.iter_documents(
                strategy="degree",
                mode="invalid",
            )
        )


def test_invalid_strategy_raises(contexts):
    with pytest.raises(ValueError):
        next(
            contexts.train.iter_documents(
                strategy="invalid",
                mode="no_branch",
            )
        )


def test_split_accessor(contexts):
    assert contexts.split("train") is contexts.train
    assert contexts.split("validation") is contexts.validation
    assert contexts.split("test") is contexts.test


def test_invalid_split_accessor(contexts):
    with pytest.raises(ValueError):
        contexts.split("invalid")


def test_invalid_dataset():
    root = tempfile.mkdtemp()

    graph_cache = GraphArtifactCache(
        root_dir=f"{root}/graphs"
    )

    sequence_cache = SequenceArtifactCache(
        root_dir=f"{root}/sequences"
    )

    graph_builder = WordGraphBuilder(
        window_size=3
    )

    centrality_calculator = CentralityCalculator()

    graph_context_builder = DatasetGraphContextBuilder(
        cache=graph_cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    graph_context = graph_context_builder.build(
        dataset="aapd",
        train_corpus=make_corpus("train", []),
        validation_corpus=make_corpus("validation", []),
        test_corpus=make_corpus("test", []),
    )

    builder = DatasetSequenceContextBuilder(
        graph_context=graph_context,
        sequence_cache=sequence_cache,
        top_k=3,
        max_nodes=10,
        sequence_length=5,
    )

    with pytest.raises(ValueError):
        builder.build("unknown")


def test_invalid_top_k(contexts):
    root = tempfile.mkdtemp()

    sequence_cache = SequenceArtifactCache(
        root_dir=f"{root}/sequences"
    )

    graph_context = contexts

    with pytest.raises(ValueError):
        DatasetSequenceContextBuilder(
            graph_context=graph_context,
            sequence_cache=sequence_cache,
            top_k=0,
            max_nodes=10,
            sequence_length=5,
        )


def test_invalid_max_nodes(contexts):
    root = tempfile.mkdtemp()

    sequence_cache = SequenceArtifactCache(
        root_dir=f"{root}/sequences"
    )

    graph_context = contexts

    with pytest.raises(ValueError):
        DatasetSequenceContextBuilder(
            graph_context=graph_context,
            sequence_cache=sequence_cache,
            top_k=3,
            max_nodes=0,
            sequence_length=5,
        )


def test_invalid_sequence_length(contexts):
    root = tempfile.mkdtemp()

    sequence_cache = SequenceArtifactCache(
        root_dir=f"{root}/sequences"
    )

    graph_context = contexts

    with pytest.raises(ValueError):
        DatasetSequenceContextBuilder(
            graph_context=graph_context,
            sequence_cache=sequence_cache,
            top_k=3,
            max_nodes=10,
            sequence_length=0,
        )

def test_top_k_is_applied_before_graph_processor(monkeypatch, contexts):
    captured = {}

    original_process_strategy = (
        __import__(
            "src.experiments.sequence_context",
            fromlist=["GraphProcessor"],
        ).GraphProcessor.process_strategy
    )

    def spy_process_strategy(self, graph, strategy_result, doc_id=None):
        captured["ranked_nodes"] = list(strategy_result.ranked_nodes)
        captured["top_k_nodes"] = list(strategy_result.top_k_nodes)

        return original_process_strategy(
            self,
            graph,
            strategy_result,
            doc_id=doc_id,
        )

    monkeypatch.setattr(
        "src.experiments.sequence_context.GraphProcessor.process_strategy",
        spy_process_strategy,
    )

    result = next(
        contexts.train.iter_documents(
            strategy="closeness",
            mode="no_branch",
        )
    )

    assert result.sequences

    assert len(captured["ranked_nodes"]) == 3
    assert len(captured["top_k_nodes"]) == 3