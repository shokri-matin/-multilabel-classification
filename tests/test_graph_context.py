import tempfile

import pytest

from src.corpus.contracts import Corpus
from src.data.contracts import RawDocument
from src.experiments.graph_cache import GraphArtifactCache
from src.experiments.graph_context import (
    DatasetGraphContextBuilder,
    DocumentGraphContext,
    SplitGraphContext,
)
from src.graph.centrality import CentralityCalculator
from src.graph.word_graph import WordGraphBuilder
from src.preprocessing.contracts import ProcessedDocument, ProcessedToken


def make_document(doc_id, labels=None):
    return ProcessedDocument(
        doc_id=doc_id,
        text="alpha beta gamma",
        labels=labels or ["L1"],
        tokens=[
            ProcessedToken(text="alpha", position=0),
            ProcessedToken(text="beta", position=1),
            ProcessedToken(text="gamma", position=2),
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
def components():
    cache_dir = tempfile.mkdtemp()

    cache = GraphArtifactCache(cache_dir)

    graph_builder = WordGraphBuilder(window_size=3)

    centrality_calculator = CentralityCalculator()

    return cache, graph_builder, centrality_calculator


def test_build_dataset_graph_context(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    context = builder.build(
        dataset="aapd",
        train_corpus=make_corpus("train", ["train_1"]),
        validation_corpus=make_corpus("validation", ["val_1"]),
        test_corpus=make_corpus("test", ["test_1"]),
    )

    assert context.dataset == "aapd"
    assert isinstance(context.train, SplitGraphContext)
    assert isinstance(context.validation, SplitGraphContext)
    assert isinstance(context.test, SplitGraphContext)


def test_split_context_has_correct_split(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    context = builder.build(
        "aapd",
        make_corpus("train", ["t1"]),
        make_corpus("validation", ["v1"]),
        make_corpus("test", ["x1"]),
    )

    assert context.train.split == "train"
    assert context.validation.split == "validation"
    assert context.test.split == "test"


def test_builds_graph_for_document_when_cache_missing(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    context = builder.build(
        "aapd",
        make_corpus("train", ["doc_1"]),
        make_corpus("validation", []),
        make_corpus("test", []),
    )

    result = next(context.train.iter_documents())

    assert isinstance(result, DocumentGraphContext)
    assert result.doc_id == "doc_1"
    assert result.graph is not None
    assert result.centralities


def test_all_centrality_strategies_are_cached(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    context = builder.build(
        "aapd",
        make_corpus("train", ["doc_1"]),
        make_corpus("validation", []),
        make_corpus("test", []),
    )

    result = next(context.train.iter_documents())

    expected = {
        "closeness",
        "degree",
        "betweenness",
        "pagerank",
        "clustering",
        "closeness*degree*clustering",
    }

    assert set(result.centralities.keys()) == expected


def test_second_access_reuses_cache(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    context = builder.build(
        "aapd",
        make_corpus("train", ["doc_1"]),
        make_corpus("validation", []),
        make_corpus("test", []),
    )

    first = next(context.train.iter_documents())
    second = next(context.train.iter_documents())

    assert first.doc_id == second.doc_id
    assert first.graph.number_of_nodes() == second.graph.number_of_nodes()
    assert first.graph.number_of_edges() == second.graph.number_of_edges()


def test_get_document_returns_requested_document(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    context = builder.build(
        "aapd",
        make_corpus("train", ["doc_1", "doc_2"]),
        make_corpus("validation", []),
        make_corpus("test", []),
    )

    result = context.train.get_document("doc_2")

    assert result.doc_id == "doc_2"


def test_get_document_raises_for_unknown_document(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    context = builder.build(
        "aapd",
        make_corpus("train", ["doc_1"]),
        make_corpus("validation", []),
        make_corpus("test", []),
    )

    with pytest.raises(KeyError):
        context.train.get_document("unknown")


def test_split_accessor(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    context = builder.build(
        "aapd",
        make_corpus("train", ["t1"]),
        make_corpus("validation", ["v1"]),
        make_corpus("test", ["x1"]),
    )

    assert context.split("train") is context.train
    assert context.split("validation") is context.validation
    assert context.split("test") is context.test


def test_invalid_split_accessor(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    context = builder.build(
        "aapd",
        make_corpus("train", []),
        make_corpus("validation", []),
        make_corpus("test", []),
    )

    with pytest.raises(ValueError):
        context.split("invalid")


def test_invalid_dataset(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    with pytest.raises(ValueError):
        builder.build(
            "unknown",
            make_corpus("train", []),
            make_corpus("validation", []),
            make_corpus("test", []),
        )


def test_invalid_train_split(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    with pytest.raises(ValueError):
        builder.build(
            "aapd",
            make_corpus("validation", []),
            make_corpus("validation", []),
            make_corpus("test", []),
        )


def test_invalid_validation_split(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    with pytest.raises(ValueError):
        builder.build(
            "aapd",
            make_corpus("train", []),
            make_corpus("train", []),
            make_corpus("test", []),
        )


def test_invalid_test_split(components):
    cache, graph_builder, centrality_calculator = components

    builder = DatasetGraphContextBuilder(
        cache=cache,
        graph_builder=graph_builder,
        centrality_calculator=centrality_calculator,
    )

    with pytest.raises(ValueError):
        builder.build(
            "aapd",
            make_corpus("train", []),
            make_corpus("validation", []),
            make_corpus("validation", []),
        )