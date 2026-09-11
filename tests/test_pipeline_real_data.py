from pathlib import Path

from src.corpus.contracts import Corpus
from src.corpus.storage import CorpusStorage
from src.data.aapd import AAPDLoader
from src.data.rcv1 import RCV1Loader
from src.graph.word_graph import WordGraphBuilder
from src.preprocessing.pipeline import PreprocessingConfig
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.tokenizer import (
    AAPDTokenizer,
    WhitespaceTokenizer,
)
import networkx as nx

SAMPLE_SIZE = 5
PROCESSED_ROOT = Path("data/processed")


def test_aapd_pipeline_with_real_data(tmp_path):
    # ---------------------------------------------------------
    # 1. Load AAPD
    # ---------------------------------------------------------

    loader = AAPDLoader("data/raw/aapd")

    raw_documents = loader.load_split("train")

    assert len(raw_documents) > 0

    sample = raw_documents[:SAMPLE_SIZE]

    # ---------------------------------------------------------
    # 2. Preprocessing
    # ---------------------------------------------------------

    config = PreprocessingConfig(
        lowercase=True,
        remove_stopwords=True,
        stemming=True,
    )

    pipeline = PreprocessingPipeline(
        tokenizer=AAPDTokenizer(),
        config=config,
    )

    processed_documents = pipeline.process_many(
        sample,
        show_progress=True,
    )

    assert len(processed_documents) == len(sample)

    for raw, processed in zip(
        sample,
        processed_documents,
    ):
        assert processed.doc_id == raw.doc_id
        assert processed.labels == raw.labels
        assert processed.token_count > 0

    # ---------------------------------------------------------
    # 3. Create Corpus
    # ---------------------------------------------------------

    corpus = Corpus(
        name="aapd",
        split="train",
        documents=processed_documents,
    )

    assert corpus.size == SAMPLE_SIZE

    # ---------------------------------------------------------
    # 4. Save / Load
    # ---------------------------------------------------------

    storage = CorpusStorage(tmp_path)

    storage.save(corpus)

    loaded_corpus = storage.load(
        name="aapd",
        split="train",
    )

    assert loaded_corpus.size == corpus.size
    assert loaded_corpus.document_ids == corpus.document_ids
    assert loaded_corpus.labels == corpus.labels

    # ---------------------------------------------------------
    # 5. Build Word Graph
    # ---------------------------------------------------------

    graph_builder = WordGraphBuilder(
        window_size=3,
    )

    for document in loaded_corpus.documents:

        word_graph = graph_builder.build(document)

        assert word_graph.doc_id == document.doc_id
        assert word_graph.graph.number_of_nodes() > 0

        assert nx.number_of_selfloops(
            word_graph.graph
        ) == 0

        # Every node must have positions
        for node in word_graph.graph.nodes:
            assert "positions" in word_graph.graph.nodes[node]
            assert len(
                word_graph.graph.nodes[node]["positions"]
            ) > 0

        # Every edge must have weight
        for source, target, data in word_graph.graph.edges(
            data=True
        ):
            assert source != target
            assert "weight" in data
            assert data["weight"] >= 1


def test_rcv1_pipeline_with_real_data(tmp_path):
    # ---------------------------------------------------------
    # 1. Load RCV1
    # ---------------------------------------------------------

    loader = RCV1Loader("data/raw/rcv1")

    raw_documents = loader.load_split("train")

    assert len(raw_documents) > 0

    sample = raw_documents[:SAMPLE_SIZE]

    # ---------------------------------------------------------
    # 2. Preprocessing
    # ---------------------------------------------------------

    config = PreprocessingConfig(
        lowercase=False,
        remove_stopwords=False,
        stemming=False,
    )

    pipeline = PreprocessingPipeline(
        tokenizer=WhitespaceTokenizer(),
        config=config,
    )

    processed_documents = pipeline.process_many(
        sample,
        show_progress=True,
    )

    assert len(processed_documents) == len(sample)

    for raw, processed in zip(
        sample,
        processed_documents,
    ):
        assert processed.doc_id == raw.doc_id
        assert processed.labels == raw.labels
        assert processed.token_count > 0

    # ---------------------------------------------------------
    # 3. Create Corpus
    # ---------------------------------------------------------

    corpus = Corpus(
        name="rcv1",
        split="train",
        documents=processed_documents,
    )

    assert corpus.size == SAMPLE_SIZE

    # ---------------------------------------------------------
    # 4. Save / Load
    # ---------------------------------------------------------

    storage = CorpusStorage(tmp_path)

    storage.save(corpus)

    loaded_corpus = storage.load(
        name="rcv1",
        split="train",
    )

    assert loaded_corpus.size == corpus.size
    assert loaded_corpus.document_ids == corpus.document_ids
    assert loaded_corpus.labels == corpus.labels

    # ---------------------------------------------------------
    # 5. Build Word Graph
    # ---------------------------------------------------------

    graph_builder = WordGraphBuilder(
        window_size=3,
    )

    for document in loaded_corpus.documents:

        word_graph = graph_builder.build(document)

        assert word_graph.doc_id == document.doc_id
        assert word_graph.graph.number_of_nodes() > 0

        for node in word_graph.graph.nodes:
            assert "positions" in word_graph.graph.nodes[node]
            assert len(
                word_graph.graph.nodes[node]["positions"]
            ) > 0

        for source, target, data in word_graph.graph.edges(
            data=True
        ):
            assert source != target
            assert "weight" in data
            assert data["weight"] >= 1

