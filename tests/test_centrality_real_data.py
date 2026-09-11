
import math
from pathlib import Path

from src.data.aapd import AAPDLoader
from src.data.rcv1 import RCV1Loader

from src.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
)

from src.preprocessing.tokenizer import (
    AAPDTokenizer,
    WhitespaceTokenizer,
)

from src.graph.word_graph import WordGraphBuilder
from src.graph.centrality import (
    CENTRALITY_STRATEGIES,
    CentralityCalculator,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

SAMPLE_SIZE = 5


def build_aapd_pipeline():
    config = PreprocessingConfig(
        lowercase=True,
        remove_stopwords=True,
        stemming=True,
    )

    return PreprocessingPipeline(
        tokenizer=AAPDTokenizer(),
        config=config,
    )


def build_rcv1_pipeline():
    config = PreprocessingConfig(
        lowercase=False,
        remove_stopwords=False,
        stemming=False,
    )

    return PreprocessingPipeline(
        tokenizer=WhitespaceTokenizer(),
        config=config,
    )


def validate_centrality_results(
    graph,
    results,
    top_k,
):
    # ---------------------------------------------------------
    # 1. All strategies must be present
    # ---------------------------------------------------------

    assert set(results.keys()) == set(
        CENTRALITY_STRATEGIES
    )

    # ---------------------------------------------------------
    # 2. Validate every strategy
    # ---------------------------------------------------------

    for strategy in CENTRALITY_STRATEGIES:

        result = results[strategy]

        # Every graph node must have a score.
        assert set(result.scores.keys()) == set(
            graph.nodes
        )

        # Ranking must contain every graph node exactly once.
        assert len(result.ranked_nodes) == (
            graph.number_of_nodes()
        )

        assert set(result.ranked_nodes) == set(
            graph.nodes
        )

        # -----------------------------------------------------
        # Validate scores
        # -----------------------------------------------------

        for score in result.scores.values():

            assert isinstance(
                score,
                (int, float),
            )

            assert math.isfinite(score)

            assert score >= 0.0

        # -----------------------------------------------------
        # Validate ranking
        # -----------------------------------------------------

        for first, second in zip(
            result.ranked_nodes,
            result.ranked_nodes[1:],
        ):
            assert (
                result.scores[first]
                >= result.scores[second]
            )

        # -----------------------------------------------------
        # Validate Top-W
        # -----------------------------------------------------

        expected_top_k = min(
            top_k,
            graph.number_of_nodes(),
        )

        assert len(
            result.top_k_nodes
        ) == expected_top_k

        # Every selected node must exist in graph.
        assert all(
            node in graph
            for node in result.top_k_nodes
        )

        # Top-W must be the first W ranked nodes.
        assert result.top_k_nodes == (
            result.ranked_nodes[:expected_top_k]
        )


def test_aapd_real_data_centrality():

    # ---------------------------------------------------------
    # 1. Load AAPD
    # ---------------------------------------------------------

    loader = AAPDLoader(
        RAW_DATA_DIR / "aapd"
    )

    raw_documents = loader.load_split(
        "train"
    )

    assert len(raw_documents) > 0

    sample = raw_documents[:SAMPLE_SIZE]

    # ---------------------------------------------------------
    # 2. Preprocessing
    # ---------------------------------------------------------

    pipeline = build_aapd_pipeline()

    processed_documents = pipeline.process_many(
        sample,
        show_progress=False,
    )

    assert len(processed_documents) == (
        len(sample)
    )

    for raw, processed in zip(
        sample,
        processed_documents,
    ):
        assert processed.doc_id == raw.doc_id
        assert processed.labels == raw.labels
        assert processed.token_count > 0

    # ---------------------------------------------------------
    # 3. Word Graph
    # ---------------------------------------------------------

    graph_builder = WordGraphBuilder(
        window_size=3,
    )

    calculator = CentralityCalculator()

    tested_documents = 0

    for document in processed_documents:

        word_graph = graph_builder.build(
            document
        )

        graph = word_graph.graph

        assert graph.number_of_nodes() > 0

        # A graph with only one node cannot provide
        # a meaningful centrality ranking.
        if graph.number_of_nodes() < 2:
            continue

        # -----------------------------------------------------
        # AAPD → W = 200
        # -----------------------------------------------------

        results = calculator.rank(
            graph,
            top_k=200,
        )

        validate_centrality_results(
            graph=graph,
            results=results,
            top_k=200,
        )

        tested_documents += 1

    assert tested_documents > 0


def test_rcv1_real_data_centrality():

    # ---------------------------------------------------------
    # 1. Load RCV1
    # ---------------------------------------------------------

    loader = RCV1Loader(
        RAW_DATA_DIR / "rcv1"
    )

    raw_documents = loader.load_split(
        "train"
    )

    assert len(raw_documents) > 0

    sample = raw_documents[:SAMPLE_SIZE]

    # ---------------------------------------------------------
    # 2. Preprocessing
    # ---------------------------------------------------------

    pipeline = build_rcv1_pipeline()

    processed_documents = pipeline.process_many(
        sample,
        show_progress=False,
    )

    assert len(processed_documents) == (
        len(sample)
    )

    for raw, processed in zip(
        sample,
        processed_documents,
    ):
        assert processed.doc_id == raw.doc_id
        assert processed.labels == raw.labels
        assert processed.token_count > 0

    # ---------------------------------------------------------
    # 3. Word Graph
    # ---------------------------------------------------------

    graph_builder = WordGraphBuilder(
        window_size=3,
    )

    calculator = CentralityCalculator()

    tested_documents = 0

    for document in processed_documents:

        word_graph = graph_builder.build(
            document
        )

        graph = word_graph.graph

        assert graph.number_of_nodes() > 0

        if graph.number_of_nodes() < 2:
            continue

        # -----------------------------------------------------
        # RCV1 → W = 100
        # -----------------------------------------------------

        results = calculator.rank(
            graph,
            top_k=100,
        )

        validate_centrality_results(
            graph=graph,
            results=results,
            top_k=100,
        )

        tested_documents += 1

    assert tested_documents > 0
