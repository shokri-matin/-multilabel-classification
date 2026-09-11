# tests/test_word_sequence_real_data.py

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
from src.graph.centrality import CentralityCalculator
from src.graph.processor import GraphProcessor

from src.sequence.word_sequence import (
    SequenceConfig,
    WordSequenceBuilder,
)


SAMPLE_SIZE = 5
WINDOW_SIZE = 3

SEQUENCE_LENGTH = 25
MAX_NODES = 20

AAPD_TOP_W = 200
RCV1_TOP_W = 100


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


def validate_sequence(
    sequence,
    expected_mode,
):
    """
    Validate one real-data WordSequence.
    """

    assert sequence.mode == expected_mode

    assert sequence.strategy in {
        "closeness",
        "degree",
        "betweenness",
        "pagerank",
        "clustering",
        "closeness*degree*clustering",
    }

    # Fixed-length sequence.
    assert len(sequence.words) == SEQUENCE_LENGTH
    assert len(sequence.positions) == SEQUENCE_LENGTH

    # Metadata must exist.
    assert sequence.central_node
    assert sequence.best_neighbor

    # Original length must be valid.
    assert sequence.original_length >= 1

    # Number of real words cannot exceed original length.
    real_word_count = sum(
        word != "<PAD>"
        for word in sequence.words
    )

    assert real_word_count <= sequence.original_length

    # Padding positions must be -1.
    for word, position in zip(
        sequence.words,
        sequence.positions,
    ):
        if word == "<PAD>":
            assert position == -1
        else:
            assert position >= 0

    # If sequence was not truncated, all original
    # nodes should be represented.
    if sequence.original_length <= SEQUENCE_LENGTH:
        assert real_word_count == sequence.original_length

    # If sequence was padded, it must actually contain PAD.
    if sequence.original_length < SEQUENCE_LENGTH:
        assert "<PAD>" in sequence.words

    # If sequence exactly fills h, no PAD should exist.
    if sequence.original_length == SEQUENCE_LENGTH:
        assert "<PAD>" not in sequence.words

    # Positions of real words must be monotonically increasing.
    real_positions = [
        position
        for word, position in zip(
            sequence.words,
            sequence.positions,
        )
        if word != "<PAD>"
    ]

    assert real_positions == sorted(real_positions)


def run_real_data_sequence_test(
    loader,
    pipeline,
    split,
    top_w,
):
    raw_documents = loader.load_split(split)

    sample_documents = raw_documents[:SAMPLE_SIZE]

    processed_documents = pipeline.process_many(
        sample_documents,
        show_progress=False,
    )

    graph_builder = WordGraphBuilder(
        window_size=WINDOW_SIZE,
    )

    centrality_calculator = CentralityCalculator()

    sequence_builder = WordSequenceBuilder(
        SequenceConfig(
            sequence_length=SEQUENCE_LENGTH,
            padding_token="<PAD>",
        )
    )

    for document in processed_documents:

        word_graph = graph_builder.build(document)

        graph = word_graph.graph

        # Ignore documents that produce an empty graph.
        if graph.number_of_nodes() == 0:
            continue

        top_k = min(
            top_w,
            graph.number_of_nodes(),
        )

        centrality_results = centrality_calculator.rank(
            graph,
            top_k=top_k,
        )

        for strategy_result in centrality_results.values():

            for mode in (
                "no_branch",
                "branch",
            ):

                processor = GraphProcessor(
                    max_nodes=MAX_NODES,
                    mode=mode,
                )

                subgraph_result = processor.process_strategy(
                    graph=graph,
                    strategy_result=strategy_result,
                )

                # No-Branch must create exactly one
                # subgraph per Top-W central node.
                if mode == "no_branch":
                    assert len(
                        subgraph_result.subgraphs
                    ) == len(
                        strategy_result.top_k_nodes
                    )

                # Branch must produce at least one
                # subgraph for every central node.
                else:
                    assert len(
                        subgraph_result.subgraphs
                    ) >= len(
                        strategy_result.top_k_nodes
                    )

                for subgraph in subgraph_result.subgraphs:

                    sequence = sequence_builder.build(
                        subgraph=subgraph,
                        mode=mode,
                    )

                    validate_sequence(
                        sequence=sequence,
                        expected_mode=mode,
                    )

                    # The sequence central node must
                    # correspond to a Top-W node.
                    assert (
                        sequence.central_node
                        in strategy_result.top_k_nodes
                    )


def test_aapd_real_data_word_sequences():
    loader = AAPDLoader(
        "data/raw/aapd"
    )

    pipeline = build_aapd_pipeline()

    run_real_data_sequence_test(
        loader=loader,
        pipeline=pipeline,
        split="train",
        top_w=AAPD_TOP_W,
    )


def test_rcv1_real_data_word_sequences():
    loader = RCV1Loader(
        "data/raw/rcv1"
    )

    pipeline = build_rcv1_pipeline()

    run_real_data_sequence_test(
        loader=loader,
        pipeline=pipeline,
        split="train",
        top_w=RCV1_TOP_W,
    )