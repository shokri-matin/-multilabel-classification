# tests/test_processor_real_data.py

import networkx as nx

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


SAMPLE_SIZE = 5
WINDOW_SIZE = 3
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


def validate_subgraph_result(
    result,
    original_graph: nx.Graph,
    expected_mode: str,
):
    assert result.mode == expected_mode

    assert result.strategy in {
        "closeness",
        "degree",
        "betweenness",
        "pagerank",
        "clustering",
        "closeness*degree*clustering",
    }

    assert len(result.subgraphs) > 0

    for subgraph in result.subgraphs:

        # Central node must exist in original graph.
        assert subgraph.central_node in original_graph

        # Every extracted node must exist in original graph.
        assert all(
            node in original_graph
            for node in subgraph.nodes
        )

        # Extracted graph must contain exactly the selected nodes.
        assert set(subgraph.graph.nodes) == set(
            subgraph.nodes
        )

        # The extracted graph must be a valid NetworkX graph.
        assert isinstance(
            subgraph.graph,
            nx.Graph,
        )

        # No self loops.
        assert nx.number_of_selfloops(
            subgraph.graph
        ) == 0

        # Central node must always be included.
        assert subgraph.central_node in subgraph.nodes

        # The selected best neighbor must exist.
        assert subgraph.best_neighbor in original_graph

        # Best neighbor must be adjacent to central node,
        # except for the isolated-node case.
        if subgraph.central_node != subgraph.best_neighbor:
            assert original_graph.has_edge(
                subgraph.central_node,
                subgraph.best_neighbor,
            )

        # Maximum subgraph size.
        assert len(subgraph.nodes) <= MAX_NODES

        # No empty subgraphs.
        assert len(subgraph.nodes) >= 1


def run_real_data_processor_test(
    loader,
    pipeline,
    split: str,
    top_w: int,
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

    for document in processed_documents:

        word_graph = graph_builder.build(document)

        graph = word_graph.graph

        # Skip completely empty documents.
        if graph.number_of_nodes() == 0:
            continue

        centrality_results = centrality_calculator.rank(
            graph,
            top_k=min(
                top_w,
                graph.number_of_nodes(),
            ),
        )

        for strategy, strategy_result in centrality_results.items():

            # Make sure Top-W nodes are valid.
            assert len(
                strategy_result.top_k_nodes
            ) == min(
                top_w,
                graph.number_of_nodes(),
            )

            assert all(
                node in graph
                for node in strategy_result.top_k_nodes
            )

            for mode in ("no_branch", "branch"):

                processor = GraphProcessor(
                    max_nodes=MAX_NODES,
                    mode=mode,
                )

                result = processor.process_strategy(
                    graph=graph,
                    strategy_result=strategy_result,
                    doc_id= document.doc_id
                )

                validate_subgraph_result(
                    result=result,
                    original_graph=graph,
                    expected_mode=mode,
                )

                central_nodes = {
                    subgraph.central_node
                    for subgraph in result.subgraphs
                }

                # Every extracted central node must belong
                # to the Top-W set.
                assert central_nodes.issubset(
                    set(strategy_result.top_k_nodes)
                )

                if mode == "no_branch":
                    # Exactly one subgraph per Top-W central node.
                    assert len(
                        result.subgraphs
                    ) == len(
                        strategy_result.top_k_nodes
                    )

                else:
                    # Branch mode:
                    # at least one subgraph per central node.
                    assert len(
                        result.subgraphs
                    ) >= len(
                        strategy_result.top_k_nodes
                    )


def test_aapd_real_data_processor():
    loader = AAPDLoader(
        "data/raw/aapd"
    )

    pipeline = build_aapd_pipeline()

    run_real_data_processor_test(
        loader=loader,
        pipeline=pipeline,
        split="train",
        top_w=AAPD_TOP_W,
    )


def test_rcv1_real_data_processor():
    loader = RCV1Loader(
        "data/raw/rcv1"
    )

    pipeline = build_rcv1_pipeline()

    run_real_data_processor_test(
        loader=loader,
        pipeline=pipeline,
        split="train",
        top_w=RCV1_TOP_W,
    )