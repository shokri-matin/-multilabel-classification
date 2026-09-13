import numpy as np

from src.graph.processor import SubgraphResult
from src.sequence.word_sequence import (
    SequenceConfig,
    WordSequenceBuilder,
)
from src.embedding.base import EmbeddingOutput


def test_doc_id_propagates_from_subgraph_to_sequence():
    graph = __import__("networkx").Graph()

    graph.add_node(
        "machine",
        positions=[0],
    )
    graph.add_node(
        "learning",
        positions=[1],
    )

    graph.add_edge(
        "machine",
        "learning",
        weight=1,
    )

    subgraph = SubgraphResult(
        doc_id="doc_001",
        strategy="degree",
        central_node="machine",
        central_score=1.0,
        best_neighbor="learning",
        nodes=["machine", "learning"],
        graph=graph,
    )

    builder = WordSequenceBuilder(
        SequenceConfig(
            sequence_length=5,
        )
    )

    sequence = builder.build(
        subgraph=subgraph,
        mode="no_branch",
    )

    assert sequence.doc_id == "doc_001"


def test_doc_id_propagates_from_sequence_to_embedding_output():
    sequence_doc_id = "doc_002"

    sequence = __import__(
        "src.sequence.word_sequence",
        fromlist=["WordSequence"],
    ).WordSequence(
        doc_id=sequence_doc_id,
        strategy="pagerank",
        mode="branch",
        central_node="graph",
        best_neighbor="learning",
        words=[
            "graph",
            "learning",
            "<PAD>",
        ],
        positions=[
            0,
            1,
            -1,
        ],
        original_length=2,
    )

    embeddings = np.zeros(
        (3, 300),
        dtype=np.float32,
    )

    output = EmbeddingOutput(
        doc_id=sequence.doc_id,
        strategy=sequence.strategy,
        mode=sequence.mode,
        central_node=sequence.central_node,
        best_neighbor=sequence.best_neighbor,
        words=sequence.words.copy(),
        embeddings=embeddings,
        embedding_dim=300,
        sequence_length=sequence.length,
    )

    assert output.doc_id == "doc_002"
    assert output.doc_id == sequence.doc_id