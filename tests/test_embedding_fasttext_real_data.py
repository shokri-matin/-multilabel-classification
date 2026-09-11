from pathlib import Path

import numpy as np

from src.data.aapd import AAPDLoader
from src.data.rcv1 import RCV1Loader
from src.embedding.fasttext import (
    FastTextConfig,
    FastTextEmbedder,
)
from src.graph.centrality import CentralityCalculator
from src.graph.processor import GraphProcessor
from src.graph.word_graph import WordGraphBuilder
from src.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
)
from src.preprocessing.tokenizer import (
    AAPDTokenizer,
    WhitespaceTokenizer,
)
from src.sequence.word_sequence import (
    SequenceConfig,
    WordSequenceBuilder,
)


MODEL_PATH = Path(
    "src/models/fasttext/crawl-300d-2M-subword.bin"
)

SAMPLE_SIZE = 5
WINDOW_SIZE = 3
MAX_NODES = 20
SEQUENCE_LENGTH = 25

AAPD_TOP_W = 200
RCV1_TOP_W = 100


def build_sequences(
    dataset_name: str,
    model_path: str,
):
    if dataset_name == "aapd":
        loader = AAPDLoader("data/raw/aapd")

        preprocessing_config = PreprocessingConfig(
            lowercase=True,
            remove_stopwords=True,
            stemming=True,
        )

        tokenizer = AAPDTokenizer()
        top_w = AAPD_TOP_W

    elif dataset_name == "rcv1":
        loader = RCV1Loader("data/raw/rcv1")

        preprocessing_config = PreprocessingConfig(
            lowercase=False,
            remove_stopwords=False,
            stemming=False,
        )

        tokenizer = WhitespaceTokenizer()
        top_w = RCV1_TOP_W

    else:
        raise ValueError(
            f"Unknown dataset: {dataset_name}"
        )

    pipeline = PreprocessingPipeline(
        tokenizer=tokenizer,
        config=preprocessing_config,
    )

    raw_documents = loader.load_split("train")[
        :SAMPLE_SIZE
    ]

    documents = pipeline.process_many(
        raw_documents,
        show_progress=True,
    )

    graph_builder = WordGraphBuilder(
        window_size=WINDOW_SIZE
    )

    centrality_calculator = CentralityCalculator()

    sequence_builder = WordSequenceBuilder(
        SequenceConfig(
            sequence_length=SEQUENCE_LENGTH,
        )
    )

    fasttext_embedder = FastTextEmbedder(
        FastTextConfig(
            model_path=model_path,
            embedding_dim=300,
            sequence_length=SEQUENCE_LENGTH,
            normalize=False,
        )
    )

    all_results = []

    for document in documents:

        word_graph = graph_builder.build(
            document
        )

        strategy_results = centrality_calculator.rank(
            word_graph.graph,
            top_k=top_w,
        )

        for strategy, strategy_result in strategy_results.items():

            for mode in ("no_branch", "branch"):

                processor = GraphProcessor(
                    max_nodes=MAX_NODES,
                    mode=mode,
                )

                subgraph_result = processor.process_strategy(
                    graph=word_graph.graph,
                    strategy_result=strategy_result,
                )

                for subgraph in subgraph_result.subgraphs:

                    sequence = sequence_builder.build(
                        subgraph=subgraph,
                        mode=mode,
                    )

                    embedding = fasttext_embedder.embed(
                        sequence
                    )

                    all_results.append(
                        embedding
                    )

    return all_results


def test_fasttext_real_aapd():
    results = build_sequences(
        dataset_name="aapd",
        model_path=str(MODEL_PATH),
    )

    assert results

    for result in results:

        assert result.embedding_dim == 300
        assert result.sequence_length == SEQUENCE_LENGTH

        assert result.embeddings.shape == (
            SEQUENCE_LENGTH,
            300,
        )

        assert result.embeddings.dtype == np.float32

        assert len(result.words) == SEQUENCE_LENGTH

        padding_indices = [
            index
            for index, word in enumerate(result.words)
            if word == "<PAD>"
        ]

        for index in padding_indices:
            assert np.allclose(
                result.embeddings[index],
                0.0,
            )


def test_fasttext_real_rcv1():
    results = build_sequences(
        dataset_name="rcv1",
        model_path=str(MODEL_PATH),
    )

    assert results

    for result in results:

        assert result.embedding_dim == 300
        assert result.sequence_length == SEQUENCE_LENGTH

        assert result.embeddings.shape == (
            SEQUENCE_LENGTH,
            300,
        )

        assert result.embeddings.dtype == np.float32

        assert len(result.words) == SEQUENCE_LENGTH

        padding_indices = [
            index
            for index, word in enumerate(result.words)
            if word == "<PAD>"
        ]

        for index in padding_indices:
            assert np.allclose(
                result.embeddings[index],
                0.0,
            )