import torch

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

from src.corpus.contracts import Corpus
from src.corpus.builder import CorpusBuilder

from src.graph.word_graph import WordGraphBuilder
from src.graph.centrality import CentralityCalculator
from src.graph.processor import GraphProcessor

from src.sequence.word_sequence import (
    SequenceConfig,
    WordSequenceBuilder,
)

from src.embedding.fasttext import (
    FastTextEmbedder,
    FastTextConfig,
)

from src.classifier.dataset import (
    ClassifierDataset,
    LabelEncoder,
)

from src.classifier.dataloader import (
    ClassifierDataLoader,
)

SAMPLE_SIZE = 3
SEQUENCE_LENGTH = 25

AAPD_TOP_W = 200
RCV1_TOP_W = 100

FASTTEXT_MODEL_PATH = (
    "src/models/fasttext/crawl-300d-2M-subword.bin"
)


def build_aapd_corpus():
    loader = AAPDLoader("data/raw/aapd")

    pipeline = PreprocessingPipeline(
        tokenizer=AAPDTokenizer(),
        config=PreprocessingConfig(
            lowercase=True,
            remove_stopwords=True,
            stemming=True,
        ),
    )

    builder = CorpusBuilder(
        name="aapd",
        loader=loader,
        preprocessing_pipeline=pipeline,
    )

    raw_documents = loader.load_split("train")[:SAMPLE_SIZE]

    processed_documents = pipeline.process_many(
        raw_documents,
        show_progress=False,
    )

    return Corpus(
        name="aapd",
        split="train",
        documents=processed_documents,
    )


def build_rcv1_corpus():
    loader = RCV1Loader("data/raw/rcv1")

    pipeline = PreprocessingPipeline(
        tokenizer=WhitespaceTokenizer(),
        config=PreprocessingConfig(
            lowercase=False,
            remove_stopwords=False,
            stemming=False,
        ),
    )

    raw_documents = loader.load_split("train")[:SAMPLE_SIZE]

    processed_documents = pipeline.process_many(
        raw_documents,
        show_progress=False,
    )

    from src.corpus.contracts import Corpus

    return Corpus(
        name="rcv1",
        split="train",
        documents=processed_documents,
    )


def build_sequences(corpus, mode, top_w):
    graph_builder = WordGraphBuilder(
        window_size=3
    )

    centrality_calculator = CentralityCalculator()

    processor = GraphProcessor(
        max_nodes=30,
        mode=mode,
    )

    sequence_builder = WordSequenceBuilder(
        SequenceConfig(
            sequence_length=SEQUENCE_LENGTH
        )
    )

    all_sequences = []

    for document in corpus.documents:
        word_graph = graph_builder.build(document)

        strategy_results = centrality_calculator.rank(
            word_graph.graph,
            top_k=top_w,
        )

        processed = processor.process_all(
            graph=word_graph.graph,
            strategy_results=strategy_results,
            doc_id=document.doc_id,
        )

        for strategy_result in processed.values():
            for subgraph in strategy_result.subgraphs:
                sequence = sequence_builder.build(
                    subgraph=subgraph,
                    mode=mode,
                )

                assert sequence.doc_id == document.doc_id
                assert len(sequence.words) == SEQUENCE_LENGTH
                assert len(sequence.positions) == SEQUENCE_LENGTH

                all_sequences.append(sequence)

    return all_sequences


def build_document_labels(corpus):
    return {
        document.doc_id: document.labels
        for document in corpus.documents
    }


def validate_dataset(
    sequences,
    document_labels,
    embedding_outputs,
):
    assert len(sequences) == len(embedding_outputs)

    for sequence, embedding in zip(
        sequences,
        embedding_outputs,
    ):
        assert sequence.doc_id == embedding.doc_id
        assert sequence.strategy == embedding.strategy
        assert sequence.mode == embedding.mode
        assert embedding.embeddings.shape == (
            SEQUENCE_LENGTH,
            embedding.embedding_dim,
        )

    vocabulary = sorted(
        {
            label
            for labels in document_labels.values()
            for label in labels
        }
    )

    encoder = LabelEncoder(vocabulary)

    dataset = ClassifierDataset(
        embeddings=embedding_outputs,
        document_labels=document_labels,
        label_encoder=encoder,
    )

    dataloader = ClassifierDataLoader(
        dataset=dataset,
        batch_size=2,
        shuffle=False,
    )

    batch = next(iter(dataloader))

    assert batch.inputs_embeds.shape[0] <= 2
    assert batch.inputs_embeds.shape[1] == SEQUENCE_LENGTH
    assert batch.inputs_embeds.shape[2] == embedding_outputs[0].embedding_dim

    assert batch.attention_mask.shape == (
        batch.inputs_embeds.shape[0],
        SEQUENCE_LENGTH,
    )

    assert batch.labels.shape == (
        batch.inputs_embeds.shape[0],
        len(vocabulary),
    )

    assert len(batch.doc_ids) == batch.inputs_embeds.shape[0]

    for doc_id in batch.doc_ids:
        assert doc_id in document_labels

    return dataset, dataloader


def test_aapd_fasttext_real_pipeline():
    corpus = build_aapd_corpus()

    sequences = build_sequences(
        corpus=corpus,
        mode="no_branch",
        top_w=AAPD_TOP_W,
    )

    assert sequences

    embedder = FastTextEmbedder(
        FastTextConfig(
            model_path=FASTTEXT_MODEL_PATH,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dim=300,
        )
    )

    embedding_outputs = embedder.embed_many(
        sequences[:10]
    )

    document_labels = build_document_labels(
        corpus
    )

    dataset, dataloader = validate_dataset(
        sequences=sequences[:10],
        document_labels=document_labels,
        embedding_outputs=embedding_outputs,
    )

    assert dataset.input_dim == 300
    assert dataset.sequence_length == SEQUENCE_LENGTH
    assert dataset.num_labels == len(
        dataset.label_encoder.label_vocabulary
    )


def test_rcv1_fasttext_real_pipeline():
    corpus = build_rcv1_corpus()

    sequences = build_sequences(
        corpus=corpus,
        mode="no_branch",
        top_w=RCV1_TOP_W,
    )

    assert sequences

    embedder = FastTextEmbedder(
        FastTextConfig(
            model_path=FASTTEXT_MODEL_PATH,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dim=300,
        )
    )

    embedding_outputs = embedder.embed_many(
        sequences[:10]
    )

    document_labels = build_document_labels(
        corpus
    )

    dataset, dataloader = validate_dataset(
        sequences=sequences[:10],
        document_labels=document_labels,
        embedding_outputs=embedding_outputs,
    )

    assert dataset.input_dim == 300
    assert dataset.sequence_length == SEQUENCE_LENGTH


def test_aapd_branch_and_no_branch_have_valid_doc_ids():
    corpus = build_aapd_corpus()

    no_branch_sequences = build_sequences(
        corpus=corpus,
        mode="no_branch",
        top_w=AAPD_TOP_W,
    )

    branch_sequences = build_sequences(
        corpus=corpus,
        mode="branch",
        top_w=AAPD_TOP_W,
    )

    assert no_branch_sequences
    assert branch_sequences

    valid_doc_ids = {
        document.doc_id
        for document in corpus.documents
    }

    assert all(
        sequence.doc_id in valid_doc_ids
        for sequence in no_branch_sequences
    )

    assert all(
        sequence.doc_id in valid_doc_ids
        for sequence in branch_sequences
    )


def test_attention_mask_matches_words():
    corpus = build_aapd_corpus()

    sequences = build_sequences(
        corpus=corpus,
        mode="no_branch",
        top_w=AAPD_TOP_W,
    )

    embedder = FastTextEmbedder(
        FastTextConfig(
            model_path=FASTTEXT_MODEL_PATH,
            sequence_length=SEQUENCE_LENGTH,
            embedding_dim=300,
        )
    )

    embedding_outputs = embedder.embed_many(
        sequences[:5]
    )

    document_labels = build_document_labels(
        corpus
    )

    vocabulary = sorted(
        {
            label
            for labels in document_labels.values()
            for label in labels
        }
    )

    dataset = ClassifierDataset(
        embeddings=embedding_outputs,
        document_labels=document_labels,
        label_encoder=LabelEncoder(vocabulary),
    )

    for index in range(len(dataset)):
        sample = dataset[index]

        words = embedding_outputs[index].words

        expected_mask = torch.tensor(
            [
                0 if word == "<PAD>" else 1
                for word in words
            ],
            dtype=torch.long,
        )

        assert torch.equal(
            sample["attention_mask"],
            expected_mask,
        )