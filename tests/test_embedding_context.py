import numpy as np
import pytest

from src.embedding.base import BaseEmbedder, EmbeddingOutput
from src.experiments.embedding_cache import EmbeddingArtifactCache
from src.experiments.embedding_context import (
    DatasetEmbeddingContextBuilder,
    DocumentEmbeddingContext,
    SplitEmbeddingContext,
)


class FakeEmbedder(BaseEmbedder):

    def __init__(self, dimension: int = 4):
        self._dimension = dimension
        self.calls = 0

    @property
    def embedding_dim(self) -> int:
        return self._dimension

    def embed(self, sequence):

        self.calls += 1

        return EmbeddingOutput(
            doc_id=sequence.doc_id,
            strategy=sequence.strategy,
            mode=sequence.mode,
            central_node=sequence.central_node,
            best_neighbor=sequence.best_neighbor,
            words=sequence.words.copy(),
            embeddings=np.ones(
                (
                    sequence.length,
                    self.embedding_dim,
                ),
                dtype=np.float32,
            ),
            embedding_dim=self.embedding_dim,
            sequence_length=sequence.length,
        )


class FakeSequenceContext:

    def __init__(self, document):
        self.dataset = "aapd"
        self.document = document

    def iter_documents(self, strategy, mode):
        yield self.document

    def get_document(self, doc_id, strategy, mode):

        if doc_id != self.document.doc_id:
            raise KeyError(doc_id)

        return self.document

    def split(self, split):
        return self


def make_sequence_context():

    from src.experiments.sequence_context import (
        DocumentSequenceContext,
    )

    from src.sequence.word_sequence import (
        WordSequence,
    )

    sequence = WordSequence(
        doc_id="doc_1",
        strategy="closeness",
        mode="no_branch",
        central_node="traffic",
        best_neighbor="light",
        words=["traffic", "light"],
        positions=[0, 1],
        original_length=2,
    )

    return DocumentSequenceContext(
        doc_id="doc_1",
        strategy="closeness",
        mode="no_branch",
        sequences=[sequence],
    )


def test_embedding_context_builds_and_caches(tmp_path):

    sequence_context = make_sequence_context()

    embedding_cache = EmbeddingArtifactCache(
        tmp_path
    )

    split_context = SplitEmbeddingContext(
        dataset="aapd",
        split="train",
        sequence_context=FakeSequenceContext(
            sequence_context
        ),
        embedding_cache=embedding_cache,
    )

    embedder = FakeEmbedder()

    result = split_context.get_document(
        doc_id="doc_1",
        strategy="closeness",
        mode="no_branch",
        embedding="bert",
        embedder=embedder,
    )

    assert isinstance(
        result,
        DocumentEmbeddingContext,
    )

    assert len(result.outputs) == 1
    assert result.outputs[0].embedding_dim == 4
    assert embedder.calls == 1

    cached = split_context.get_document(
        doc_id="doc_1",
        strategy="closeness",
        mode="no_branch",
        embedding="bert",
        embedder=embedder,
    )

    assert len(cached.outputs) == 1

    # The second call must use the cache.
    assert embedder.calls == 1


def test_embedding_context_supports_fasttext(tmp_path):

    sequence_context = make_sequence_context()

    embedding_cache = EmbeddingArtifactCache(
        tmp_path
    )

    split_context = SplitEmbeddingContext(
        dataset="aapd",
        split="train",
        sequence_context=FakeSequenceContext(
            sequence_context
        ),
        embedding_cache=embedding_cache,
    )

    embedder = FakeEmbedder(
        dimension=300
    )

    result = split_context.get_document(
        doc_id="doc_1",
        strategy="closeness",
        mode="no_branch",
        embedding="fasttext",
        embedder=embedder,
    )

    assert result.embedding == "fasttext"
    assert result.outputs[0].embedding_dim == 300


def test_embedding_cache_is_independent_of_classifier(
    tmp_path,
):

    sequence_context = make_sequence_context()

    embedding_cache = EmbeddingArtifactCache(
        tmp_path
    )

    split_context = SplitEmbeddingContext(
        dataset="aapd",
        split="train",
        sequence_context=FakeSequenceContext(
            sequence_context
        ),
        embedding_cache=embedding_cache,
    )

    embedder = FakeEmbedder()

    result = split_context.get_document(
        doc_id="doc_1",
        strategy="closeness",
        mode="no_branch",
        embedding="bert",
        embedder=embedder,
    )

    assert result.outputs

    # The cache key contains no classifier component.
    assert embedding_cache.exists(
        dataset="aapd",
        doc_id="doc_1",
        centrality="closeness",
        mode="no_branch",
        embedding="bert",
    )


def test_invalid_embedding_is_rejected(tmp_path):

    sequence_context = make_sequence_context()

    embedding_cache = EmbeddingArtifactCache(
        tmp_path
    )

    split_context = SplitEmbeddingContext(
        dataset="aapd",
        split="train",
        sequence_context=FakeSequenceContext(
            sequence_context
        ),
        embedding_cache=embedding_cache,
    )

    with pytest.raises(ValueError):

        split_context.get_document(
            doc_id="doc_1",
            strategy="closeness",
            mode="no_branch",
            embedding="word2vec",
            embedder=FakeEmbedder(),
        )


def test_embedder_dimension_mismatch_is_rejected(
    tmp_path,
):

    sequence_context = make_sequence_context()

    embedding_cache = EmbeddingArtifactCache(
        tmp_path
    )

    split_context = SplitEmbeddingContext(
        dataset="aapd",
        split="train",
        sequence_context=FakeSequenceContext(
            sequence_context
        ),
        embedding_cache=embedding_cache,
    )

    class BadEmbedder(FakeEmbedder):

        @property
        def embedding_dim(self):
            return 8

        def embed(self, sequence):

            return EmbeddingOutput(
                doc_id=sequence.doc_id,
                strategy=sequence.strategy,
                mode=sequence.mode,
                central_node=sequence.central_node,
                best_neighbor=sequence.best_neighbor,
                words=sequence.words.copy(),
                embeddings=np.ones(
                    (sequence.length, 4),
                    dtype=np.float32,
                ),
                embedding_dim=4,
                sequence_length=sequence.length,
            )

    with pytest.raises(ValueError):

        split_context.get_document(
            doc_id="doc_1",
            strategy="closeness",
            mode="no_branch",
            embedding="bert",
            embedder=BadEmbedder(),
        )


def test_dataset_builder_rejects_wrong_dataset(tmp_path):

    class FakeDatasetSequenceContext:

        dataset = "rcv1"

    builder = DatasetEmbeddingContextBuilder(
        sequence_context=FakeDatasetSequenceContext(),
        embedding_cache=EmbeddingArtifactCache(
            tmp_path
        ),
    )

    with pytest.raises(ValueError):
        builder.build("aapd")