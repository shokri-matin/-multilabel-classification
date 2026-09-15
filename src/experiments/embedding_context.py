from dataclasses import dataclass
from typing import Iterator, List

from src.embedding.base import BaseEmbedder, EmbeddingOutput
from src.experiments.embedding_cache import EmbeddingArtifactCache
from src.experiments.sequence_context import (
    DatasetSequenceContext,
    DocumentSequenceContext,
)


@dataclass(frozen=True)
class DocumentEmbeddingContext:
    """
    Embedding artifacts for one document and one experiment configuration.
    """

    doc_id: str
    strategy: str
    mode: str
    embedding: str
    outputs: List[EmbeddingOutput]


@dataclass
class SplitEmbeddingContext:
    """
    Lazy embedding preparation context for one corpus split.

    Embeddings are generated document-by-document and cached.
    """

    dataset: str
    split: str
    sequence_context: object
    embedding_cache: EmbeddingArtifactCache

    def iter_documents(
        self,
        strategy: str,
        mode: str,
        embedding: str,
        embedder: BaseEmbedder,
    ) -> Iterator[DocumentEmbeddingContext]:

        self._validate_embedding(embedding)

        for sequence_document in self.sequence_context.iter_documents(
            strategy=strategy,
            mode=mode,
        ):
            yield self._get_or_build(
                sequence_document=sequence_document,
                embedding=embedding,
                embedder=embedder,
            )

    def get_document(
        self,
        doc_id: str,
        strategy: str,
        mode: str,
        embedding: str,
        embedder: BaseEmbedder,
    ) -> DocumentEmbeddingContext:

        self._validate_embedding(embedding)

        sequence_document = self.sequence_context.get_document(
            doc_id=doc_id,
            strategy=strategy,
            mode=mode,
        )

        return self._get_or_build(
            sequence_document=sequence_document,
            embedding=embedding,
            embedder=embedder,
        )

    def _get_or_build(
        self,
        sequence_document: DocumentSequenceContext,
        embedding: str,
        embedder: BaseEmbedder,
    ) -> DocumentEmbeddingContext:

        doc_id = sequence_document.doc_id
        strategy = sequence_document.strategy
        mode = sequence_document.mode

        self._validate_embedding(embedding)

        if self.embedding_cache.exists(
            dataset=self.dataset,
            doc_id=doc_id,
            centrality=strategy,
            mode=mode,
            embedding=embedding,
        ):
            outputs = self.embedding_cache.load(
                dataset=self.dataset,
                doc_id=doc_id,
                centrality=strategy,
                mode=mode,
                embedding=embedding,
            )

            return DocumentEmbeddingContext(
                doc_id=doc_id,
                strategy=strategy,
                mode=mode,
                embedding=embedding,
                outputs=outputs,
            )

        outputs = embedder.embed_many(
            sequence_document.sequences
        )

        self._validate_outputs(
            outputs=outputs,
            doc_id=doc_id,
            strategy=strategy,
            mode=mode,
            embedding=embedding,
            embedder=embedder,
        )

        self.embedding_cache.save(
            dataset=self.dataset,
            doc_id=doc_id,
            centrality=strategy,
            mode=mode,
            embedding=embedding,
            outputs=outputs,
        )

        return DocumentEmbeddingContext(
            doc_id=doc_id,
            strategy=strategy,
            mode=mode,
            embedding=embedding,
            outputs=outputs,
        )

    @staticmethod
    def _validate_embedding(
        embedding: str,
    ) -> None:

        if embedding not in {
            "bert",
            "fasttext",
        }:
            raise ValueError(
                f"Unsupported embedding: {embedding}"
            )

    @staticmethod
    def _validate_outputs(
        outputs: List[EmbeddingOutput],
        doc_id: str,
        strategy: str,
        mode: str,
        embedding: str,
        embedder: BaseEmbedder,
    ) -> None:

        if not isinstance(outputs, list):
            raise TypeError(
                "embedder.embed_many() must return a list"
            )

        for output in outputs:

            if not isinstance(
                output,
                EmbeddingOutput,
            ):
                raise TypeError(
                    "All embedding outputs must be "
                    "EmbeddingOutput instances"
                )

            if output.doc_id != doc_id:
                raise ValueError(
                    "Embedding output doc_id does not "
                    "match the requested document"
                )

            if output.strategy != strategy:
                raise ValueError(
                    "Embedding output strategy does not "
                    "match the requested strategy"
                )

            if output.mode != mode:
                raise ValueError(
                    "Embedding output mode does not "
                    "match the requested mode"
                )

            if output.embedding_dim != embedder.embedding_dim:
                raise ValueError(
                    "Embedding output dimension does not "
                    "match the embedder dimension"
                )


@dataclass
class DatasetEmbeddingContext:
    """
    Lazy embedding preparation context for one dataset.
    """

    dataset: str
    train: SplitEmbeddingContext
    validation: SplitEmbeddingContext
    test: SplitEmbeddingContext

    def split(
        self,
        split_name: str,
    ) -> SplitEmbeddingContext:

        if split_name == "train":
            return self.train

        if split_name == "validation":
            return self.validation

        if split_name == "test":
            return self.test

        raise ValueError(
            f"Unsupported split: {split_name}"
        )


class DatasetEmbeddingContextBuilder:
    """
    Build reusable embedding contexts on top of
    sequence artifacts.
    """

    SUPPORTED_DATASETS = {
        "aapd",
        "rcv1",
    }

    SUPPORTED_EMBEDDINGS = {
        "bert",
        "fasttext",
    }

    def __init__(
        self,
        sequence_context: DatasetSequenceContext,
        embedding_cache: EmbeddingArtifactCache,
    ):
        self.sequence_context = sequence_context
        self.embedding_cache = embedding_cache

    def build(
        self,
        dataset: str,
    ) -> DatasetEmbeddingContext:

        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset}"
            )

        if self.sequence_context.dataset != dataset:
            raise ValueError(
                "sequence_context dataset does not "
                "match dataset"
            )

        return DatasetEmbeddingContext(
            dataset=dataset,
            train=self._build_split("train"),
            validation=self._build_split("validation"),
            test=self._build_split("test"),
        )

    def _build_split(
        self,
        split: str,
    ) -> SplitEmbeddingContext:

        return SplitEmbeddingContext(
            dataset=self.sequence_context.dataset,
            split=split,
            sequence_context=self.sequence_context.split(
                split
            ),
            embedding_cache=self.embedding_cache,
        )