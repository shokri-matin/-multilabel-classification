from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np

from src.sequence.word_sequence import WordSequence


@dataclass
class EmbeddingOutput:
    """
    Result of embedding one word sequence.
    """
    doc_id: str
    strategy: str
    mode: str

    central_node: str
    best_neighbor: str

    words: List[str]

    embeddings: np.ndarray

    embedding_dim: int

    sequence_length: int


class BaseEmbedder(ABC):
    """
    Base interface for all word-sequence embedders.
    """

    @property
    @abstractmethod
    def embedding_dim(self) -> int:
        """
        Return the embedding dimension.
        """
        ...

    @abstractmethod
    def embed(
        self,
        sequence: WordSequence,
    ) -> EmbeddingOutput:
        """
        Embed one WordSequence.
        """
        ...

    def embed_many(
        self,
        sequences: List[WordSequence],
    ) -> List[EmbeddingOutput]:
        """
        Embed multiple WordSequence objects.
        """

        return [
            self.embed(sequence)
            for sequence in sequences
        ]