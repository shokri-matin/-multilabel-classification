from dataclasses import dataclass

import fasttext
import numpy as np

from src.embedding.base import BaseEmbedder, EmbeddingOutput
from src.embedding.config import EmbeddingConfig
from src.sequence.word_sequence import WordSequence


@dataclass(frozen=True)
class FastTextConfig(EmbeddingConfig):
    """
    FastText-specific configuration.
    """

    model_path: str = ""

    embedding_dim: int = 300
    padding_token: str = "<PAD>"

    def __post_init__(self):
        super().__post_init__()

        if not self.model_path:
            raise ValueError(
                "model_path must not be empty"
            )

        if self.embedding_dim <= 0:
            raise ValueError(
                "embedding_dim must be positive"
            )

        if not self.padding_token:
            raise ValueError(
                "padding_token must not be empty"
            )


class FastTextEmbedder(BaseEmbedder):
    """
    FastText-based word sequence embedder.

    One original word is mapped to one FastText vector.

    The native FastText .bin model is used so that
    out-of-vocabulary words can receive vectors generated
    from their subword information.

    <PAD> is represented by an all-zero vector.
    """

    def __init__(
        self,
        config: FastTextConfig,
    ):
        self.config = config

        self.model = self._load_model(
            config.model_path
        )

        actual_dim = self.model.get_dimension()

        if actual_dim != config.embedding_dim:
            raise ValueError(
                f"FastText model dimension is {actual_dim}, "
                f"but config specifies {config.embedding_dim}"
            )

        self._embedding_dim = actual_dim

    @staticmethod
    def _load_model(
        model_path: str,
    ):
        """
        Load a native FastText .bin model.
        """

        return fasttext.load_model(
            model_path
        )

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _get_word_vector(
        self,
        word: str,
    ) -> np.ndarray:
        """
        Get a FastText vector for one word.

        FastText can generate vectors for out-of-vocabulary
        words using subword information.
        """

        vector = self.model.get_word_vector(
            word
        )

        return np.asarray(
            vector,
            dtype=np.float32,
        )

    def _embed_words(
        self,
        words: list[str],
    ) -> np.ndarray:
        """
        Convert words into a fixed-size embedding matrix.
        """

        result = np.zeros(
            (
                len(words),
                self.embedding_dim,
            ),
            dtype=np.float32,
        )

        for index, word in enumerate(words):

            if word == self.config.padding_token:
                continue

            result[index] = self._get_word_vector(
                word
            )

        if self.config.normalize:

            norms = np.linalg.norm(
                result,
                axis=1,
                keepdims=True,
            )

            nonzero = norms.squeeze(-1) > 0

            result[nonzero] /= norms[nonzero]

        return result

    def embed(
        self,
        sequence: WordSequence,
    ) -> EmbeddingOutput:

        embeddings = self._embed_words(
            sequence.words
        )

        return EmbeddingOutput(
            doc_id= sequence.doc_id,
            strategy=sequence.strategy,
            mode=sequence.mode,
            central_node=sequence.central_node,
            best_neighbor=sequence.best_neighbor,
            words=sequence.words.copy(),
            embeddings=embeddings,
            embedding_dim=self.embedding_dim,
            sequence_length=sequence.length,
        )