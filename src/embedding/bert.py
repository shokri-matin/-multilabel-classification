from dataclasses import dataclass
from typing import List

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from src.embedding.base import BaseEmbedder, EmbeddingOutput
from src.embedding.config import EmbeddingConfig
from src.sequence.word_sequence import WordSequence


@dataclass(frozen=True)
class BERTConfig(EmbeddingConfig):
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    layer: int = -1

    def __post_init__(self):
        super().__post_init__()

        if not self.model_name:
            raise ValueError("model_name must not be empty")

        if self.max_length <= 0:
            raise ValueError("max_length must be positive")


class BERTEmbedder(BaseEmbedder):
    """
    BERT-based word sequence embedder.

    Each original word is mapped to one vector by averaging
    the hidden states of its corresponding BERT subwords.
    """

    def __init__(
        self,
        config: BERTConfig | None = None,
        device: str | None = None,
    ):
        self.config = config or BERTConfig()

        self.device = device or (
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        torch.manual_seed(self.config.seed)

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.model_name,
            use_fast=True,
        )

        self.model = AutoModel.from_pretrained(
            self.config.model_name,
        )

        self.model.to(self.device)
        self.model.eval()

        hidden_size = self.model.config.hidden_size

        if self.config.layer >= 0:
            num_layers = self.model.config.num_hidden_layers

            if self.config.layer >= num_layers:
                raise ValueError(
                    f"Invalid layer {self.config.layer}. "
                    f"Model has {num_layers} hidden layers."
                )

        self._embedding_dim = hidden_size

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    def _get_hidden_states(
        self,
        words: List[str],
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Tokenize words while preserving word boundaries.

        Returns:
            hidden_states:
                Shape: (bert_sequence_length, hidden_size)

            word_ids:
                Shape: (bert_sequence_length,)
        """

        encoding = self.tokenizer(
            words,
            is_split_into_words=True,
            add_special_tokens=True,
            truncation=True,
            max_length=self.config.max_length,
            return_tensors="pt",
            return_attention_mask=True,
        )

        input_ids = encoding["input_ids"].to(self.device)
        attention_mask = encoding["attention_mask"].to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_hidden_states=True,
            )

        if self.config.layer == -1:
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs.hidden_states[
                self.config.layer + 1
            ]

        hidden_states = hidden_states[0]

        word_ids = encoding.word_ids(batch_index=0)

        word_ids_tensor = torch.tensor(
            [
                -1 if word_id is None else word_id
                for word_id in word_ids
            ],
            device=self.device,
        )

        return hidden_states, word_ids_tensor

    def _embed_words(
        self,
        words: List[str],
    ) -> np.ndarray:
        """
        Convert original words into one vector per word.
        """

        real_word_indices = [
            index
            for index, word in enumerate(words)
            if word != "<PAD>"
        ]

        real_words = [
            words[index]
            for index in real_word_indices
        ]

        result = np.zeros(
            (
                len(words),
                self.embedding_dim,
            ),
            dtype=np.float32,
        )

        if not real_words:
            return result

        hidden_states, word_ids = self._get_hidden_states(
            real_words
        )

        for local_index, original_index in enumerate(
            real_word_indices
        ):
            mask = word_ids == local_index

            if not torch.any(mask):
                continue

            word_embedding = hidden_states[mask].mean(
                dim=0
            )

            result[original_index] = (
                word_embedding
                .detach()
                .cpu()
                .numpy()
                .astype(np.float32)
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
            doc_id=sequence.doc_id,
            strategy=sequence.strategy,
            mode=sequence.mode,
            central_node=sequence.central_node,
            best_neighbor=sequence.best_neighbor,
            words=sequence.words.copy(),
            embeddings=embeddings,
            embedding_dim=self.embedding_dim,
            sequence_length=sequence.length,
        )