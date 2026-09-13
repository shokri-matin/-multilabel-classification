from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

from src.embedding.base import EmbeddingOutput


@dataclass(frozen=True)
class ClassifierSample:
    doc_id: str | None
    strategy: str
    mode: str
    central_node: str
    best_neighbor: str
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor


class LabelEncoder:
    def __init__(self, label_vocabulary: Sequence[str]):
        vocabulary = list(label_vocabulary)

        if not vocabulary:
            raise ValueError(
                "label_vocabulary must not be empty"
            )

        if len(vocabulary) != len(set(vocabulary)):
            raise ValueError(
                "label_vocabulary must contain unique labels"
            )

        self.label_vocabulary = vocabulary
        self.label_to_index: Dict[str, int] = {
            label: index
            for index, label in enumerate(vocabulary)
        }

    @property
    def num_labels(self) -> int:
        return len(self.label_vocabulary)

    def encode(self, labels: Sequence[str]) -> torch.Tensor:
        encoded = torch.zeros(
            self.num_labels,
            dtype=torch.float32,
        )

        for label in labels:
            if label not in self.label_to_index:
                raise ValueError(
                    f"Unknown label: '{label}'"
                )

            encoded[self.label_to_index[label]] = 1.0

        return encoded

    def decode(self, encoded: torch.Tensor) -> List[str]:
        if encoded.ndim != 1:
            raise ValueError(
                "encoded labels must have shape (num_labels,)"
            )

        if encoded.shape[0] != self.num_labels:
            raise ValueError(
                f"Expected {self.num_labels} labels, "
                f"got {encoded.shape[0]}"
            )

        return [
            label
            for index, label in enumerate(self.label_vocabulary)
            if encoded[index].item() > 0.5
        ]


class ClassifierDataset(Dataset):
    def __init__(
        self,
        embeddings: Sequence[EmbeddingOutput],
        document_labels: Dict[str, Sequence[str]],
        label_encoder: LabelEncoder,
        padding_token: str = "<PAD>",
    ):
        if not embeddings:
            raise ValueError(
                "embeddings must not be empty"
            )

        if not padding_token:
            raise ValueError(
                "padding_token must not be empty"
            )

        self.embeddings = list(embeddings)
        self.document_labels = document_labels
        self.label_encoder = label_encoder
        self.padding_token = padding_token

        self.samples = self._build_samples()

    def _build_samples(self) -> List[ClassifierSample]:
        samples = []

        for embedding_output in self.embeddings:
            doc_id = embedding_output.doc_id

            if doc_id is None:
                raise ValueError(
                    "EmbeddingOutput.doc_id must not be None "
                    "when building ClassifierDataset"
                )

            if doc_id not in self.document_labels:
                raise ValueError(
                    f"No labels found for document: '{doc_id}'"
                )

            words = embedding_output.words

            attention_mask = torch.tensor(
                [
                    0 if word == self.padding_token else 1
                    for word in words
                ],
                dtype=torch.long,
            )

            inputs_embeds = torch.from_numpy(
                np.asarray(
                    embedding_output.embeddings,
                    dtype=np.float32,
                )
            )

            labels = self.label_encoder.encode(
                self.document_labels[doc_id]
            )

            samples.append(
                ClassifierSample(
                    doc_id=doc_id,
                    strategy=embedding_output.strategy,
                    mode=embedding_output.mode,
                    central_node=embedding_output.central_node,
                    best_neighbor=embedding_output.best_neighbor,
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    labels=labels,
                )
            )

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> Dict[str, object]:
        sample = self.samples[index]

        return {
            "doc_id": sample.doc_id,
            "strategy": sample.strategy,
            "mode": sample.mode,
            "central_node": sample.central_node,
            "best_neighbor": sample.best_neighbor,
            "inputs_embeds": sample.inputs_embeds,
            "attention_mask": sample.attention_mask,
            "labels": sample.labels,
        }

    @property
    def input_dim(self) -> int:
        return self.samples[0].inputs_embeds.shape[-1]

    @property
    def sequence_length(self) -> int:
        return self.samples[0].inputs_embeds.shape[0]

    @property
    def num_labels(self) -> int:
        return self.label_encoder.num_labels