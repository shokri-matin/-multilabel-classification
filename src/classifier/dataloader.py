from dataclasses import dataclass
from typing import Dict, List

import torch
from torch.utils.data import DataLoader

from src.classifier.dataset import ClassifierDataset


@dataclass
class ClassifierBatch:
    inputs_embeds: torch.Tensor
    attention_mask: torch.Tensor
    labels: torch.Tensor
    doc_ids: List[str | None]
    strategies: List[str]
    modes: List[str]
    central_nodes: List[str]
    best_neighbors: List[str]


def classifier_collate_fn(
    batch: List[Dict[str, object]],
) -> ClassifierBatch:
    if not batch:
        raise ValueError("batch must not be empty")

    inputs_embeds = torch.stack(
        [
            sample["inputs_embeds"]
            for sample in batch
        ]
    )

    attention_mask = torch.stack(
        [
            sample["attention_mask"]
            for sample in batch
        ]
    )

    labels = torch.stack(
        [
            sample["labels"]
            for sample in batch
        ]
    )

    doc_ids = [
        sample["doc_id"]
        for sample in batch
    ]

    strategies = [
        sample["strategy"]
        for sample in batch
    ]

    modes = [
        sample["mode"]
        for sample in batch
    ]

    central_nodes = [
        sample["central_node"]
        for sample in batch
    ]

    best_neighbors = [
        sample["best_neighbor"]
        for sample in batch
    ]

    return ClassifierBatch(
        inputs_embeds=inputs_embeds,
        attention_mask=attention_mask,
        labels=labels,
        doc_ids=doc_ids,
        strategies=strategies,
        modes=modes,
        central_nodes=central_nodes,
        best_neighbors=best_neighbors,
    )


class ClassifierDataLoader:
    def __init__(
        self,
        dataset: ClassifierDataset,
        batch_size: int = 8,
        shuffle: bool = False,
    ):
        if batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        self.dataset = dataset
        self.batch_size = batch_size
        self.shuffle = shuffle

        self.loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            collate_fn=classifier_collate_fn,
        )

    def __len__(self) -> int:
        return len(self.loader)

    def __iter__(self):
        return iter(self.loader)