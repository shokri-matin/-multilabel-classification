from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence


@dataclass(frozen=True)
class LabelVocabulary:
    labels: List[str]
    label_to_index: Dict[str, int]

    @property
    def num_labels(self) -> int:
        return len(self.labels)

    def encode(self, labels: Sequence[str]) -> List[int]:
        encoded = [0] * self.num_labels

        for label in labels:
            if label not in self.label_to_index:
                raise ValueError(
                    f"Unknown label: '{label}'"
                )

            encoded[self.label_to_index[label]] = 1

        return encoded

    def decode(self, encoded: Sequence[int]) -> List[str]:
        if len(encoded) != self.num_labels:
            raise ValueError(
                f"Expected {self.num_labels} values, "
                f"got {len(encoded)}"
            )

        return [
            label
            for label, value in zip(self.labels, encoded)
            if value == 1
        ]


class LabelVocabularyBuilder:
    def build(
        self,
        documents: Iterable,
    ) -> LabelVocabulary:
        labels = set()

        for document in documents:
            for label in document.labels:
                labels.add(label)

        if not labels:
            raise ValueError(
                "No labels found in documents"
            )

        sorted_labels = sorted(labels)

        label_to_index = {
            label: index
            for index, label in enumerate(sorted_labels)
        }

        return LabelVocabulary(
            labels=sorted_labels,
            label_to_index=label_to_index,
        )