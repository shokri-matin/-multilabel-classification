from dataclasses import dataclass
from typing import List

from src.preprocessing.contracts import ProcessedDocument


@dataclass
class Corpus:
    """
    A collection of processed documents for a specific split.
    """

    name: str
    split: str
    documents: List[ProcessedDocument]

    @property
    def size(self) -> int:
        return len(self.documents)

    @property
    def document_ids(self) -> List[str]:
        return [document.doc_id for document in self.documents]

    @property
    def labels(self) -> List[List[str]]:
        return [document.labels for document in self.documents]
