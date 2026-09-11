
from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class RawDocument:
    """
    Raw document loaded directly from a dataset.

    No preprocessing should be performed here.
    """

    doc_id: str
    text: str
    labels: List[str]

