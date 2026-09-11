from pathlib import Path
from typing import List

from src.data.contracts import RawDocument


class AAPDLoader:
    """
    Loader for the AAPD dataset.

    Expected directory structure:

        root/
        ├── text_train
        ├── text_validation
        ├── text_test
        ├── label_train
        ├── label_validation
        └── label_test

    Each line in a text file represents one document.
    Each corresponding line in the labels file contains
    the labels of that document.
    """

    SPLITS = {
        "train": ("text_train", "label_train"),
        "validation": ("text_validation", "label_validation"),
        "test": ("text_test", "label_test"),
    }

    def __init__(self, root: str | Path):
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"AAPD directory does not exist: {self.root}"
            )

    def load_split(self, split: str) -> List[RawDocument]:
        """
        Load one dataset split.

        Parameters
        ----------
        split:
            One of: train, validation, test.
        """

        if split not in self.SPLITS:
            raise ValueError(
                f"Unknown split '{split}'. "
                f"Expected one of: {list(self.SPLITS)}"
            )

        text_file_name, label_file_name = self.SPLITS[split]

        text_path = self.root / text_file_name
        label_path = self.root / label_file_name

        if not text_path.exists():
            raise FileNotFoundError(
                f"Text file not found: {text_path}"
            )

        if not label_path.exists():
            raise FileNotFoundError(
                f"Label file not found: {label_path}"
            )

        texts = self._read_lines(text_path)
        labels = self._read_lines(label_path)

        if len(texts) != len(labels):
            raise ValueError(
                f"Number of documents ({len(texts)}) does not match "
                f"number of label rows ({len(labels)}) "
                f"for split '{split}'."
            )

        documents = []

        for index, (text, label_line) in enumerate(zip(texts, labels)):
            document_labels = self._parse_labels(label_line)

            documents.append(
                RawDocument(
                    doc_id=f"aapd_{split}_{index}",
                    text=text,
                    labels=document_labels,
                )
            )

        return documents

    @staticmethod
    def _read_lines(path: Path) -> List[str]:
        """
        Read a text file while preserving one line per document.
        """

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:
            return [line.rstrip("\n\r") for line in file]

    @staticmethod
    def _parse_labels(line: str) -> List[str]: 
        """
        Parse an AAPD label line.

        AAPD labels are whitespace-separated.
        """

        return line.strip().split()