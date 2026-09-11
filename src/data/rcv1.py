from pathlib import Path
from typing import Dict, List, Tuple

from src.data.contracts import RawDocument


class RCV1Loader:
    """
    Loader for the tokenized RCV1 dataset.

    Expected directory structure:

        root/
        ├── lyrl2004_tokens_train
        ├── lyrl2004_tokens_test_pt0
        ├── lyrl2004_tokens_test_pt1
        ├── lyrl2004_tokens_test_pt2
        ├── lyrl2004_tokens_test_pt3
        └── rcv1-v2.topics.qrels

    The token files have the following format:

        .I 2286
        .W
        recov recov recov ...

    The qrels file has the following format:

        E11 2286 1
        ECAT 2286 1
        M11 2286 1
    """

    TRAIN_FILE = "lyrl2004_tokens_train.dat"

    TEST_FILES = (
        "lyrl2004_tokens_test_pt0.dat",
        "lyrl2004_tokens_test_pt1.dat",
        "lyrl2004_tokens_test_pt2.dat",
        "lyrl2004_tokens_test_pt3.dat",
    )

    QRELS_FILE = "rcv1-v2.topics.qrels"

    def __init__(self, root: str | Path):
        self.root = Path(root)

        if not self.root.exists():
            raise FileNotFoundError(
                f"RCV1 directory does not exist: {self.root}"
            )

        self._labels: Dict[str, List[str]] | None = None

    def load_train(self) -> List[RawDocument]:
        """
        Load the RCV1 training split.
        """

        labels = self._load_labels()

        documents = self._load_token_file(
            self.root / self.TRAIN_FILE,
            labels,
        )

        return documents

    def load_test(self) -> List[RawDocument]:
        """
        Load the complete RCV1 test split.

        The test split is distributed across four files.
        """

        labels = self._load_labels()

        documents: List[RawDocument] = []

        for filename in self.TEST_FILES:
            path = self.root / filename

            documents.extend(
                self._load_token_file(path, labels)
            )

        return documents

    def load_split(self, split: str) -> List[RawDocument]:
        """
        Load a dataset split.

        Supported splits:

            train
            test
        """

        if split == "train":
            return self.load_train()

        if split == "test":
            return self.load_test()

        raise ValueError(
            f"Unsupported split '{split}'. "
            "Expected 'train' or 'test'."
        )

    def _load_token_file(
        self,
        path: Path,
        labels: Dict[str, List[str]],
    ) -> List[RawDocument]:
        """
        Parse one RCV1 token file.
        """

        if not path.exists():
            raise FileNotFoundError(
                f"RCV1 token file not found: {path}"
            )

        documents: List[RawDocument] = []

        current_doc_id: str | None = None
        current_text: List[str] = []
        reading_text = False

        with path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            for raw_line in file:
                line = raw_line.strip()

                if not line:
                    continue

                # Document identifier
                if line.startswith(".I"):
                    if current_doc_id is not None:
                        documents.append(
                            self._build_document(
                                current_doc_id,
                                current_text,
                                labels,
                            )
                        )

                    parts = line.split(maxsplit=1)

                    if len(parts) != 2:
                        raise ValueError(
                            f"Invalid document ID line: {line}"
                        )

                    current_doc_id = parts[1].strip()
                    current_text = []
                    reading_text = False

                    continue

                # Beginning of document text
                if line == ".W":
                    reading_text = True
                    continue

                # Text
                if reading_text:
                    current_text.append(line)

        # Flush final document
        if current_doc_id is not None:
            documents.append(
                self._build_document(
                    current_doc_id,
                    current_text,
                    labels,
                )
            )

        return documents

    def _build_document(
        self,
        doc_id: str,
        text_lines: List[str],
        labels: Dict[str, List[str]],
    ) -> RawDocument:
        """
        Build a RawDocument from parsed text and labels.
        """

        text = " ".join(text_lines).strip()

        document_labels = labels.get(doc_id, [])

        return RawDocument(
            doc_id=f"rcv1_{doc_id}",
            text=text,
            labels=document_labels,
        )

    def _load_labels(self) -> Dict[str, List[str]]:
        """
        Load and cache RCV1 topic labels.
        """

        if self._labels is not None:
            return self._labels

        qrels_path = self.root / self.QRELS_FILE

        if not qrels_path.exists():
            raise FileNotFoundError(
                f"RCV1 qrels file not found: {qrels_path}"
            )

        labels: Dict[str, List[str]] = {}

        with qrels_path.open(
            "r",
            encoding="utf-8",
            errors="replace",
        ) as file:

            for line_number, raw_line in enumerate(
                file,
                start=1,
            ):
                line = raw_line.strip()

                if not line:
                    continue

                parts = line.split()

                if len(parts) != 3:
                    raise ValueError(
                        f"Invalid qrels line at "
                        f"{line_number}: {line}"
                    )

                topic, doc_id, relevance = parts

                # Only positive relevance is a label.
                if relevance != "1":
                    continue

                labels.setdefault(doc_id, []).append(topic)

        # Deterministic label order
        for doc_id in labels:
            labels[doc_id] = sorted(
                set(labels[doc_id])
            )

        self._labels = labels

        return labels
