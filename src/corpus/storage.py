import json
from pathlib import Path
from typing import Any

from src.corpus.contracts import Corpus
from src.preprocessing.contracts import ProcessedDocument, ProcessedToken


class CorpusStorage:
    DOCUMENTS_FILE = "documents.jsonl"
    METADATA_FILE = "metadata.json"

    def __init__(self, root: str | Path):
        self.root = Path(root)

    def save(self, corpus: Corpus) -> Path:
        output_dir = (
            self.root
            / corpus.name
            / corpus.split
        )

        output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        documents_path = output_dir / self.DOCUMENTS_FILE
        metadata_path = output_dir / self.METADATA_FILE

        self._save_documents(
            corpus.documents,
            documents_path,
        )

        self._save_metadata(
            corpus,
            metadata_path,
        )

        return output_dir

    def load(
        self,
        name: str,
        split: str,
    ) -> Corpus:
        input_dir = self.root / name / split

        if not input_dir.exists():
            raise FileNotFoundError(
                f"Corpus directory not found: {input_dir}"
            )

        documents_path = input_dir / self.DOCUMENTS_FILE
        metadata_path = input_dir / self.METADATA_FILE

        documents = self._load_documents(documents_path)
        metadata = self._load_metadata(metadata_path)

        return Corpus(
            name=metadata["name"],
            split=metadata["split"],
            documents=documents,
        )

    def _save_documents(
        self,
        documents: list[ProcessedDocument],
        path: Path,
    ) -> None:
        with path.open(
            "w",
            encoding="utf-8",
        ) as file:

            for document in documents:
                record = {
                    "doc_id": document.doc_id,
                    "text": document.text,
                    "labels": document.labels,
                    "tokens": [
                        {
                            "text": token.text,
                            "position": token.position,
                        }
                        for token in document.tokens
                    ],
                }

                file.write(
                    json.dumps(
                        record,
                        ensure_ascii=False,
                    )
                    + "\n"
                )

    def _load_documents(
        self,
        path: Path,
    ) -> list[ProcessedDocument]:

        if not path.exists():
            raise FileNotFoundError(
                f"Documents file not found: {path}"
            )

        documents = []

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line in file:
                record: dict[str, Any] = json.loads(line)

                tokens = [
                    ProcessedToken(
                        text=token["text"],
                        position=token["position"],
                    )
                    for token in record["tokens"]
                ]

                documents.append(
                    ProcessedDocument(
                        doc_id=record["doc_id"],
                        text=record["text"],
                        labels=record["labels"],
                        tokens=tokens,
                    )
                )

        return documents

    def _save_metadata(
        self,
        corpus: Corpus,
        path: Path,
    ) -> None:

        metadata = {
            "name": corpus.name,
            "split": corpus.split,
            "size": corpus.size,
        }

        with path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                metadata,
                file,
                indent=2,
                ensure_ascii=False,
            )

    def _load_metadata(
        self,
        path: Path,
    ) -> dict[str, Any]:

        if not path.exists():
            raise FileNotFoundError(
                f"Metadata file not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

