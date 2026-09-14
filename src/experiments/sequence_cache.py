import pickle
from pathlib import Path
from typing import List

from src.sequence.word_sequence import WordSequence


class SequenceArtifactCache:
    """
    Persistent cache for graph-derived word sequences.

    Cache key:
        dataset / doc_id / centrality / mode

    Embedding and classifier are intentionally NOT part
    of the cache key because sequences are independent
    of the embedding and classifier.
    """

    def __init__(self, root_dir: str | Path):
        if not root_dir:
            raise ValueError("root_dir must not be empty")

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

    @staticmethod
    def _validate_component(
        value: str,
        name: str,
    ) -> None:
        if not value:
            raise ValueError(
                f"{name} must not be empty"
            )

        if "/" in value or "\\" in value:
            raise ValueError(
                f"{name} must not contain path separators"
            )

    def _path(
        self,
        dataset: str,
        doc_id: str,
        centrality: str,
        mode: str,
    ) -> Path:

        self._validate_component(
            dataset,
            "dataset",
        )

        self._validate_component(
            doc_id,
            "doc_id",
        )

        self._validate_component(
            centrality,
            "centrality",
        )

        self._validate_component(
            mode,
            "mode",
        )

        centrality_id = centrality.replace(
            "*",
            "_",
        )

        directory = (
            self.root_dir
            / dataset
            / doc_id
        )

        directory.mkdir(
            parents=True,
            exist_ok=True,
        )

        return (
            directory
            / f"{centrality_id}__{mode}.pkl"
        )

    def exists(
        self,
        dataset: str,
        doc_id: str,
        centrality: str,
        mode: str,
    ) -> bool:

        return self._path(
            dataset=dataset,
            doc_id=doc_id,
            centrality=centrality,
            mode=mode,
        ).exists()

    def save(
        self,
        dataset: str,
        doc_id: str,
        centrality: str,
        mode: str,
        sequences: List[WordSequence],
    ) -> Path:

        if not isinstance(sequences, list):
            raise TypeError(
                "sequences must be a list"
            )

        for sequence in sequences:
            if not isinstance(
                sequence,
                WordSequence,
            ):
                raise TypeError(
                    "all sequences must be "
                    "WordSequence instances"
                )

        path = self._path(
            dataset=dataset,
            doc_id=doc_id,
            centrality=centrality,
            mode=mode,
        )

        artifact = {
            "dataset": dataset,
            "doc_id": doc_id,
            "centrality": centrality,
            "mode": mode,
            "sequences": sequences,
        }

        temporary_path = path.with_suffix(
            ".tmp"
        )

        with temporary_path.open("wb") as file:
            pickle.dump(
                artifact,
                file,
                protocol=pickle.HIGHEST_PROTOCOL,
            )

        temporary_path.replace(path)

        return path

    def load(
        self,
        dataset: str,
        doc_id: str,
        centrality: str,
        mode: str,
    ) -> List[WordSequence]:

        path = self._path(
            dataset=dataset,
            doc_id=doc_id,
            centrality=centrality,
            mode=mode,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Sequence artifact not found: {path}"
            )

        with path.open("rb") as file:
            artifact = pickle.load(file)

        if artifact.get("dataset") != dataset:
            raise ValueError(
                "Artifact dataset mismatch"
            )

        if artifact.get("doc_id") != doc_id:
            raise ValueError(
                "Artifact doc_id mismatch"
            )

        if artifact.get("centrality") != centrality:
            raise ValueError(
                "Artifact centrality mismatch"
            )

        if artifact.get("mode") != mode:
            raise ValueError(
                "Artifact mode mismatch"
            )

        sequences = artifact.get("sequences")

        if not isinstance(sequences, list):
            raise ValueError(
                "Artifact does not contain "
                "a valid sequence list"
            )

        for sequence in sequences:
            if not isinstance(
                sequence,
                WordSequence,
            ):
                raise ValueError(
                    "Artifact contains an invalid "
                    "WordSequence"
                )

        return sequences

    def delete(
        self,
        dataset: str,
        doc_id: str,
        centrality: str,
        mode: str,
    ) -> None:

        path = self._path(
            dataset=dataset,
            doc_id=doc_id,
            centrality=centrality,
            mode=mode,
        )

        if path.exists():
            path.unlink()