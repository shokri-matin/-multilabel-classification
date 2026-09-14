from pathlib import Path
from typing import List

import numpy as np

from src.embedding.base import EmbeddingOutput


class EmbeddingArtifactCache:
    """
    Persistent cache for embedding outputs.

    Cache key:

        dataset / doc_id / centrality / mode / embedding

    Classifier is intentionally excluded because the same
    embedding can be reused by multiple classifiers.
    """

    SUPPORTED_EMBEDDINGS = {
        "bert",
        "fasttext",
    }

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
        embedding: str,
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

        self._validate_component(
            embedding,
            "embedding",
        )

        if embedding not in self.SUPPORTED_EMBEDDINGS:
            raise ValueError(
                f"Unsupported embedding: {embedding}"
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
            / (
                f"{centrality_id}"
                f"__{mode}"
                f"__{embedding}.npz"
            )
        )

    def exists(
        self,
        dataset: str,
        doc_id: str,
        centrality: str,
        mode: str,
        embedding: str,
    ) -> bool:

        return self._path(
            dataset=dataset,
            doc_id=doc_id,
            centrality=centrality,
            mode=mode,
            embedding=embedding,
        ).exists()

    def save(
        self,
        dataset: str,
        doc_id: str,
        centrality: str,
        mode: str,
        embedding: str,
        outputs: List[EmbeddingOutput],
    ) -> Path:

        if not isinstance(outputs, list):
            raise TypeError(
                "outputs must be a list"
            )

        for output in outputs:
            if not isinstance(
                output,
                EmbeddingOutput,
            ):
                raise TypeError(
                    "all outputs must be "
                    "EmbeddingOutput instances"
                )

        path = self._path(
            dataset=dataset,
            doc_id=doc_id,
            centrality=centrality,
            mode=mode,
            embedding=embedding,
        )

        if outputs:
            embeddings = np.stack(
                [
                    output.embeddings
                    for output in outputs
                ]
            )

            words = np.array(
                [
                    output.words
                    for output in outputs
                ],
                dtype=object,
            )

            doc_ids = np.array(
                [
                    output.doc_id
                    for output in outputs
                ],
                dtype=object,
            )

            strategies = np.array(
                [
                    output.strategy
                    for output in outputs
                ],
                dtype=object,
            )

            modes = np.array(
                [
                    output.mode
                    for output in outputs
                ],
                dtype=object,
            )

            central_nodes = np.array(
                [
                    output.central_node
                    for output in outputs
                ],
                dtype=object,
            )

            best_neighbors = np.array(
                [
                    output.best_neighbor
                    for output in outputs
                ],
                dtype=object,
            )

            embedding_dim = (
                outputs[0].embedding_dim
            )

            sequence_length = (
                outputs[0].sequence_length
            )

        else:
            embeddings = np.empty(
                (0, 0, 0),
                dtype=np.float32,
            )

            words = np.empty(
                (0, 0),
                dtype=object,
            )

            doc_ids = np.empty(
                (0,),
                dtype=object,
            )

            strategies = np.empty(
                (0,),
                dtype=object,
            )

            modes = np.empty(
                (0,),
                dtype=object,
            )

            central_nodes = np.empty(
                (0,),
                dtype=object,
            )

            best_neighbors = np.empty(
                (0,),
                dtype=object,
            )

            embedding_dim = 0
            sequence_length = 0

        temporary_path = path.with_suffix(
            ".tmp.npz"
        )

        np.savez_compressed(
            temporary_path,
            embeddings=embeddings,
            words=words,
            doc_ids=doc_ids,
            strategies=strategies,
            modes=modes,
            central_nodes=central_nodes,
            best_neighbors=best_neighbors,
            embedding_dim=np.array(
                embedding_dim,
                dtype=np.int64,
            ),
            sequence_length=np.array(
                sequence_length,
                dtype=np.int64,
            ),
        )

        temporary_path.replace(path)

        return path

    def load(
        self,
        dataset: str,
        doc_id: str,
        centrality: str,
        mode: str,
        embedding: str,
    ) -> List[EmbeddingOutput]:

        path = self._path(
            dataset=dataset,
            doc_id=doc_id,
            centrality=centrality,
            mode=mode,
            embedding=embedding,
        )

        if not path.exists():
            raise FileNotFoundError(
                f"Embedding artifact not found: {path}"
            )

        data = np.load(
            path,
            allow_pickle=True,
        )

        embeddings = data["embeddings"]
        words = data["words"]
        doc_ids = data["doc_ids"]
        strategies = data["strategies"]
        modes = data["modes"]
        central_nodes = data["central_nodes"]
        best_neighbors = data["best_neighbors"]

        embedding_dim = int(
            data["embedding_dim"].item()
        )

        sequence_length = int(
            data["sequence_length"].item()
        )

        outputs = []

        for index in range(
            embeddings.shape[0]
        ):
            outputs.append(
                EmbeddingOutput(
                    doc_id=(
                        doc_ids[index].item()
                        if hasattr(
                            doc_ids[index],
                            "item",
                        )
                        else doc_ids[index]
                    ),
                    strategy=(
                        strategies[index].item()
                        if hasattr(
                            strategies[index],
                            "item",
                        )
                        else strategies[index]
                    ),
                    mode=(
                        modes[index].item()
                        if hasattr(
                            modes[index],
                            "item",
                        )
                        else modes[index]
                    ),
                    central_node=(
                        central_nodes[index].item()
                        if hasattr(
                            central_nodes[index],
                            "item",
                        )
                        else central_nodes[index]
                    ),
                    best_neighbor=(
                        best_neighbors[index].item()
                        if hasattr(
                            best_neighbors[index],
                            "item",
                        )
                        else best_neighbors[index]
                    ),
                    words=[
                        (
                            word.item()
                            if hasattr(
                                word,
                                "item",
                            )
                            else word
                        )
                        for word in words[index]
                    ],
                    embeddings=np.asarray(
                        embeddings[index],
                        dtype=np.float32,
                    ),
                    embedding_dim=embedding_dim,
                    sequence_length=sequence_length,
                )
            )

        return outputs

    def delete(
        self,
        dataset: str,
        doc_id: str,
        centrality: str,
        mode: str,
        embedding: str,
    ) -> None:

        path = self._path(
            dataset=dataset,
            doc_id=doc_id,
            centrality=centrality,
            mode=mode,
            embedding=embedding,
        )

        if path.exists():
            path.unlink()