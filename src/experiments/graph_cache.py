import pickle
from pathlib import Path
from typing import Dict

import networkx as nx

from src.graph.centrality import StrategyResult


class GraphArtifactCache:
    """
    Persistent cache for document-level word graphs and
    their centrality results.

    Each document gets one artifact containing:
    - doc_id
    - word graph
    - all centrality strategy results
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
    def _validate_doc_id(doc_id: str) -> None:
        if not doc_id:
            raise ValueError("doc_id must not be empty")

        if "/" in doc_id or "\\" in doc_id:
            raise ValueError(
                "doc_id must not contain path separators"
            )

    def _path(self, doc_id: str) -> Path:
        self._validate_doc_id(doc_id)

        return self.root_dir / f"{doc_id}.pkl"

    def exists(self, doc_id: str) -> bool:
        return self._path(doc_id).exists()

    def save(
        self,
        doc_id: str,
        graph: nx.Graph,
        centralities: Dict[str, StrategyResult],
    ) -> Path:

        if not isinstance(graph, nx.Graph):
            raise TypeError(
                "graph must be an instance of networkx.Graph"
            )

        path = self._path(doc_id)

        artifact = {
            "doc_id": doc_id,
            "graph": graph,
            "centralities": centralities,
        }

        temporary_path = path.with_suffix(".tmp")

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
        doc_id: str,
    ) -> tuple[nx.Graph, Dict[str, StrategyResult]]:

        path = self._path(doc_id)

        if not path.exists():
            raise FileNotFoundError(
                f"Graph artifact not found: {path}"
            )

        with path.open("rb") as file:
            artifact = pickle.load(file)

        if artifact.get("doc_id") != doc_id:
            raise ValueError(
                f"Artifact doc_id mismatch: "
                f"expected '{doc_id}', "
                f"got '{artifact.get('doc_id')}'"
            )

        graph = artifact.get("graph")
        centralities = artifact.get("centralities")

        if not isinstance(graph, nx.Graph):
            raise ValueError(
                "Artifact does not contain a valid graph"
            )

        if not isinstance(centralities, dict):
            raise ValueError(
                "Artifact does not contain valid centralities"
            )

        return graph, centralities

    def delete(self, doc_id: str) -> None:
        path = self._path(doc_id)

        if path.exists():
            path.unlink()