import json
from pathlib import Path
from typing import Any


class ArtifactStore:
    """
    Persistent JSON artifact storage for experiment metadata/results.

    The store is intentionally simple:
    - JSON files for metadata/results
    - deterministic paths
    - atomic writes
    """

    def __init__(self, root_dir: str | Path):
        if not root_dir:
            raise ValueError("root_dir must not be empty")

        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def _validate_name(self, name: str) -> None:
        if not name:
            raise ValueError("artifact name must not be empty")

        if "/" in name or "\\" in name:
            raise ValueError(
                "artifact name must not contain path separators"
            )

    def path(self, name: str) -> Path:
        self._validate_name(name)
        return self.root_dir / f"{name}.json"

    def exists(self, name: str) -> bool:
        return self.path(name).exists()

    def save_json(
        self,
        name: str,
        data: Any,
    ) -> Path:
        path = self.path(name)

        temporary_path = path.with_suffix(".tmp")

        with temporary_path.open(
            "w",
            encoding="utf-8",
        ) as file:
            json.dump(
                data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        temporary_path.replace(path)

        return path

    def load_json(self, name: str) -> Any:
        path = self.path(name)

        if not path.exists():
            raise FileNotFoundError(
                f"Artifact not found: {path}"
            )

        with path.open(
            "r",
            encoding="utf-8",
        ) as file:
            return json.load(file)

    def delete(self, name: str) -> None:
        path = self.path(name)

        if path.exists():
            path.unlink()