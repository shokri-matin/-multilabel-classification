from dataclasses import asdict
from pathlib import Path
from typing import Callable

from src.experiments.artifacts import ArtifactStore
from src.experiments.config import (
    ExperimentConfig,
    ExperimentResult,
)


class ExperimentRunner:
    """
    Orchestrates one experiment.

    Heavy reusable artifacts such as:
    - processed corpus
    - word graphs
    - centrality results

    should be prepared once and reused across experiments.
    """

    def __init__(
        self,
        artifact_dir: str | Path = "results/experiments",
    ):
        self.artifacts = ArtifactStore(artifact_dir)

    @staticmethod
    def _result_to_dict(
        result: ExperimentResult,
    ) -> dict:
        return asdict(result)

    def result_exists(
        self,
        experiment: ExperimentConfig,
    ) -> bool:
        return self.artifacts.exists(
            experiment.experiment_id
        )

    def save_result(
        self,
        result: ExperimentResult,
    ) -> Path:
        return self.artifacts.save_json(
            result.experiment_id,
            self._result_to_dict(result),
        )

    def load_result(
        self,
        experiment: ExperimentConfig,
    ) -> ExperimentResult:
        data = self.artifacts.load_json(
            experiment.experiment_id
        )

        return ExperimentResult(**data)

    def run(
        self,
        experiment: ExperimentConfig,
        execute: Callable[
            [ExperimentConfig],
            ExperimentResult,
        ],
        skip_completed: bool = True,
    ) -> ExperimentResult:

        if skip_completed and self.result_exists(experiment):
            result = self.load_result(experiment)

            if result.status == "completed":
                return result

        result = execute(experiment)

        self.save_result(result)

        return result