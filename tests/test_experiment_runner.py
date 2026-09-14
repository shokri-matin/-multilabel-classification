from pathlib import Path

import pytest

from src.experiments.artifacts import ArtifactStore
from src.experiments.config import ExperimentConfig
from src.experiments.config import ExperimentResult
from src.experiments.runner import ExperimentRunner


def make_config() -> ExperimentConfig:
    return ExperimentConfig(
        dataset="aapd",
        mode="no_branch",
        centrality="closeness",
        embedding="fasttext",
        classifier="bert",
        top_k=200,
        max_nodes=25,
        sequence_length=25,
    )


def make_result(
    experiment_id: str,
    status: str = "completed",
) -> ExperimentResult:

    return ExperimentResult(
        experiment_id=experiment_id,
        dataset="aapd",
        mode="no_branch",
        centrality="closeness",
        embedding="fasttext",
        classifier="bert",
        train_loss=0.1,
        validation_loss=0.2,
        micro_precision=0.8,
        micro_recall=0.7,
        micro_f1=0.75,
        macro_precision=0.6,
        macro_recall=0.5,
        macro_f1=0.55,
        hamming_loss=0.1,
        best_epoch=2,
        training_epochs=3,
        checkpoint_path="results/checkpoints/best.pt",
        status=status,
    )


def test_artifact_store_save_and_load(tmp_path: Path):
    store = ArtifactStore(tmp_path)

    data = {
        "name": "test",
        "value": 42,
    }

    path = store.save_json("artifact", data)

    assert path.exists()
    assert store.exists("artifact")
    assert store.load_json("artifact") == data


def test_artifact_store_delete(tmp_path: Path):
    store = ArtifactStore(tmp_path)

    store.save_json(
        "artifact",
        {"value": 1},
    )

    assert store.exists("artifact")

    store.delete("artifact")

    assert not store.exists("artifact")


def test_artifact_store_missing_file(tmp_path: Path):
    store = ArtifactStore(tmp_path)

    with pytest.raises(FileNotFoundError):
        store.load_json("missing")


def test_artifact_store_rejects_empty_name(tmp_path: Path):
    store = ArtifactStore(tmp_path)

    with pytest.raises(ValueError):
        store.save_json("", {"value": 1})


def test_artifact_store_rejects_path_separator(tmp_path: Path):
    store = ArtifactStore(tmp_path)

    with pytest.raises(ValueError):
        store.save_json(
            "nested/artifact",
            {"value": 1},
        )


def test_runner_save_and_load_result(tmp_path: Path):
    runner = ExperimentRunner(tmp_path)
    config = make_config()

    result = make_result(
        config.experiment_id
    )

    path = runner.save_result(result)

    assert path.exists()
    assert runner.result_exists(config)

    loaded = runner.load_result(config)

    assert loaded.experiment_id == result.experiment_id
    assert loaded.status == "completed"
    assert loaded.micro_f1 == 0.75


def test_runner_executes_experiment(tmp_path: Path):
    runner = ExperimentRunner(tmp_path)
    config = make_config()

    calls = []

    def execute(experiment):
        calls.append(experiment.experiment_id)

        return make_result(
            experiment.experiment_id
        )

    result = runner.run(
        experiment=config,
        execute=execute,
    )

    assert result.status == "completed"
    assert calls == [config.experiment_id]
    assert runner.result_exists(config)


def test_runner_skips_completed_experiment(tmp_path: Path):
    runner = ExperimentRunner(tmp_path)
    config = make_config()

    existing_result = make_result(
        config.experiment_id
    )

    runner.save_result(existing_result)

    calls = []

    def execute(experiment):
        calls.append(experiment.experiment_id)

        return make_result(
            experiment.experiment_id
        )

    result = runner.run(
        experiment=config,
        execute=execute,
        skip_completed=True,
    )

    assert result.status == "completed"
    assert calls == []


def test_runner_reruns_when_skip_completed_is_false(
    tmp_path: Path,
):
    runner = ExperimentRunner(tmp_path)
    config = make_config()

    existing_result = make_result(
        config.experiment_id
    )

    runner.save_result(existing_result)

    calls = []

    def execute(experiment):
        calls.append(experiment.experiment_id)

        return make_result(
            experiment.experiment_id
        )

    runner.run(
        experiment=config,
        execute=execute,
        skip_completed=False,
    )

    assert calls == [config.experiment_id]


def test_runner_reruns_failed_experiment(tmp_path: Path):
    runner = ExperimentRunner(tmp_path)
    config = make_config()

    failed_result = make_result(
        config.experiment_id,
        status="failed",
    )

    runner.save_result(failed_result)

    calls = []

    def execute(experiment):
        calls.append(experiment.experiment_id)

        return make_result(
            experiment.experiment_id,
            status="completed",
        )

    result = runner.run(
        experiment=config,
        execute=execute,
        skip_completed=True,
    )

    assert calls == [config.experiment_id]
    assert result.status == "completed"