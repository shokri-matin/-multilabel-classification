from pathlib import Path

from src.experiments.application import ExperimentApplication
from src.experiments.config import ExperimentConfig, ExperimentResult
from src.experiments.runner import ExperimentRunner


def create_experiment():
    return ExperimentConfig(
        dataset="aapd",
        mode="no_branch",
        centrality="closeness",
        embedding="bert",
        classifier="bert",
        top_k=200,
        max_nodes=50,
        sequence_length=25,
    )


def create_result(experiment):
    return ExperimentResult(
        experiment_id=experiment.experiment_id,
        dataset=experiment.dataset,
        mode=experiment.mode,
        centrality=experiment.centrality,
        embedding=experiment.embedding,
        classifier=experiment.classifier,
        train_loss=0.2,
        validation_loss=0.3,
        micro_precision=0.8,
        micro_recall=0.7,
        micro_f1=0.75,
        macro_precision=0.7,
        macro_recall=0.6,
        macro_f1=0.65,
        hamming_loss=0.1,
        best_epoch=2,
        training_epochs=3,
        checkpoint_path="results/checkpoints/best.pt",
        status="completed",
    )


class FakeExecutor:
    def __init__(self, experiment):
        self.experiment = experiment
        self.executed = False

    def execute(self):
        self.executed = True
        return create_result(self.experiment)


def test_application_runs_experiment(tmp_path):
    experiment = create_experiment()

    runner = ExperimentRunner(
        artifact_dir=tmp_path / "experiments"
    )

    executors = []

    def executor_factory(config):
        executor = FakeExecutor(config)
        executors.append(executor)
        return executor

    application = ExperimentApplication(
        runner=runner,
        executor_factory=executor_factory,
    )

    result = application.run(
        experiment=experiment,
        skip_completed=True,
    )

    assert result.status == "completed"
    assert result.experiment_id == experiment.experiment_id
    assert len(executors) == 1
    assert executors[0].executed is True

    result_path = (
        Path(tmp_path)
        / "experiments"
        / f"{experiment.experiment_id}.json"
    )

    assert result_path.exists()


def test_application_skips_completed_experiment(tmp_path):
    experiment = create_experiment()

    runner = ExperimentRunner(
        artifact_dir=tmp_path / "experiments"
    )

    executors = []

    def executor_factory(config):
        executor = FakeExecutor(config)
        executors.append(executor)
        return executor

    application = ExperimentApplication(
        runner=runner,
        executor_factory=executor_factory,
    )

    first_result = application.run(
        experiment=experiment,
        skip_completed=True,
    )

    second_result = application.run(
        experiment=experiment,
        skip_completed=True,
    )

    assert first_result.status == "completed"
    assert second_result.status == "completed"

    # Executor should only run once.
    assert len(executors) == 1
    assert executors[0].executed is True


def test_application_can_force_rerun(tmp_path):
    experiment = create_experiment()

    runner = ExperimentRunner(
        artifact_dir=tmp_path / "experiments"
    )

    executors = []

    def executor_factory(config):
        executor = FakeExecutor(config)
        executors.append(executor)
        return executor

    application = ExperimentApplication(
        runner=runner,
        executor_factory=executor_factory,
    )

    application.run(
        experiment=experiment,
        skip_completed=True,
    )

    application.run(
        experiment=experiment,
        skip_completed=False,
    )

    assert len(executors) == 2