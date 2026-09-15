from src.experiments.config import ExperimentConfig, ExperimentResult
from src.experiments.executor import ExperimentExecutor
from src.experiments.runner import ExperimentRunner


class ExperimentApplication:
    def __init__(
        self,
        runner: ExperimentRunner,
        executor_factory,
    ):
        self.runner = runner
        self.executor_factory = executor_factory

    def run(
        self,
        experiment: ExperimentConfig,
        skip_completed: bool = True,
    ) -> ExperimentResult:

        def execute(
            config: ExperimentConfig,
        ) -> ExperimentResult:

            executor = self.executor_factory(config)

            return executor.execute()

        return self.runner.run(
            experiment=experiment,
            execute=execute,
            skip_completed=skip_completed,
        )