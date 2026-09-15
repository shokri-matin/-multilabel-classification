from typing import Optional

import torch
from torch import nn
from torch.optim import AdamW

from src.experiments.checkpoint import CheckpointManager
from src.experiments.classifier_factory import ClassifierFactory
from src.experiments.config import (
    ExperimentConfig,
    ExperimentResult,
)
from src.experiments.evaluator import ExperimentEvaluator
from src.experiments.experiment_context import ExperimentContext
from src.experiments.training_config import TrainingConfig
from src.experiments.trainer import TrainingManager


class ExperimentExecutor:
    def __init__(
        self,
        context: ExperimentContext,
        training_config: TrainingConfig,
        evaluator: ExperimentEvaluator | None = None,
    ):
        self.context = context
        self.training_config = training_config
        self.evaluator = evaluator

    def _build_model(
        self,
        data,
    ) -> nn.Module:

        return ClassifierFactory.create(
            classifier=data.experiment.classifier,
            input_dim=data.input_dim,
            num_labels=data.num_labels,
        )

    def _build_optimizer(
        self,
        model: nn.Module,
    ) -> torch.optim.Optimizer:

        return AdamW(
            model.parameters(),
            lr=self.training_config.learning_rate,
            weight_decay=self.training_config.weight_decay,
        )

    def _get_device(self) -> torch.device:

        if self.training_config.device == "auto":
            return torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

        return torch.device(
            self.training_config.device
        )

    def _build_evaluator(
        self,
        data,
    ) -> ExperimentEvaluator:

        if self.evaluator is not None:
            return self.evaluator

        return ExperimentEvaluator(
            label_encoder=data.train_dataset.label_encoder,
        )

    @staticmethod
    def _find_epoch_metrics(
        training_result,
        epoch: Optional[int],
    ):
        if epoch is None:
            return None

        for metrics in training_result.history.epochs:
            if metrics.epoch == epoch:
                return metrics

        return None

    def _build_result(
        self,
        experiment: ExperimentConfig,
        training_result,
        evaluation_result,
    ) -> ExperimentResult:

        best_epoch = training_result.best_epoch

        best_metrics = self._find_epoch_metrics(
            training_result,
            best_epoch,
        )

        train_loss = (
            best_metrics.train_loss
            if best_metrics is not None
            else None
        )

        validation_loss = (
            best_metrics.validation_loss
            if best_metrics is not None
            else training_result.best_validation_loss
        )

        return ExperimentResult(
            experiment_id=experiment.experiment_id,
            dataset=experiment.dataset,
            mode=experiment.mode,
            centrality=experiment.centrality,
            embedding=experiment.embedding,
            classifier=experiment.classifier,
            train_loss=train_loss,
            validation_loss=validation_loss,
            micro_precision=evaluation_result.micro_precision,
            micro_recall=evaluation_result.micro_recall,
            micro_f1=evaluation_result.micro_f1,
            macro_precision=evaluation_result.macro_precision,
            macro_recall=evaluation_result.macro_recall,
            macro_f1=evaluation_result.macro_f1,
            hamming_loss=evaluation_result.hamming_loss,
            best_epoch=best_epoch,
            training_epochs=training_result.history.num_epochs,
            checkpoint_path=training_result.checkpoint_path,
            status="completed",
            error=None,
        )

    def execute(self) -> ExperimentResult:

        experiment = self.context.experiment

        try:
            # --------------------------------------------------
            # 1. Build experiment data
            # --------------------------------------------------
            data = self.context.build()

            # --------------------------------------------------
            # 2. Resolve device
            # --------------------------------------------------
            device = self._get_device()

            # --------------------------------------------------
            # 3. Build classifier
            # --------------------------------------------------
            model = self._build_model(data)
            model.to(device)

            # --------------------------------------------------
            # 4. Loss
            # --------------------------------------------------
            criterion = nn.BCEWithLogitsLoss()

            # --------------------------------------------------
            # 5. Optimizer
            # --------------------------------------------------
            optimizer = self._build_optimizer(model)

            # --------------------------------------------------
            # 6. Training
            # --------------------------------------------------
            training_manager = TrainingManager(
                model=model,
                criterion=criterion,
                optimizer=optimizer,
                config=self.training_config,
                device=device,
            )

            training_result = training_manager.fit(
                train_dataloader=data.train_loader,
                validation_dataloader=data.validation_loader,
            )

            # --------------------------------------------------
            # 7. Restore best checkpoint
            # --------------------------------------------------
            checkpoint_path = (
                training_result.checkpoint_path
            )

            if checkpoint_path is not None:

                checkpoint_manager = CheckpointManager(
                    checkpoint_dir=(
                        self.training_config.checkpoint_dir
                    ),
                    save_best_only=(
                        self.training_config.save_best_only
                    ),
                )

                checkpoint_manager.load(
                    path=checkpoint_path,
                    model=model,
                    map_location=device,
                )

            # --------------------------------------------------
            # 8. Evaluation
            # --------------------------------------------------
            evaluator = self._build_evaluator(data)

            evaluation_result = evaluator.evaluate(
                model=model,
                dataloader=data.test_loader,
                document_labels=(
                    self.context
                    .dataset_context
                    .labels_for_split("test")
                ),
                device=device,
            )

            # --------------------------------------------------
            # 9. Build final result
            # --------------------------------------------------
            return self._build_result(
                experiment=experiment,
                training_result=training_result,
                evaluation_result=evaluation_result,
            )

        except Exception as exc:

            return ExperimentResult(
                experiment_id=experiment.experiment_id,
                dataset=experiment.dataset,
                mode=experiment.mode,
                centrality=experiment.centrality,
                embedding=experiment.embedding,
                classifier=experiment.classifier,
                train_loss=None,
                validation_loss=None,
                micro_precision=None,
                micro_recall=None,
                micro_f1=None,
                macro_precision=None,
                macro_recall=None,
                macro_f1=None,
                hamming_loss=None,
                best_epoch=None,
                training_epochs=0,
                checkpoint_path=None,
                status="failed",
                error=str(exc),
            )