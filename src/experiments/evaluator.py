from typing import Dict

import numpy as np
import torch

from src.classifier.dataset import LabelEncoder
from src.evaluation.multilabel import (
    DocumentAggregator,
    EvaluationResult,
    MultiLabelEvaluator,
    evaluate_document_level,
)


class ExperimentEvaluator:
    def __init__(
        self,
        label_encoder: LabelEncoder,
        evaluator: MultiLabelEvaluator | None = None,
    ):
        if label_encoder is None:
            raise ValueError(
                "label_encoder must not be None"
            )

        self.label_encoder = label_encoder

        self.evaluator = (
            evaluator
            if evaluator is not None
            else MultiLabelEvaluator()
        )

    @torch.no_grad()
    def evaluate(
        self,
        model: torch.nn.Module,
        dataloader,
        document_labels: Dict[str, object],
        device: torch.device,
    ) -> EvaluationResult:

        model.eval()

        all_doc_ids: list[str] = []
        all_probabilities: list[np.ndarray] = []

        for batch in dataloader:
            inputs_embeds = batch.inputs_embeds.to(device)
            attention_mask = batch.attention_mask.to(device)

            logits = model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
            )

            probabilities = torch.sigmoid(logits)

            all_doc_ids.extend(batch.doc_ids)

            all_probabilities.append(
                probabilities.detach()
                .cpu()
                .numpy()
            )

        if not all_probabilities:
            raise ValueError(
                "No predictions were generated"
            )

        probabilities = np.concatenate(
            all_probabilities,
            axis=0,
        )

        document_predictions = (
            DocumentAggregator.aggregate_mean(
                doc_ids=all_doc_ids,
                probabilities=probabilities,
            )
        )

        return evaluate_document_level(
            evaluator=self.evaluator,
            document_predictions=document_predictions,
            document_labels=document_labels,
            label_encoder=self.label_encoder,
        )