from dataclasses import dataclass
from typing import List, Sequence

import numpy as np
from sklearn.metrics import (
    f1_score,
    hamming_loss,
    precision_score,
    recall_score,
)


@dataclass(frozen=True)
class EvaluationResult:
    micro_precision: float
    micro_recall: float
    micro_f1: float

    macro_precision: float
    macro_recall: float
    macro_f1: float

    hamming_loss: float

    threshold: float
    samples: int
    labels: int

class MultiLabelEvaluator:
    def __init__(
        self,
        threshold: float = 0.5,
    ):
        if not 0.0 < threshold < 1.0:
            raise ValueError(
                "threshold must be between 0 and 1"
            )

        self.threshold = threshold

    def probabilities_to_predictions(
        self,
        probabilities: np.ndarray,
    ) -> np.ndarray:
        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        if probabilities.ndim != 2:
            raise ValueError(
                "probabilities must have shape "
                "(samples, labels)"
            )

        return (
            probabilities >= self.threshold
        ).astype(np.int32)

    def logits_to_predictions(
        self,
        logits: np.ndarray,
    ) -> np.ndarray:
        logits = np.asarray(
            logits,
            dtype=np.float32,
        )

        if logits.ndim != 2:
            raise ValueError(
                "logits must have shape "
                "(samples, labels)"
            )

        probabilities = 1.0 / (
            1.0 + np.exp(-logits)
        )

        return self.probabilities_to_predictions(
            probabilities
        )

    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
    ) -> EvaluationResult:

        y_true = np.asarray(
            y_true,
            dtype=np.int32,
        )

        y_pred = np.asarray(
            y_pred,
            dtype=np.int32,
        )

        if y_true.ndim != 2:
            raise ValueError(
                "y_true must have shape "
                "(samples, labels)"
            )

        if y_pred.ndim != 2:
            raise ValueError(
                "y_pred must have shape "
                "(samples, labels)"
            )

        if y_true.shape != y_pred.shape:
            raise ValueError(
                "y_true and y_pred must have "
                "the same shape"
            )

        if y_true.shape[0] == 0:
            raise ValueError(
                "evaluation requires at least "
                "one sample"
            )

        samples, labels = y_true.shape

        return EvaluationResult(
            micro_precision=precision_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            ),
            micro_recall=recall_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            ),
            micro_f1=f1_score(
                y_true,
                y_pred,
                average="micro",
                zero_division=0,
            ),
            macro_precision=precision_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            macro_recall=recall_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            macro_f1=f1_score(
                y_true,
                y_pred,
                average="macro",
                zero_division=0,
            ),
            hamming_loss=hamming_loss(
                y_true,
                y_pred,
            ),
            threshold=self.threshold,
            samples=samples,
            labels=labels,
        )

@dataclass(frozen=True)
class DocumentPrediction:
    doc_id: str
    probabilities: np.ndarray

class DocumentAggregator:

    @staticmethod
    def aggregate_mean(
        doc_ids: Sequence[str],
        probabilities: np.ndarray,
    ) -> List[DocumentPrediction]:

        probabilities = np.asarray(
            probabilities,
            dtype=np.float32,
        )

        if probabilities.ndim != 2:
            raise ValueError(
                "probabilities must have shape "
                "(samples, labels)"
            )

        if len(doc_ids) != probabilities.shape[0]:
            raise ValueError(
                "doc_ids length must match "
                "number of probability samples"
            )

        grouped = {}

        for index, doc_id in enumerate(doc_ids):

            if doc_id not in grouped:
                grouped[doc_id] = []

            grouped[doc_id].append(
                probabilities[index]
            )

        results = []

        for doc_id in sorted(grouped):

            document_probabilities = np.mean(
                np.stack(grouped[doc_id]),
                axis=0,
            )

            results.append(
                DocumentPrediction(
                    doc_id=doc_id,
                    probabilities=document_probabilities,
                )
            )

        return results

def evaluate_document_level(
    evaluator: MultiLabelEvaluator,
    document_predictions: Sequence[
        DocumentPrediction
    ],
    document_labels: dict[str, Sequence[int]],
) -> EvaluationResult:

    if not document_predictions:
        raise ValueError(
            "document_predictions must not be empty"
        )

    y_true = []
    probabilities = []

    for prediction in document_predictions:

        if prediction.doc_id not in document_labels:
            raise ValueError(
                f"No labels found for document: "
                f"'{prediction.doc_id}'"
            )

        y_true.append(
            document_labels[prediction.doc_id]
        )

        probabilities.append(
            prediction.probabilities
        )

    y_true = np.asarray(
        y_true,
        dtype=np.int32,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float32,
    )

    y_pred = evaluator.probabilities_to_predictions(
        probabilities
    )

    return evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
    )