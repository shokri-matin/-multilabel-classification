import numpy as np
import pytest

from src.evaluation.multilabel import (
    DocumentAggregator,
    MultiLabelEvaluator,
    evaluate_document_level,
)


def test_probability_threshold():

    evaluator = MultiLabelEvaluator(
        threshold=0.5
    )

    probabilities = np.array(
        [
            [0.9, 0.2, 0.5],
            [0.4, 0.8, 0.1],
        ]
    )

    predictions = (
        evaluator.probabilities_to_predictions(
            probabilities
        )
    )

    expected = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
        ]
    )

    assert np.array_equal(
        predictions,
        expected,
    )


def test_logits_to_predictions():

    evaluator = MultiLabelEvaluator(
        threshold=0.5
    )

    logits = np.array(
        [
            [10.0, -10.0],
            [-10.0, 10.0],
        ]
    )

    predictions = (
        evaluator.logits_to_predictions(
            logits
        )
    )

    expected = np.array(
        [
            [1, 0],
            [0, 1],
        ]
    )

    assert np.array_equal(
        predictions,
        expected,
    )


def test_evaluation_perfect_prediction():

    evaluator = MultiLabelEvaluator()

    y_true = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
        ]
    )

    y_pred = y_true.copy()

    result = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert result.micro_precision == 1.0
    assert result.micro_recall == 1.0
    assert result.micro_f1 == 1.0

    assert result.macro_precision == 1.0
    assert result.macro_recall == 1.0
    assert result.macro_f1 == 1.0

    assert result.hamming_loss == 0.0

    assert result.samples == 2
    assert result.labels == 3


def test_evaluation_non_perfect_prediction():

    evaluator = MultiLabelEvaluator()

    y_true = np.array(
        [
            [1, 0, 1],
            [0, 1, 0],
        ]
    )

    y_pred = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
        ]
    )

    result = evaluator.evaluate(
        y_true=y_true,
        y_pred=y_pred,
    )

    assert 0.0 <= result.micro_f1 <= 1.0
    assert 0.0 <= result.macro_f1 <= 1.0
    assert 0.0 <= result.hamming_loss <= 1.0


def test_document_mean_aggregation():

    doc_ids = [
        "doc_1",
        "doc_1",
        "doc_2",
    ]

    probabilities = np.array(
        [
            [0.8, 0.2],
            [0.6, 0.4],
            [0.1, 0.9],
        ]
    )

    results = DocumentAggregator.aggregate_mean(
        doc_ids=doc_ids,
        probabilities=probabilities,
    )

    assert len(results) == 2

    assert results[0].doc_id == "doc_1"
    assert results[1].doc_id == "doc_2"

    assert np.allclose(
        results[0].probabilities,
        [0.7, 0.3],
    )

    assert np.allclose(
        results[1].probabilities,
        [0.1, 0.9],
    )


def test_document_level_evaluation():

    evaluator = MultiLabelEvaluator(
        threshold=0.5
    )

    doc_ids = [
        "doc_1",
        "doc_1",
        "doc_2",
    ]

    probabilities = np.array(
        [
            [0.9, 0.1],
            [0.7, 0.3],
            [0.1, 0.9],
        ]
    )

    document_predictions = (
        DocumentAggregator.aggregate_mean(
            doc_ids=doc_ids,
            probabilities=probabilities,
        )
    )

    document_labels = {
        "doc_1": [1, 0],
        "doc_2": [0, 1],
    }

    result = evaluate_document_level(
        evaluator=evaluator,
        document_predictions=document_predictions,
        document_labels=document_labels,
    )

    assert result.micro_f1 == 1.0
    assert result.macro_f1 == 1.0


def test_invalid_threshold():

    with pytest.raises(ValueError):
        MultiLabelEvaluator(
            threshold=0.0
        )

    with pytest.raises(ValueError):
        MultiLabelEvaluator(
            threshold=1.0
        )


def test_invalid_probability_shape():

    evaluator = MultiLabelEvaluator()

    with pytest.raises(ValueError):
        evaluator.probabilities_to_predictions(
            np.array([0.5, 0.6])
        )


def test_invalid_evaluation_shape():

    evaluator = MultiLabelEvaluator()

    y_true = np.array(
        [[1, 0, 1]]
    )

    y_pred = np.array(
        [[1, 0]]
    )

    with pytest.raises(ValueError):
        evaluator.evaluate(
            y_true=y_true,
            y_pred=y_pred,
        )


def test_empty_evaluation():

    evaluator = MultiLabelEvaluator()

    with pytest.raises(ValueError):
        evaluator.evaluate(
            y_true=np.empty((0, 3)),
            y_pred=np.empty((0, 3)),
        )