import pytest

from src.corpus.contracts import Corpus
from src.preprocessing.contracts import (
    ProcessedDocument,
    ProcessedToken,
)
from src.experiments.dataset_context import (
    DatasetContextBuilder,
)


def make_document(
    doc_id: str,
    labels: list[str],
) -> ProcessedDocument:

    return ProcessedDocument(
        doc_id=doc_id,
        text=f"text for {doc_id}",
        labels=labels,
        tokens=[
            ProcessedToken(
                text="text",
                position=0,
            ),
            ProcessedToken(
                text="document",
                position=1,
            ),
        ],
    )


def make_corpus(
    name: str,
    split: str,
    documents: list[ProcessedDocument],
) -> Corpus:

    return Corpus(
        name=name,
        split=split,
        documents=documents,
    )


def make_context():
    train = make_corpus(
        "aapd",
        "train",
        [
            make_document(
                "train_1",
                ["A", "B"],
            ),
            make_document(
                "train_2",
                ["B", "C"],
            ),
        ],
    )

    validation = make_corpus(
        "aapd",
        "validation",
        [
            make_document(
                "validation_1",
                ["A", "C"],
            ),
        ],
    )

    test = make_corpus(
        "aapd",
        "test",
        [
            make_document(
                "test_1",
                ["B"],
            ),
        ],
    )

    return DatasetContextBuilder().build(
        dataset="aapd",
        train_corpus=train,
        validation_corpus=validation,
        test_corpus=test,
    )


def test_build_context():
    context = make_context()

    assert context.dataset == "aapd"

    assert context.train_size == 2
    assert context.validation_size == 1
    assert context.test_size == 1


def test_label_vocabulary_comes_from_train_only():
    context = make_context()

    assert context.label_vocabulary.labels == [
        "A",
        "B",
        "C",
    ]


def test_num_labels():
    context = make_context()

    assert context.num_labels == 3


def test_train_document_labels():
    context = make_context()

    assert context.train_document_labels == {
        "train_1": ["A", "B"],
        "train_2": ["B", "C"],
    }


def test_validation_document_labels():
    context = make_context()

    assert context.validation_document_labels == {
        "validation_1": ["A", "C"],
    }


def test_test_document_labels():
    context = make_context()

    assert context.test_document_labels == {
        "test_1": ["B"],
    }


def test_labels_for_split():
    context = make_context()

    assert context.labels_for_split(
        "train"
    ) == context.train_document_labels

    assert context.labels_for_split(
        "validation"
    ) == context.validation_document_labels

    assert context.labels_for_split(
        "test"
    ) == context.test_document_labels


def test_invalid_split():
    context = make_context()

    with pytest.raises(ValueError):
        context.labels_for_split("unknown")


def test_invalid_dataset():
    train = make_corpus(
        "aapd",
        "train",
        [
            make_document(
                "train_1",
                ["A"],
            )
        ],
    )

    validation = make_corpus(
        "aapd",
        "validation",
        [
            make_document(
                "validation_1",
                ["A"],
            )
        ],
    )

    test = make_corpus(
        "aapd",
        "test",
        [
            make_document(
                "test_1",
                ["A"],
            )
        ],
    )

    with pytest.raises(ValueError):
        DatasetContextBuilder().build(
            dataset="unknown",
            train_corpus=train,
            validation_corpus=validation,
            test_corpus=test,
        )


def test_invalid_train_split():
    train = make_corpus(
        "aapd",
        "validation",
        [
            make_document(
                "train_1",
                ["A"],
            )
        ],
    )

    validation = make_corpus(
        "aapd",
        "validation",
        [
            make_document(
                "validation_1",
                ["A"],
            )
        ],
    )

    test = make_corpus(
        "aapd",
        "test",
        [
            make_document(
                "test_1",
                ["A"],
            )
        ],
    )

    with pytest.raises(ValueError):
        DatasetContextBuilder().build(
            dataset="aapd",
            train_corpus=train,
            validation_corpus=validation,
            test_corpus=test,
        )


def test_invalid_validation_split():
    train = make_corpus(
        "aapd",
        "train",
        [
            make_document(
                "train_1",
                ["A"],
            )
        ],
    )

    validation = make_corpus(
        "aapd",
        "test",
        [
            make_document(
                "validation_1",
                ["A"],
            )
        ],
    )

    test = make_corpus(
        "aapd",
        "test",
        [
            make_document(
                "test_1",
                ["A"],
            )
        ],
    )

    with pytest.raises(ValueError):
        DatasetContextBuilder().build(
            dataset="aapd",
            train_corpus=train,
            validation_corpus=validation,
            test_corpus=test,
        )


def test_invalid_test_split():
    train = make_corpus(
        "aapd",
        "train",
        [
            make_document(
                "train_1",
                ["A"],
            )
        ],
    )

    validation = make_corpus(
        "aapd",
        "validation",
        [
            make_document(
                "validation_1",
                ["A"],
            )
        ],
    )

    test = make_corpus(
        "aapd",
        "validation",
        [
            make_document(
                "test_1",
                ["A"],
            )
        ],
    )

    with pytest.raises(ValueError):
        DatasetContextBuilder().build(
            dataset="aapd",
            train_corpus=train,
            validation_corpus=validation,
            test_corpus=test,
        )