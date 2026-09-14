from dataclasses import dataclass

import pytest

from src.classifier.label_vocabulary import (
    LabelVocabulary,
    LabelVocabularyBuilder,
)


@dataclass
class FakeDocument:
    labels: list[str]


def test_build_vocabulary():
    documents = [
        FakeDocument(["CCAT", "GCAT"]),
        FakeDocument(["MCAT"]),
        FakeDocument(["CCAT", "ECAT"]),
    ]

    builder = LabelVocabularyBuilder()

    vocabulary = builder.build(documents)

    assert vocabulary.labels == [
        "CCAT",
        "ECAT",
        "GCAT",
        "MCAT",
    ]

    assert vocabulary.num_labels == 4


def test_label_to_index_is_deterministic():
    documents = [
        FakeDocument(["MCAT", "CCAT"]),
        FakeDocument(["GCAT", "ECAT"]),
    ]

    vocabulary = LabelVocabularyBuilder().build(
        documents
    )

    assert vocabulary.label_to_index == {
        "CCAT": 0,
        "ECAT": 1,
        "GCAT": 2,
        "MCAT": 3,
    }


def test_encode():
    documents = [
        FakeDocument(["CCAT", "GCAT", "MCAT"])
    ]

    vocabulary = LabelVocabularyBuilder().build(
        documents
    )

    encoded = vocabulary.encode(
        ["CCAT", "MCAT"]
    )

    assert encoded == [1, 0, 1]


def test_decode():
    documents = [
        FakeDocument(["CCAT", "GCAT", "MCAT"])
    ]

    vocabulary = LabelVocabularyBuilder().build(
        documents
    )

    decoded = vocabulary.decode(
        [1, 0, 1]
    )

    assert decoded == [
        "CCAT",
        "MCAT",
    ]


def test_unknown_label_raises_error():
    documents = [
        FakeDocument(["CCAT", "GCAT"])
    ]

    vocabulary = LabelVocabularyBuilder().build(
        documents
    )

    with pytest.raises(
        ValueError,
        match="Unknown label",
    ):
        vocabulary.encode(["UNKNOWN"])


def test_invalid_encoded_length_raises_error():
    documents = [
        FakeDocument(["CCAT", "GCAT"])
    ]

    vocabulary = LabelVocabularyBuilder().build(
        documents
    )

    with pytest.raises(
        ValueError,
        match="Expected 2 values",
    ):
        vocabulary.decode([1])


def test_empty_documents_raise_error():
    with pytest.raises(
        ValueError,
        match="No labels found",
    ):
        LabelVocabularyBuilder().build([])


def test_duplicate_labels_are_removed():
    documents = [
        FakeDocument(["CCAT", "CCAT", "GCAT"]),
        FakeDocument(["GCAT", "CCAT"]),
    ]

    vocabulary = LabelVocabularyBuilder().build(
        documents
    )

    assert vocabulary.labels == [
        "CCAT",
        "GCAT",
    ]

    assert vocabulary.num_labels == 2


def test_train_vocabulary_does_not_include_validation_labels():
    train_documents = [
        FakeDocument(["CCAT", "GCAT"]),
        FakeDocument(["MCAT"]),
    ]

    validation_documents = [
        FakeDocument(["CCAT", "NEW_LABEL"]),
    ]

    train_vocabulary = (
        LabelVocabularyBuilder().build(
            train_documents
        )
    )

    assert "NEW_LABEL" not in train_vocabulary.labels

    with pytest.raises(
        ValueError,
        match="Unknown label",
    ):
        train_vocabulary.encode(
            validation_documents[0].labels
        )