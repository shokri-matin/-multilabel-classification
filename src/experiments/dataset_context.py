from dataclasses import dataclass
from typing import Dict, Sequence

from src.classifier.label_vocabulary import (
    LabelVocabulary,
    LabelVocabularyBuilder,
)
from src.corpus.contracts import Corpus


@dataclass
class DatasetContext:
    """
    Shared dataset-level context reused by all experiments
    belonging to the same dataset.

    The context contains:
    - processed train corpus
    - processed validation corpus
    - processed test corpus
    - label vocabulary built ONLY from training labels
    - document-level labels for each split
    """

    dataset: str
    train_corpus: Corpus
    validation_corpus: Corpus
    test_corpus: Corpus
    label_vocabulary: LabelVocabulary
    train_document_labels: Dict[str, Sequence[str]]
    validation_document_labels: Dict[str, Sequence[str]]
    test_document_labels: Dict[str, Sequence[str]]

    def __post_init__(self):
        if not self.dataset:
            raise ValueError(
                "dataset must not be empty"
            )

        if not isinstance(
            self.train_corpus,
            Corpus,
        ):
            raise TypeError(
                "train_corpus must be a Corpus"
            )

        if not isinstance(
            self.validation_corpus,
            Corpus,
        ):
            raise TypeError(
                "validation_corpus must be a Corpus"
            )

        if not isinstance(
            self.test_corpus,
            Corpus,
        ):
            raise TypeError(
                "test_corpus must be a Corpus"
            )

        if not isinstance(
            self.label_vocabulary,
            LabelVocabulary,
        ):
            raise TypeError(
                "label_vocabulary must be a LabelVocabulary"
            )

    @property
    def num_labels(self) -> int:
        return self.label_vocabulary.num_labels

    @property
    def train_size(self) -> int:
        return self.train_corpus.size

    @property
    def validation_size(self) -> int:
        return self.validation_corpus.size

    @property
    def test_size(self) -> int:
        return self.test_corpus.size

    def labels_for_split(
        self,
        split: str,
    ) -> Dict[str, Sequence[str]]:
        if split == "train":
            return self.train_document_labels

        if split == "validation":
            return self.validation_document_labels

        if split == "test":
            return self.test_document_labels

        raise ValueError(
            f"Unsupported split: {split}"
        )


class DatasetContextBuilder:
    """
    Builds DatasetContext from already processed corpora.

    Label vocabulary is ALWAYS built from train corpus only.
    """

    SUPPORTED_DATASETS = {
        "aapd",
        "rcv1",
    }

    def build(
        self,
        dataset: str,
        train_corpus: Corpus,
        validation_corpus: Corpus,
        test_corpus: Corpus,
    ) -> DatasetContext:

        if dataset not in self.SUPPORTED_DATASETS:
            raise ValueError(
                f"Unsupported dataset: {dataset}"
            )

        self._validate_corpus_split(
            train_corpus,
            expected_split="train",
        )

        self._validate_corpus_split(
            validation_corpus,
            expected_split="validation",
        )

        self._validate_corpus_split(
            test_corpus,
            expected_split="test",
        )

        label_vocabulary = (
            LabelVocabularyBuilder().build(
                train_corpus.documents
            )
        )

        train_document_labels = (
            self._build_document_labels(
                train_corpus
            )
        )

        validation_document_labels = (
            self._build_document_labels(
                validation_corpus
            )
        )

        test_document_labels = (
            self._build_document_labels(
                test_corpus
            )
        )

        return DatasetContext(
            dataset=dataset,
            train_corpus=train_corpus,
            validation_corpus=validation_corpus,
            test_corpus=test_corpus,
            label_vocabulary=label_vocabulary,
            train_document_labels=train_document_labels,
            validation_document_labels=(
                validation_document_labels
            ),
            test_document_labels=(
                test_document_labels
            ),
        )

    @staticmethod
    def _validate_corpus_split(
        corpus: Corpus,
        expected_split: str,
    ) -> None:

        if not isinstance(corpus, Corpus):
            raise TypeError(
                "corpus must be a Corpus"
            )

        if corpus.split != expected_split:
            raise ValueError(
                f"Expected corpus split "
                f"'{expected_split}', "
                f"got '{corpus.split}'"
            )

    @staticmethod
    def _build_document_labels(
        corpus: Corpus,
    ) -> Dict[str, Sequence[str]]:

        return {
            document.doc_id: list(
                document.labels
            )
            for document in corpus.documents
        }