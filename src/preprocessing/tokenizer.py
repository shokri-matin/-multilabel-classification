import re
from abc import ABC, abstractmethod
from typing import List

from nltk.stem import PorterStemmer


class BaseTokenizer(ABC):
    """
    Base tokenizer interface.
    """

    @abstractmethod
    def tokenize(self, text: str) -> List[str]:
        """
        Convert raw text into tokens.
        """
        raise NotImplementedError


class WhitespaceTokenizer(BaseTokenizer):
    """
    Tokenizer for already-tokenized text.

    This tokenizer preserves the existing tokenization and
    simply splits the text on whitespace.
    """

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        return text.split()


class AAPDTokenizer(BaseTokenizer):
    """
    Tokenizer for raw AAPD documents.

    The tokenizer:
        1. Converts text to lowercase.
        2. Extracts word-like tokens.
        3. Preserves token order.
    """

    TOKEN_PATTERN = re.compile(
        r"[A-Za-z]+(?:'[A-Za-z]+)?"
    )

    def tokenize(self, text: str) -> List[str]:
        if not text:
            return []

        return self.TOKEN_PATTERN.findall(text)


class StopwordRemover:
    """
    Removes English stopwords from a token sequence.

    This component does not modify token positions.
    """

    DEFAULT_STOPWORDS = {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "here",
        "hers",
        "herself",
        "him",
        "himself",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "itself",
        "me",
        "more",
        "most",
        "my",
        "myself",
        "no",
        "nor",
        "not",
        "of",
        "on",
        "or",
        "our",
        "ours",
        "ourselves",
        "she",
        "so",
        "some",
        "such",
        "than",
        "that",
        "the",
        "their",
        "theirs",
        "them",
        "themselves",
        "then",
        "there",
        "these",
        "they",
        "this",
        "those",
        "through",
        "to",
        "too",
        "under",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "while",
        "who",
        "whom",
        "why",
        "will",
        "with",
        "you",
        "your",
        "yours",
        "yourself",
        "yourselves",
    }

    def __init__(self, stopwords: set[str] | None = None):
        self.stopwords = (
            stopwords
            if stopwords is not None
            else self.DEFAULT_STOPWORDS
        )

    def remove(
        self,
        tokens: List[str],
    ) -> List[str]:
        """
        Remove stopwords while preserving token order.
        """

        return [
            token
            for token in tokens
            if token not in self.stopwords
        ]


class Stemmer:
    """
    English Porter stemmer.

    Stemming is applied after tokenization and stopword removal.
    """

    def __init__(self):
        self._stemmer = PorterStemmer()

    def stem(
        self,
        tokens: List[str],
    ) -> List[str]:
        """
        Stem all tokens while preserving their order.
        """

        return [
            self._stemmer.stem(token)
            for token in tokens
        ]

