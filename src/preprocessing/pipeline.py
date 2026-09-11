from dataclasses import dataclass
from typing import List, Optional
from tqdm import tqdm

from src.data.contracts import RawDocument
from src.preprocessing.contracts import (
    ProcessedDocument,
    ProcessedToken,
)

from src.preprocessing.tokenizer import (
    BaseTokenizer,
    Stemmer,
    StopwordRemover,
)


@dataclass(frozen=True)
class PreprocessingConfig:
    """
    Configuration for preprocessing.
    """

    lowercase: bool = False
    remove_stopwords: bool = False
    stemming: bool = False


class PreprocessingPipeline:
    """
    Converts RawDocument objects into ProcessedDocument objects.

    Important:
        Original token positions are preserved even when
        stopwords are removed.
    """

    def __init__(
        self,
        tokenizer: BaseTokenizer,
        config: Optional[PreprocessingConfig] = None,
    ):
        self.tokenizer = tokenizer

        self.config = (
            config
            if config is not None
            else PreprocessingConfig()
        )

        self.stopword_remover = StopwordRemover()
        self.stemmer = Stemmer()

    def process(
        self,
        document: RawDocument,
    ) -> ProcessedDocument:
        """
        Process a single document.
        """

        tokens = self.tokenizer.tokenize(document.text)

        # Store original positions BEFORE filtering.
        indexed_tokens = [
            (token, position)
            for position, token in enumerate(tokens)
        ]

        # Optional lowercase transformation.
        if self.config.lowercase:
            indexed_tokens = [
                (token.lower(), position)
                for token, position in indexed_tokens
            ]

        # Stopword removal.
        if self.config.remove_stopwords:
            indexed_tokens = [
                (token, position)
                for token, position in indexed_tokens
                if token not in self.stopword_remover.stopwords
            ]

        # Stemming.
        if self.config.stemming:
            indexed_tokens = [
                (
                    self.stemmer.stem([token])[0],
                    position,
                )
                for token, position in indexed_tokens
            ]

        processed_tokens = [
            ProcessedToken(
                text=token,
                position=position,
            )
            for token, position in indexed_tokens
        ]

        return ProcessedDocument(
            doc_id=document.doc_id,
            text=document.text,
            labels=list(document.labels),
            tokens=processed_tokens,
        )

    def process_many(
        self,
        documents: List[RawDocument],
        show_progress=True

    ) -> List[ProcessedDocument]:
        """
        Process multiple documents.
        """

        iterator = documents

        if show_progress:
            iterator = tqdm(
                documents,
                desc="Preprocessing documents",
                unit="doc"
            )

        return [
            self.process(document)
            for document in iterator
        ]
