from typing import List

from src.corpus.contracts import Corpus
from src.data.contracts import RawDocument
from src.preprocessing.pipeline import PreprocessingPipeline


class CorpusBuilder:
    def __init__(
        self,
        name: str,
        loader,
        preprocessing_pipeline: PreprocessingPipeline,
    ):
        self.name = name
        self.loader = loader
        self.preprocessing_pipeline = preprocessing_pipeline

    def build(self, split: str) -> Corpus:
        raw_documents: List[RawDocument] = self.loader.load_split(split)

        processed_documents = self.preprocessing_pipeline.process_many(
            raw_documents,
            show_progress=True,
        )

        return Corpus(
            name=self.name,
            split=split,
            documents=processed_documents,
        )

    def build_all(self) -> dict[str, Corpus]:
        splits = ("train", "validation", "test")

        return {
            split: self.build(split)
            for split in splits
        }

