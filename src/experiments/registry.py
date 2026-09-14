from typing import List

from src.experiments.config import (
    CENTRALITY_STRATEGIES,
    ExperimentConfig,
)


class ExperimentRegistry:

    DATASETS = [
        "aapd",
        "rcv1",
    ]

    MODES = [
        "branch",
        "no_branch",
    ]

    EMBEDDINGS = [
        "bert",
        "fasttext",
    ]

    CLASSIFIERS = [
        "bert",
        "xlnet",
    ]

    @classmethod
    def build_all(
        cls,
        top_k_by_dataset: dict[str, int],
        max_nodes: int,
        sequence_length: int,
    ) -> List[ExperimentConfig]:

        experiments = []

        for dataset in cls.DATASETS:

            top_k = top_k_by_dataset[dataset]

            for mode in cls.MODES:

                for centrality in CENTRALITY_STRATEGIES:

                    for embedding in cls.EMBEDDINGS:

                        for classifier in cls.CLASSIFIERS:

                            experiments.append(
                                ExperimentConfig(
                                    dataset=dataset,
                                    mode=mode,
                                    centrality=centrality,
                                    embedding=embedding,
                                    classifier=classifier,
                                    top_k=top_k,
                                    max_nodes=max_nodes,
                                    sequence_length=sequence_length,
                                )
                            )

        return experiments