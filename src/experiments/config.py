from dataclasses import dataclass
from typing import Literal


DatasetName = Literal["aapd", "rcv1"]
ExperimentMode = Literal["branch", "no_branch"]
EmbeddingName = Literal["bert", "fasttext"]
ClassifierName = Literal["bert", "xlnet"]


CENTRALITY_STRATEGIES = [
    "closeness",
    "degree",
    "betweenness",
    "pagerank",
    "clustering",
    "closeness*degree*clustering",
]


@dataclass(frozen=True)
class ExperimentConfig:
    dataset: DatasetName
    mode: ExperimentMode
    centrality: str
    embedding: EmbeddingName
    classifier: ClassifierName

    top_k: int
    max_nodes: int
    sequence_length: int

    def __post_init__(self):
        if self.dataset not in {
            "aapd",
            "rcv1",
        }:
            raise ValueError(
                f"Unsupported dataset: {self.dataset}"
            )

        if self.mode not in {
            "branch",
            "no_branch",
        }:
            raise ValueError(
                f"Unsupported mode: {self.mode}"
            )

        if self.centrality not in CENTRALITY_STRATEGIES:
            raise ValueError(
                f"Unsupported centrality: "
                f"{self.centrality}"
            )

        if self.embedding not in {
            "bert",
            "fasttext",
        }:
            raise ValueError(
                f"Unsupported embedding: "
                f"{self.embedding}"
            )

        if self.classifier not in {
            "bert",
            "xlnet",
        }:
            raise ValueError(
                f"Unsupported classifier: "
                f"{self.classifier}"
            )

        if self.top_k <= 0:
            raise ValueError(
                "top_k must be positive"
            )

        if self.max_nodes <= 0:
            raise ValueError(
                "max_nodes must be positive"
            )

        if self.sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

    @property
    def experiment_id(self) -> str:
        centrality_id = self.centrality.replace(
            "*",
            "_",
        )

        return (
            f"{self.dataset}"
            f"__{self.mode}"
            f"__{centrality_id}"
            f"__{self.embedding}"
            f"__{self.classifier}"
        )

from dataclasses import dataclass
from typing import Optional


@dataclass
class ExperimentResult:
    experiment_id: str

    dataset: str
    mode: str
    centrality: str
    embedding: str
    classifier: str

    train_loss: Optional[float]
    validation_loss: Optional[float]

    micro_precision: Optional[float]
    micro_recall: Optional[float]
    micro_f1: Optional[float]

    macro_precision: Optional[float]
    macro_recall: Optional[float]
    macro_f1: Optional[float]

    hamming_loss: Optional[float]

    best_epoch: Optional[int]
    training_epochs: int

    checkpoint_path: Optional[str]

    status: str
    error: Optional[str] = None