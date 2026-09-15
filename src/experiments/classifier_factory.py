from src.classifier.bert import (
    BERTClassifier,
    BERTClassifierConfig,
)
from src.classifier.xlnet import (
    XLNetClassifier,
    XLNetClassifierConfig,
)


class ClassifierFactory:
    @staticmethod
    def create(
        classifier: str,
        input_dim: int,
        num_labels: int,
        dropout: float = 0.1,
    ):
        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive"
            )

        if num_labels <= 0:
            raise ValueError(
                "num_labels must be positive"
            )

        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        if classifier == "bert":
            config = BERTClassifierConfig(
                num_labels=num_labels,
                hidden_size=768,
                dropout=dropout,
            )

            return BERTClassifier(
                input_dim=input_dim,
                config=config,
            )

        if classifier == "xlnet":
            config = XLNetClassifierConfig(
                num_labels=num_labels,
                hidden_size=768,
                dropout=dropout,
            )

            return XLNetClassifier(
                input_dim=input_dim,
                config=config,
            )

        raise ValueError(
            f"Unsupported classifier: {classifier}"
        )