from dataclasses import dataclass


@dataclass(frozen=True)
class ClassifierConfig:
    """
    Base configuration shared by Transformer classifiers.
    """

    num_labels: int

    hidden_size: int

    dropout: float = 0.1

    max_length: int = 25

    learning_rate: float = 2e-5

    batch_size: int = 8

    epochs: int = 5

    seed: int = 42

    def __post_init__(self):
        if self.num_labels <= 0:
            raise ValueError(
                "num_labels must be positive"
            )

        if self.hidden_size <= 0:
            raise ValueError(
                "hidden_size must be positive"
            )

        if not 0.0 <= self.dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1)"
            )

        if self.max_length <= 0:
            raise ValueError(
                "max_length must be positive"
            )

        if self.learning_rate <= 0:
            raise ValueError(
                "learning_rate must be positive"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )

        if self.epochs <= 0:
            raise ValueError(
                "epochs must be positive"
            )