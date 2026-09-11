from dataclasses import dataclass


@dataclass(frozen=True)
class EmbeddingConfig:
    """
    Common configuration shared by embedding implementations.
    """

    sequence_length: int = 25

    normalize: bool = False

    batch_size: int = 8

    seed: int = 42

    def __post_init__(self):
        if self.sequence_length <= 0:
            raise ValueError(
                "sequence_length must be positive"
            )

        if self.batch_size <= 0:
            raise ValueError(
                "batch_size must be positive"
            )