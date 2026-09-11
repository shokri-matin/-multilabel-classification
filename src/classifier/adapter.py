import torch
from torch import nn


class RepresentationAdapter(nn.Module):
    """
    Projects external word embeddings into the hidden dimension
    required by a Transformer classifier.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive"
            )

        if output_dim <= 0:
            raise ValueError(
                "output_dim must be positive"
            )

        self.input_dim = input_dim
        self.output_dim = output_dim

        if input_dim == output_dim:
            self.projection = nn.Identity()
        else:
            self.projection = nn.Linear(
                input_dim,
                output_dim,
            )

    def forward(
        self,
        embeddings: torch.Tensor,
    ) -> torch.Tensor:
        """
        Project embeddings.

        Input:
            (batch_size, sequence_length, input_dim)

        Output:
            (batch_size, sequence_length, output_dim)
        """

        if embeddings.ndim != 3:
            raise ValueError(
                "embeddings must have shape "
                "(batch_size, sequence_length, input_dim)"
            )

        if embeddings.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension "
                f"{self.input_dim}, "
                f"got {embeddings.shape[-1]}"
            )

        return self.projection(
            embeddings
        )