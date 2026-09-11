from abc import ABC, abstractmethod

import torch
from torch import nn


class BaseClassifier(ABC, nn.Module):
    """
    Base interface for multi-label Transformer classifiers.
    """

    def __init__(
        self,
        input_dim: int,
        config,
    ):
        super().__init__()

        if input_dim <= 0:
            raise ValueError(
                "input_dim must be positive"
            )

        self.input_dim = input_dim
        self.config = config

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """
        Hidden dimension expected by the classifier.
        """
        ...

    @abstractmethod
    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Return logits for multi-label classification.

        Expected input:
            inputs_embeds: (batch_size, sequence_length, input_dim)

        Output:
            logits: (batch_size, num_labels)
        """
        ...

    def predict_proba(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Convert logits to independent label probabilities.
        """

        logits = self.forward(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
        )

        return torch.sigmoid(logits)