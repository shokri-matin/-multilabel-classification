from dataclasses import dataclass

import torch
from torch import nn
from transformers import AutoConfig, AutoModel

from src.classifier.adapter import RepresentationAdapter
from src.classifier.base import BaseClassifier
from src.classifier.config import ClassifierConfig


@dataclass(frozen=True)
class XLNetClassifierConfig(ClassifierConfig):
    """
    XLNet classifier configuration.
    """

    model_name: str = "xlnet-base-cased"

    def __post_init__(self):
        super().__post_init__()

        if not self.model_name:
            raise ValueError(
                "model_name must not be empty"
            )


class XLNetClassifier(BaseClassifier):
    """
    XLNet-based multi-label classifier.

    External word embeddings are supplied through
    `inputs_embeds`.

    Input:
        (batch_size, sequence_length, input_dim)

    Output:
        (batch_size, num_labels)
    """

    def __init__(
        self,
        input_dim: int,
        config: XLNetClassifierConfig,
    ):
        super().__init__(
            input_dim=input_dim,
            config=config,
        )

        self.xlnet_config = AutoConfig.from_pretrained(
            config.model_name
        )

        model_hidden_size = (
            self.xlnet_config.d_model
        )

        if model_hidden_size != config.hidden_size:
            raise ValueError(
                f"XLNet hidden size is {model_hidden_size}, "
                f"but config specifies {config.hidden_size}"
            )

        self.adapter = RepresentationAdapter(
            input_dim=input_dim,
            output_dim=model_hidden_size,
        )

        self.xlnet = AutoModel.from_pretrained(
            config.model_name
        )

        self.dropout = nn.Dropout(
            config.dropout
        )

        self.classifier = nn.Linear(
            model_hidden_size,
            config.num_labels,
        )

    @property
    def hidden_size(self) -> int:
        return self.xlnet_config.d_model

    def forward(
        self,
        inputs_embeds: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:

        if inputs_embeds.ndim != 3:
            raise ValueError(
                "inputs_embeds must have shape "
                "(batch_size, sequence_length, input_dim)"
            )

        if inputs_embeds.shape[-1] != self.input_dim:
            raise ValueError(
                f"Expected input dimension "
                f"{self.input_dim}, "
                f"got {inputs_embeds.shape[-1]}"
            )

        projected_embeddings = self.adapter(
            inputs_embeds
        )

        outputs = self.xlnet(
            inputs_embeds=projected_embeddings,
            attention_mask=attention_mask,
        )

        hidden_states = outputs.last_hidden_state

        if attention_mask is None:

            pooled = hidden_states.mean(
                dim=1
            )

        else:

            mask = attention_mask.unsqueeze(-1).to(
                hidden_states.dtype
            )

            masked_hidden_states = (
                hidden_states * mask
            )

            pooled = (
                masked_hidden_states.sum(dim=1)
                / mask.sum(dim=1).clamp(min=1.0)
            )

        pooled = self.dropout(
            pooled
        )

        logits = self.classifier(
            pooled
        )

        return logits