import torch
import pytest

from src.classifier.adapter import (
    RepresentationAdapter,
)


def test_same_dimension_uses_identity():

    adapter = RepresentationAdapter(
        input_dim=768,
        output_dim=768,
    )

    embeddings = torch.randn(
        4,
        25,
        768,
    )

    output = adapter(embeddings)

    assert output.shape == (
        4,
        25,
        768,
    )

    assert isinstance(
        adapter.projection,
        torch.nn.Identity,
    )


def test_fasttext_projection():

    adapter = RepresentationAdapter(
        input_dim=300,
        output_dim=768,
    )

    embeddings = torch.randn(
        4,
        25,
        300,
    )

    output = adapter(embeddings)

    assert output.shape == (
        4,
        25,
        768,
    )

    assert isinstance(
        adapter.projection,
        torch.nn.Linear,
    )


def test_invalid_input_rank():

    adapter = RepresentationAdapter(
        input_dim=300,
        output_dim=768,
    )

    embeddings = torch.randn(
        4,
        300,
    )

    with pytest.raises(ValueError):
        adapter(embeddings)


def test_invalid_input_dimension():

    adapter = RepresentationAdapter(
        input_dim=300,
        output_dim=768,
    )

    embeddings = torch.randn(
        4,
        25,
        768,
    )

    with pytest.raises(ValueError):
        adapter(embeddings)