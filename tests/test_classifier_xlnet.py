import torch
import pytest

from src.classifier.xlnet import (
    XLNetClassifier,
    XLNetClassifierConfig,
)


MODEL_NAME = "xlnet-base-cased"


def test_xlnet_classifier_config():

    config = XLNetClassifierConfig(
        num_labels=10,
        hidden_size=768,
    )

    assert config.model_name == MODEL_NAME
    assert config.num_labels == 10
    assert config.hidden_size == 768


def test_xlnet_classifier_invalid_hidden_size():

    config = XLNetClassifierConfig(
        num_labels=10,
        hidden_size=300,
    )

    with pytest.raises(ValueError):

        XLNetClassifier(
            input_dim=300,
            config=config,
        )


def test_xlnet_classifier_fasttext_input():

    config = XLNetClassifierConfig(
        num_labels=10,
        hidden_size=768,
    )

    model = XLNetClassifier(
        input_dim=300,
        config=config,
    )

    inputs = torch.randn(
        2,
        25,
        300,
    )

    attention_mask = torch.ones(
        2,
        25,
        dtype=torch.long,
    )

    logits = model(
        inputs_embeds=inputs,
        attention_mask=attention_mask,
    )

    assert logits.shape == (
        2,
        10,
    )


def test_xlnet_classifier_bert_input():

    config = XLNetClassifierConfig(
        num_labels=10,
        hidden_size=768,
    )

    model = XLNetClassifier(
        input_dim=768,
        config=config,
    )

    inputs = torch.randn(
        2,
        25,
        768,
    )

    attention_mask = torch.ones(
        2,
        25,
        dtype=torch.long,
    )

    logits = model(
        inputs_embeds=inputs,
        attention_mask=attention_mask,
    )

    assert logits.shape == (
        2,
        10,
    )


def test_xlnet_classifier_probability_range():

    config = XLNetClassifierConfig(
        num_labels=10,
        hidden_size=768,
    )

    model = XLNetClassifier(
        input_dim=300,
        config=config,
    )

    inputs = torch.randn(
        2,
        25,
        300,
    )

    attention_mask = torch.ones(
        2,
        25,
        dtype=torch.long,
    )

    probabilities = model.predict_proba(
        inputs_embeds=inputs,
        attention_mask=attention_mask,
    )

    assert probabilities.shape == (
        2,
        10,
    )

    assert torch.all(
        probabilities >= 0
    )

    assert torch.all(
        probabilities <= 1
    )


def test_xlnet_classifier_with_padding():

    config = XLNetClassifierConfig(
        num_labels=10,
        hidden_size=768,
    )

    model = XLNetClassifier(
        input_dim=300,
        config=config,
    )

    inputs = torch.randn(
        2,
        25,
        300,
    )

    attention_mask = torch.ones(
        2,
        25,
        dtype=torch.long,
    )

    attention_mask[:, 20:] = 0

    logits = model(
        inputs_embeds=inputs,
        attention_mask=attention_mask,
    )

    assert logits.shape == (
        2,
        10,
    )