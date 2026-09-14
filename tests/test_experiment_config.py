import pytest

from src.experiments.config import (
    CENTRALITY_STRATEGIES,
    ExperimentConfig,
)
from src.experiments.registry import (
    ExperimentRegistry,
)


def make_config():
    return ExperimentConfig(
        dataset="aapd",
        mode="branch",
        centrality="pagerank",
        embedding="fasttext",
        classifier="xlnet",
        top_k=200,
        max_nodes=25,
        sequence_length=25,
    )


def test_experiment_config():

    config = make_config()

    assert config.dataset == "aapd"
    assert config.mode == "branch"
    assert config.centrality == "pagerank"
    assert config.embedding == "fasttext"
    assert config.classifier == "xlnet"

    assert config.top_k == 200
    assert config.max_nodes == 25
    assert config.sequence_length == 25


def test_experiment_id():

    config = make_config()

    assert (
        config.experiment_id
        == "aapd__branch__pagerank__fasttext__xlnet"
    )


def test_composite_centrality_id():

    config = ExperimentConfig(
        dataset="aapd",
        mode="no_branch",
        centrality=(
            "closeness*degree*clustering"
        ),
        embedding="bert",
        classifier="bert",
        top_k=200,
        max_nodes=25,
        sequence_length=25,
    )

    assert (
        config.experiment_id
        == (
            "aapd__no_branch__"
            "closeness_degree_clustering__"
            "bert__bert"
        )
    )


def test_invalid_dataset():

    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset="unknown",
            mode="branch",
            centrality="pagerank",
            embedding="bert",
            classifier="bert",
            top_k=200,
            max_nodes=25,
            sequence_length=25,
        )


def test_invalid_mode():

    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset="aapd",
            mode="invalid",
            centrality="pagerank",
            embedding="bert",
            classifier="bert",
            top_k=200,
            max_nodes=25,
            sequence_length=25,
        )


def test_invalid_centrality():

    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset="aapd",
            mode="branch",
            centrality="tfidf",
            embedding="bert",
            classifier="bert",
            top_k=200,
            max_nodes=25,
            sequence_length=25,
        )


def test_invalid_embedding():

    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset="aapd",
            mode="branch",
            centrality="pagerank",
            embedding="word2vec",
            classifier="bert",
            top_k=200,
            max_nodes=25,
            sequence_length=25,
        )


def test_invalid_classifier():

    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset="aapd",
            mode="branch",
            centrality="pagerank",
            embedding="bert",
            classifier="lstm",
            top_k=200,
            max_nodes=25,
            sequence_length=25,
        )


def test_invalid_top_k():

    with pytest.raises(ValueError):
        ExperimentConfig(
            dataset="aapd",
            mode="branch",
            centrality="pagerank",
            embedding="bert",
            classifier="bert",
            top_k=0,
            max_nodes=25,
            sequence_length=25,
        )


def test_registry_generates_96_experiments():

    experiments = (
        ExperimentRegistry.build_all(
            top_k_by_dataset={
                "aapd": 200,
                "rcv1": 100,
            },
            max_nodes=25,
            sequence_length=25,
        )
    )

    assert len(experiments) == 96


def test_registry_has_all_centralities():

    experiments = (
        ExperimentRegistry.build_all(
            top_k_by_dataset={
                "aapd": 200,
                "rcv1": 100,
            },
            max_nodes=25,
            sequence_length=25,
        )
    )

    strategies = {
        experiment.centrality
        for experiment in experiments
    }

    assert strategies == set(
        CENTRALITY_STRATEGIES
    )


def test_registry_has_both_modes():

    experiments = (
        ExperimentRegistry.build_all(
            top_k_by_dataset={
                "aapd": 200,
                "rcv1": 100,
            },
            max_nodes=25,
            sequence_length=25,
        )
    )

    modes = {
        experiment.mode
        for experiment in experiments
    }

    assert modes == {
        "branch",
        "no_branch",
    }


def test_registry_has_both_embeddings():

    experiments = (
        ExperimentRegistry.build_all(
            top_k_by_dataset={
                "aapd": 200,
                "rcv1": 100,
            },
            max_nodes=25,
            sequence_length=25,
        )
    )

    embeddings = {
        experiment.embedding
        for experiment in experiments
    }

    assert embeddings == {
        "bert",
        "fasttext",
    }


def test_registry_has_both_classifiers():

    experiments = (
        ExperimentRegistry.build_all(
            top_k_by_dataset={
                "aapd": 200,
                "rcv1": 100,
            },
            max_nodes=25,
            sequence_length=25,
        )
    )

    classifiers = {
        experiment.classifier
        for experiment in experiments
    }

    assert classifiers == {
        "bert",
        "xlnet",
    }