import numpy as np

from src.embedding.fasttext import (
    FastTextConfig,
    FastTextEmbedder,
)


MODEL_PATH = (
    "src/models/fasttext/crawl-300d-2M-subword.bin"
)


def test_real_fasttext_model():
    config = FastTextConfig(
        model_path=MODEL_PATH,
        embedding_dim=300,
        normalize=False,
    )

    embedder = FastTextEmbedder(config)

    assert embedder.embedding_dim == 300

    known_vector = embedder._get_word_vector(
        "machine"
    )

    assert known_vector.shape == (300,)
    assert known_vector.dtype == np.float32
    assert np.linalg.norm(known_vector) > 0


def test_real_fasttext_oov():
    config = FastTextConfig(
        model_path=MODEL_PATH,
        embedding_dim=300,
        normalize=False,
    )

    embedder = FastTextEmbedder(config)

    oov_vector = embedder._get_word_vector(
        "machinelearningxyz"
    )

    assert oov_vector.shape == (300,)
    assert oov_vector.dtype == np.float32
    assert np.linalg.norm(oov_vector) > 0