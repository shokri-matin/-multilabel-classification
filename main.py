from src.data.aapd import AAPDLoader
from src.preprocessing.pipeline import PreprocessingConfig
from src.preprocessing.pipeline import PreprocessingPipeline
from src.preprocessing.tokenizer import AAPDTokenizer
from src.corpus.storage import CorpusStorage


def main():

    # ---------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------

    raw_root = "data/raw/aapd"
    processed_root = "data/processed"

    # ---------------------------------------------------------
    # Loader
    # ---------------------------------------------------------

    loader = AAPDLoader(raw_root)

    # ---------------------------------------------------------
    # Preprocessing
    # ---------------------------------------------------------

    config = PreprocessingConfig(
        lowercase=True,
        remove_stopwords=True,
        stemming=True,
    )

    tokenizer = AAPDTokenizer()

    pipeline = PreprocessingPipeline(
        tokenizer=tokenizer,
        config=config,
    )

    # ---------------------------------------------------------
    # Load only a small sample
    # ---------------------------------------------------------

    raw_documents = loader.load_split("train")

    processed_documents = pipeline.process_many(
        raw_documents,
        show_progress=True,
    )

    # ---------------------------------------------------------
    # Create corpus
    # ---------------------------------------------------------

    from src.corpus.contracts import Corpus

    corpus = Corpus(
        name="aapd",
        split="train",
        documents=processed_documents,
    )

    print()
    print("=" * 60)
    print("CORPUS")
    print("=" * 60)

    print(f"Name: {corpus.name}")
    print(f"Split: {corpus.split}")
    print(f"Size: {corpus.size}")

    # ---------------------------------------------------------
    # Save
    # ---------------------------------------------------------

    storage = CorpusStorage(processed_root)

    output_dir = storage.save(corpus)

    print()
    print(f"Saved to: {output_dir}")

    # ---------------------------------------------------------
    # Load again
    # ---------------------------------------------------------

    loaded_corpus = storage.load(
        name="aapd",
        split="train",
    )

    print()
    print("=" * 60)
    print("LOADED CORPUS")
    print("=" * 60)

    print(f"Name: {loaded_corpus.name}")
    print(f"Split: {loaded_corpus.split}")
    print(f"Size: {loaded_corpus.size}")

    # ---------------------------------------------------------
    # Validation
    # ---------------------------------------------------------

    assert loaded_corpus.name == corpus.name
    assert loaded_corpus.split == corpus.split
    assert loaded_corpus.size == corpus.size

    assert (
        loaded_corpus.document_ids
        == corpus.document_ids
    )

    assert (
        loaded_corpus.labels
        == corpus.labels
    )

    for original, loaded in zip(
        corpus.documents,
        loaded_corpus.documents,
    ):
        assert original.doc_id == loaded.doc_id
        assert original.text == loaded.text
        assert original.labels == loaded.labels
        assert original.token_texts == loaded.token_texts
        assert original.positions == loaded.positions

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()

