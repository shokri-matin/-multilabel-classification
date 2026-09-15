from pathlib import Path

from src.data.aapd import AAPDLoader

from src.corpus.builder import CorpusBuilder
from src.corpus.storage import CorpusStorage

from src.preprocessing.pipeline import (
    PreprocessingConfig,
    PreprocessingPipeline,
)
from src.preprocessing.tokenizer import AAPDTokenizer

from src.graph.word_graph import WordGraphBuilder
from src.graph.centrality import CentralityCalculator

from src.experiments.config import ExperimentConfig
from src.experiments.dataset_context import DatasetContextBuilder

from src.experiments.graph_cache import GraphArtifactCache
from src.experiments.graph_context import (
    DatasetGraphContextBuilder,
)

from src.experiments.sequence_cache import (
    SequenceArtifactCache,
)
from src.experiments.sequence_context import (
    DatasetSequenceContextBuilder,
)

from src.experiments.embedding_cache import (
    EmbeddingArtifactCache,
)
from src.experiments.embedding_context import (
    DatasetEmbeddingContextBuilder,
)

from src.experiments.experiment_context import (
    ExperimentContextBuilder,
)

from src.experiments.training_config import (
    TrainingConfig,
)

from src.experiments.executor import (
    ExperimentExecutor,
)

from src.experiments.runner import (
    ExperimentRunner,
)

from src.embedding.bert import (
    BERTConfig,
    BERTEmbedder,
)


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

# ------------------------------------------------------------
# IMPORTANT:
# Change this path if your AAPD files are stored elsewhere.
#
# Expected files:
#
# data/aapd/
# ├── text_train
# ├── label_train
# ├── text_validation
# ├── label_validation
# ├── text_test
# └── label_test
# ------------------------------------------------------------

AAPD_ROOT = PROJECT_ROOT / "data" / "raw" / "aapd"


# ============================================================
# Smoke-test artifact directories
# ============================================================

SMOKE_ROOT = PROJECT_ROOT / "results" / "smoke"

CORPUS_ROOT = SMOKE_ROOT / "corpus"

GRAPH_CACHE_ROOT = SMOKE_ROOT / "graphs"

SEQUENCE_CACHE_ROOT = SMOKE_ROOT / "sequences"

EMBEDDING_CACHE_ROOT = SMOKE_ROOT / "embeddings"

CHECKPOINT_ROOT = SMOKE_ROOT / "checkpoints"

EXPERIMENT_RESULT_ROOT = (
    PROJECT_ROOT / "results" / "experiments"
)


# ============================================================
# Smoke-test dataset limits
# ============================================================

DATASET_LIMITS = {
    "train": 100,
    "validation": 20,
    "test": 20,
}


# ============================================================
# Limited AAPD Loader
# ============================================================

class LimitedAAPDLoader:
    """
    Wrapper around AAPDLoader used only for the smoke test.

    It limits the number of documents loaded from each split
    without modifying the production AAPDLoader.
    """

    def __init__(
        self,
        loader,
        limits,
    ):
        self.loader = loader
        self.limits = limits

    def load_split(
        self,
        split: str,
    ):
        documents = self.loader.load_split(split)

        limit = self.limits.get(split)

        if limit is None:
            return documents

        return documents[:limit]


# ============================================================
# Utility
# ============================================================

def print_section(title: str):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


# ============================================================
# Main
# ============================================================

def main():

    print_section(
        "AAPD REAL-DATA E2E SMOKE TEST"
    )

    # ========================================================
    # 0. Check paths
    # ========================================================

    print_section("0. Checking paths")

    print(f"Project root : {PROJECT_ROOT}")
    print(f"AAPD root    : {AAPD_ROOT}")

    if not AAPD_ROOT.exists():
        raise FileNotFoundError(
            f"AAPD directory does not exist:\n{AAPD_ROOT}"
        )

    print("AAPD directory: OK")

    # ========================================================
    # 1. Experiment configuration
    # ========================================================

    print_section("1. Experiment configuration")

    experiment = ExperimentConfig(
        dataset="aapd",
        mode="branch",
        centrality="closeness",
        embedding="bert",
        classifier="bert",
        top_k=50,
        max_nodes=30,
        sequence_length=32,
    )

    print(
        f"dataset       = {experiment.dataset}"
    )
    print(
        f"mode          = {experiment.mode}"
    )
    print(
        f"centrality    = {experiment.centrality}"
    )
    print(
        f"embedding     = {experiment.embedding}"
    )
    print(
        f"classifier    = {experiment.classifier}"
    )
    print(
        f"top_k         = {experiment.top_k}"
    )
    print(
        f"max_nodes     = {experiment.max_nodes}"
    )
    print(
        f"sequence_len  = {experiment.sequence_length}"
    )
    print(
        f"experiment_id = {experiment.experiment_id}"
    )

    # ========================================================
    # 2. AAPD loader
    # ========================================================

    print_section("2. Loading AAPD")

    base_loader = AAPDLoader(
        root=AAPD_ROOT,
    )

    loader = LimitedAAPDLoader(
        loader=base_loader,
        limits=DATASET_LIMITS,
    )

    print(
        f"train limit      = "
        f"{DATASET_LIMITS['train']}"
    )

    print(
        f"validation limit = "
        f"{DATASET_LIMITS['validation']}"
    )

    print(
        f"test limit       = "
        f"{DATASET_LIMITS['test']}"
    )

    # ========================================================
    # 3. Preprocessing
    # ========================================================

    print_section("3. Building preprocessing pipeline")

    preprocessing_config = PreprocessingConfig(
        lowercase=True,
        remove_stopwords=True,
        stemming=False,
    )

    tokenizer = AAPDTokenizer()

    preprocessing_pipeline = PreprocessingPipeline(
        tokenizer=tokenizer,
        config=preprocessing_config,
    )

    print(
        f"lowercase        = "
        f"{preprocessing_config.lowercase}"
    )

    print(
        f"remove_stopwords = "
        f"{preprocessing_config.remove_stopwords}"
    )

    print(
        f"stemming         = "
        f"{preprocessing_config.stemming}"
    )

    # ========================================================
    # 4. Corpus
    # ========================================================

    print_section("4. Building corpus")

    corpus_builder = CorpusBuilder(
        name="aapd",
        loader=loader,
        preprocessing_pipeline=preprocessing_pipeline,
    )

    corpora = corpus_builder.build_all()

    train_corpus = corpora["train"]
    validation_corpus = corpora["validation"]
    test_corpus = corpora["test"]

    print(
        f"train documents      = "
        f"{len(train_corpus.documents)}"
    )

    print(
        f"validation documents = "
        f"{len(validation_corpus.documents)}"
    )

    print(
        f"test documents       = "
        f"{len(test_corpus.documents)}"
    )

    # ========================================================
    # 5. Corpus storage
    # ========================================================

    print_section("5. Saving corpus artifacts")

    corpus_storage = CorpusStorage(
        root=CORPUS_ROOT,
    )

    corpus_storage.save(
        train_corpus
    )

    corpus_storage.save(
        validation_corpus
    )

    corpus_storage.save(
        test_corpus
    )

    print(
        f"Corpus artifacts saved to:\n"
        f"{CORPUS_ROOT}"
    )

    # ========================================================
    # 6. Dataset context
    # ========================================================

    print_section("6. Building dataset context")

    dataset_context_builder = DatasetContextBuilder()

    dataset_context = (
        dataset_context_builder.build(
            dataset="aapd",
            train_corpus=train_corpus,
            validation_corpus=validation_corpus,
            test_corpus=test_corpus,
        )
    )

    print(
        f"train size      = "
        f"{dataset_context.train_size}"
    )

    print(
        f"validation size = "
        f"{dataset_context.validation_size}"
    )

    print(
        f"test size       = "
        f"{dataset_context.test_size}"
    )

    print(
        f"num labels      = "
        f"{dataset_context.num_labels}"
    )

    # ========================================================
    # 7. Graph context
    # ========================================================

    print_section("7. Building graph context")

    graph_cache = GraphArtifactCache(
        root_dir=GRAPH_CACHE_ROOT,
    )

    graph_builder = WordGraphBuilder()

    centrality_calculator = (
        CentralityCalculator()
    )

    graph_context_builder = (
        DatasetGraphContextBuilder(
            cache=graph_cache,
            graph_builder=graph_builder,
            centrality_calculator=(
                centrality_calculator
            ),
        )
    )

    graph_context = (
        graph_context_builder.build(
            dataset="aapd",
            train_corpus=train_corpus,
            validation_corpus=validation_corpus,
            test_corpus=test_corpus,
        )
    )

    print(
        "Graph context created successfully."
    )

    print(
        "Centrality strategies will be "
        "computed lazily per document."
    )

    # ========================================================
    # 8. Sequence context
    # ========================================================

    print_section("8. Building sequence context")

    sequence_cache = SequenceArtifactCache(
        root_dir=SEQUENCE_CACHE_ROOT,
    )

    sequence_context_builder = (
        DatasetSequenceContextBuilder(
            graph_context=graph_context,
            sequence_cache=sequence_cache,
            top_k=experiment.top_k,
            max_nodes=experiment.max_nodes,
            sequence_length=(
                experiment.sequence_length
            ),
        )
    )

    sequence_context = (
        sequence_context_builder.build(
            dataset="aapd",
        )
    )

    print(
        "Sequence context created successfully."
    )

    print(
        f"top_k        = {experiment.top_k}"
    )

    print(
        f"max_nodes    = {experiment.max_nodes}"
    )

    print(
        f"sequence_len = "
        f"{experiment.sequence_length}"
    )

    # ========================================================
    # 9. Embedding context
    # ========================================================

    print_section("9. Building embedding context")

    embedding_cache = (
        EmbeddingArtifactCache(
            root_dir=EMBEDDING_CACHE_ROOT,
        )
    )

    embedding_context_builder = (
        DatasetEmbeddingContextBuilder(
            sequence_context=sequence_context,
            embedding_cache=embedding_cache,
        )
    )

    embedding_context = (
        embedding_context_builder.build(
            dataset="aapd",
        )
    )

    print(
        "Embedding context created successfully."
    )

    # ========================================================
    # 10. BERT embedder
    # ========================================================

    print_section("10. Loading BERT embedder")

    bert_config = BERTConfig(
        model_name="bert-base-uncased",
        max_length=128,
    )

    print(
        f"model = {bert_config.model_name}"
    )

    print(
        f"max_length = {bert_config.max_length}"
    )

    print(
        "Loading BERT model/tokenizer..."
    )

    bert_embedder = BERTEmbedder(
        config=bert_config,
    )

    print(
        "BERT loaded successfully."
    )

    print(
        f"embedding_dim = "
        f"{bert_embedder.embedding_dim}"
    )

    embedder_registry = {
        "bert": bert_embedder,
    }

    # ========================================================
    # 11. Experiment context
    # ========================================================

    print_section("11. Building experiment context")

    experiment_context_builder = (
        ExperimentContextBuilder(
            dataset_context=dataset_context,
            sequence_context=sequence_context,
            embedding_context=embedding_context,
            embedder_registry=embedder_registry,
        )
    )

    # ========================================================
    # 12. Training configuration
    # ========================================================

    training_config = TrainingConfig(
        learning_rate=2e-5,
        weight_decay=0.01,
        batch_size=8,
        epochs=1,
        gradient_accumulation_steps=1,
        max_grad_norm=1.0,
        optimizer="adamw",
        seed=42,
        device="auto",
        checkpoint_dir=str(
            CHECKPOINT_ROOT
        ),
        save_best_only=True,
        early_stopping_patience=2,
        num_workers=0,
    )

    print(
        f"batch_size = "
        f"{training_config.batch_size}"
    )

    print(
        f"epochs = "
        f"{training_config.epochs}"
    )

    print(
        f"learning_rate = "
        f"{training_config.learning_rate}"
    )

    print(
        f"device = "
        f"{training_config.device}"
    )

    # ========================================================
    # 13. Build experiment context
    # ========================================================

    experiment_context = (
        experiment_context_builder.build(
            experiment=experiment,
            batch_size=training_config.batch_size,
            shuffle_train=True,
        )
    )

    print(
        "Experiment context created."
    )

    # ========================================================
    # 14. Experiment executor
    # ========================================================

    print_section("12. Preparing experiment executor")

    executor = ExperimentExecutor(
        context=experiment_context,
        training_config=training_config,
    )

    print(
        "Experiment executor created."
    )

    # ========================================================
    # 15. Experiment runner
    # ========================================================

    runner = ExperimentRunner(
        artifact_dir=EXPERIMENT_RESULT_ROOT,
    )

    print(
        f"Result directory:\n"
        f"{EXPERIMENT_RESULT_ROOT}"
    )

    # ========================================================
    # 16. Execute
    # ========================================================

    print_section(
        "13. RUNNING REAL-DATA EXPERIMENT"
    )

    print(
        f"Experiment ID:\n"
        f"{experiment.experiment_id}"
    )

    print()
    print(
        "Pipeline:"
    )

    print(
        "AAPD"
    )
    print(
        "  -> preprocessing"
    )
    print(
        "  -> corpus"
    )
    print(
        "  -> word graph"
    )
    print(
        "  -> centrality"
    )
    print(
        "  -> graph processing"
    )
    print(
        "  -> BFS / DFS"
    )
    print(
        "  -> word sequences"
    )
    print(
        "  -> BERT embeddings"
    )
    print(
        "  -> classifier"
    )
    print(
        "  -> training"
    )
    print(
        "  -> checkpoint"
    )
    print(
        "  -> test evaluation"
    )

    print()
    print(
        "Starting..."
    )

    result = runner.run(
        experiment=experiment,
        execute=lambda config: (
            executor.execute()
        ),
        skip_completed=False,
    )

    # ========================================================
    # 17. Result
    # ========================================================

    print_section(
        "14. E2E RESULT"
    )

    print(
        f"status            = "
        f"{result.status}"
    )

    print(
        f"experiment_id     = "
        f"{result.experiment_id}"
    )

    print(
        f"dataset            = "
        f"{result.dataset}"
    )

    print(
        f"mode               = "
        f"{result.mode}"
    )

    print(
        f"centrality         = "
        f"{result.centrality}"
    )

    print(
        f"embedding          = "
        f"{result.embedding}"
    )

    print(
        f"classifier         = "
        f"{result.classifier}"
    )

    print(
        f"train_loss         = "
        f"{result.train_loss}"
    )

    print(
        f"validation_loss    = "
        f"{result.validation_loss}"
    )

    print(
        f"micro_precision    = "
        f"{result.micro_precision}"
    )

    print(
        f"micro_recall       = "
        f"{result.micro_recall}"
    )

    print(
        f"micro_f1           = "
        f"{result.micro_f1}"
    )

    print(
        f"macro_precision    = "
        f"{result.macro_precision}"
    )

    print(
        f"macro_recall       = "
        f"{result.macro_recall}"
    )

    print(
        f"macro_f1           = "
        f"{result.macro_f1}"
    )

    print(
        f"hamming_loss       = "
        f"{result.hamming_loss}"
    )

    print(
        f"best_epoch         = "
        f"{result.best_epoch}"
    )

    print(
        f"training_epochs    = "
        f"{result.training_epochs}"
    )

    print(
        f"checkpoint_path    = "
        f"{result.checkpoint_path}"
    )

    if result.error:
        print()
        print(
            "ERROR:"
        )
        print(
            result.error
        )

    print_section(
        "SMOKE TEST FINISHED"
    )


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":
    main()