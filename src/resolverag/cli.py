"""ResolveRAG command-line interface."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from resolverag.data.techqa import prepare_dataset
from resolverag.domain.models import ChunkingStrategy, RetrievalConfiguration
from resolverag.evaluation.runner import run_retrieval_benchmark
from resolverag.exceptions import (
    ConfigurationError,
    DatasetValidationError,
    EvaluationError,
    RetrievalError,
)
from resolverag.indexing.indexer import build_indexes
from resolverag.retrieval.runtime import RetrievalRuntime

app = typer.Typer(
    name="resolverag",
    help="Build and evaluate an evidence-grounded technical-support RAG system.",
    no_args_is_help=True,
)
dataset_app = typer.Typer(help="Acquire and prepare benchmark datasets.")
index_app = typer.Typer(help="Chunk, embed, and index canonical documents.")
retrieve_app = typer.Typer(help="Search indexed technical-support documents.")
evaluate_app = typer.Typer(help="Run reproducible offline evaluations.")
app.add_typer(dataset_app, name="dataset")
app.add_typer(index_app, name="index")
app.add_typer(retrieve_app, name="retrieve")
app.add_typer(evaluate_app, name="evaluate")
console = Console()


@dataset_app.command("prepare")
def prepare_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Dataset contract YAML file.",
        ),
    ] = Path("configs/datasets/techqa.yaml"),
    *,
    download: Annotated[
        bool,
        typer.Option(
            "--download/--no-download",
            help="Download missing or invalid source files from the pinned revision.",
        ),
    ] = True,
) -> None:
    """Prepare canonical documents, queries, qrels, and provenance."""

    try:
        result = prepare_dataset(config, download=download)
    except (ConfigurationError, DatasetValidationError, OSError) as error:
        console.print(f"[bold red]Dataset preparation failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    counts = result.manifest.counts
    table = Table(title="TechQA dataset prepared")
    table.add_column("Artifact or measure")
    table.add_column("Value", justify="right")
    table.add_row("Documents", f"{counts.documents:,}")
    table.add_row("Queries", f"{counts.queries:,}")
    table.add_row(
        "Answerable / unanswerable",
        f"{counts.answerable_queries:,} / {counts.unanswerable_queries:,}",
    )
    table.add_row(
        "Development / held-out",
        f"{counts.development_queries:,} / {counts.held_out_queries:,}",
    )
    table.add_row(
        "Answerable without reference answer",
        f"{counts.answerable_queries_without_reference_answer:,}",
    )
    table.add_row("Relevance judgments", f"{counts.relevance_judgments:,}")
    table.add_row("Manifest", str(result.manifest_path))
    console.print(table)


@evaluate_app.command("retrieval")
def evaluate_retrieval_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Retrieval evaluation configuration.",
        ),
    ] = Path("configs/evaluation/retrieval.yaml"),
    index_manifest: Annotated[
        Path | None,
        typer.Option(
            "--index-manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Override the full-corpus index manifest.",
        ),
    ] = None,
    sample_size: Annotated[
        int | None,
        typer.Option("--sample-size", min=1, help="Override the configured sample size."),
    ] = None,
    strategy: Annotated[
        list[ChunkingStrategy] | None,
        typer.Option("--strategy", help="Repeat to evaluate selected chunkers only."),
    ] = None,
    retriever: Annotated[
        list[RetrievalConfiguration] | None,
        typer.Option("--retriever", help="Repeat to evaluate selected retrievers only."),
    ] = None,
    *,
    resume: Annotated[
        bool,
        typer.Option(
            "--resume/--no-resume",
            help="Resume completed query/configuration pairs after interruption.",
        ),
    ] = True,
) -> None:
    """Evaluate a retrieval matrix and generate a ranked leaderboard."""

    try:
        result = run_retrieval_benchmark(
            config,
            index_manifest_path=index_manifest,
            sample_size=sample_size,
            strategies=strategy,
            retrievers=retriever,
            resume=resume,
        )
    except (ConfigurationError, EvaluationError, RetrievalError, OSError) as error:
        console.print(f"[bold red]Evaluation failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    title = "Retrieval leaderboard"
    if not result.manifest.official:
        title = f"{title} (validation only)"
    table = Table(title=title)
    table.add_column("Rank", justify="right")
    table.add_column("Chunking")
    table.add_column("Retriever")
    table.add_column("R@10", justify="right")
    table.add_column("MRR@10", justify="right")
    table.add_column("nDCG@10", justify="right")
    table.add_column("p95 ms", justify="right")
    table.add_column("Failures", justify="right")
    for rank, row in enumerate(result.leaderboard[:10], start=1):
        table.add_row(
            str(rank),
            row.strategy.value,
            row.retriever.value,
            f"{row.recall_at_10:.3f}",
            f"{row.mrr_at_10:.3f}",
            f"{row.ndcg_at_10:.3f}",
            f"{row.p95_latency_ms:.1f}",
            str(row.failures),
        )
    console.print(table)
    console.print(f"Report: {result.summary_path}")


@index_app.command("build")
def build_index_command(
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Chunking and local-index configuration.",
        ),
    ] = Path("configs/indexes/local.yaml"),
    strategy: Annotated[
        list[ChunkingStrategy] | None,
        typer.Option(
            "--strategy",
            help="Strategy to build; repeat the option to select multiple strategies.",
        ),
    ] = None,
    limit: Annotated[
        int | None,
        typer.Option(
            "--limit",
            min=1,
            help="Index only the first N documents for a smoke run.",
        ),
    ] = None,
    *,
    recreate: Annotated[
        bool,
        typer.Option(
            "--recreate/--no-recreate",
            help="Replace collections with the same deterministic names.",
        ),
    ] = True,
) -> None:
    """Build one or more chunk artifacts and Qdrant collections."""

    try:
        result = build_indexes(
            config,
            strategies=strategy,
            document_limit=limit,
            recreate=recreate,
        )
    except (ConfigurationError, DatasetValidationError, RuntimeError, OSError) as error:
        console.print(f"[bold red]Index build failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title=f"ResolveRAG indexes built ({result.manifest.scope})")
    table.add_column("Strategy")
    table.add_column("Documents", justify="right")
    table.add_column("Chunks", justify="right")
    table.add_column("p50 tokens", justify="right")
    table.add_column("p95 tokens", justify="right")
    table.add_column("Seconds", justify="right")
    for collection in result.manifest.collections:
        statistics = collection.statistics
        table.add_row(
            collection.strategy.value,
            f"{statistics.documents:,}",
            f"{statistics.chunks:,}",
            f"{statistics.p50_tokens:,}",
            f"{statistics.p95_tokens:,}",
            f"{collection.build_seconds:.2f}",
        )
    console.print(table)
    console.print(f"Manifest: {result.manifest_path}")


@retrieve_app.command("search")
def retrieve_search_command(
    query: Annotated[str, typer.Option("--query", "-q", help="Technical-support query.")],
    strategy: Annotated[
        ChunkingStrategy,
        typer.Option("--strategy", help="Chunking strategy collection to search."),
    ] = ChunkingStrategy.FIXED_256,
    retriever: Annotated[
        RetrievalConfiguration,
        typer.Option("--retriever", help="Retrieval configuration to run."),
    ] = RetrievalConfiguration.DENSE,
    config: Annotated[
        Path,
        typer.Option(
            "--config",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Retrieval configuration.",
        ),
    ] = Path("configs/retrieval/local.yaml"),
    index_manifest: Annotated[
        Path | None,
        typer.Option(
            "--index-manifest",
            exists=True,
            dir_okay=False,
            readable=True,
            help="Override the full index manifest, for example with a smoke build.",
        ),
    ] = None,
    top_k: Annotated[int | None, typer.Option("--top-k", min=1)] = None,
) -> None:
    """Search one chunking strategy with one retrieval configuration."""

    try:
        with RetrievalRuntime(
            config,
            strategy,
            index_manifest_path=index_manifest,
        ) as runtime:
            results = runtime.search(query, retriever, top_k=top_k)
    except (ConfigurationError, RetrievalError, RuntimeError, OSError) as error:
        console.print(f"[bold red]Retrieval failed:[/bold red] {error}")
        raise typer.Exit(code=1) from error

    table = Table(title=f"{retriever.value} over {strategy.value}")
    table.add_column("Rank", justify="right")
    table.add_column("Document")
    table.add_column("Score", justify="right")
    table.add_column("Passage")
    for result in results:
        excerpt = " ".join(result.chunk.context_text.split())
        if len(excerpt) > 120:
            excerpt = f"{excerpt[:117]}..."
        table.add_row(
            str(result.rank),
            result.chunk.document_id,
            f"{result.score:.4f}",
            excerpt,
        )
    console.print(table)


def main() -> None:
    """Run the ResolveRAG CLI."""

    app()


if __name__ == "__main__":
    main()
