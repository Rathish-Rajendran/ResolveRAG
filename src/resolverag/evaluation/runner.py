"""Resumable execution and reporting for the 5-by-6 retrieval matrix."""

import csv
import json
import random
from collections import defaultdict
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from statistics import fmean
from time import perf_counter
from typing import IO, Protocol

import yaml
from pydantic import ValidationError

from resolverag.data.integrity import sha256_file, sha256_text
from resolverag.data.serialization import atomic_text_writer, write_text
from resolverag.domain.models import (
    ChunkingStrategy,
    EvaluationQuery,
    RelevanceJudgment,
    RetrievalConfiguration,
    RetrievalResult,
)
from resolverag.evaluation.config import EvaluationConfig, load_evaluation_config
from resolverag.evaluation.metrics import RetrievalMetrics, evaluate_ranking
from resolverag.evaluation.models import (
    BenchmarkManifest,
    LeaderboardEntry,
    QueryEvaluation,
)
from resolverag.exceptions import EvaluationError
from resolverag.indexing.models import IndexBuildManifest
from resolverag.retrieval.runtime import RetrievalRuntime


class EvaluationRuntime(Protocol):
    def search(
        self,
        query: str,
        configuration: RetrievalConfiguration,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievalResult, ...]: ...

    def close(self) -> None: ...


RuntimeFactory = Callable[[Path, ChunkingStrategy, Path], EvaluationRuntime]


@dataclass(frozen=True)
class BenchmarkResult:
    output_directory: Path
    manifest_path: Path
    results_path: Path
    leaderboard_path: Path
    summary_path: Path
    winner_path: Path
    manifest: BenchmarkManifest
    leaderboard: tuple[LeaderboardEntry, ...]


def _default_runtime_factory(
    config_path: Path,
    strategy: ChunkingStrategy,
    index_manifest_path: Path,
) -> EvaluationRuntime:
    return RetrievalRuntime(
        config_path,
        strategy,
        index_manifest_path=index_manifest_path,
    )


def _load_jsonl[EvaluationRecord: (EvaluationQuery, RelevanceJudgment)](
    path: Path,
    model: type[EvaluationRecord],
) -> list[EvaluationRecord]:
    if not path.is_file():
        raise EvaluationError(f"Required evaluation artifact does not exist: {path}")
    records: list[EvaluationRecord] = []
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    records.append(model.model_validate_json(line))
                except ValidationError as error:
                    raise EvaluationError(
                        f"Invalid record at {path}:{line_number}: {error}"
                    ) from error
    except OSError as error:
        raise EvaluationError(f"Unable to read evaluation artifact {path}: {error}") from error
    return records


def _sample_queries(
    queries: Sequence[EvaluationQuery],
    sample_size: int | None,
    *,
    seed: int,
    strata_count: int,
) -> tuple[EvaluationQuery, ...]:
    ordered = sorted(queries, key=lambda query: (len(query.question.split()), query.query_id))
    if sample_size is None or sample_size >= len(ordered):
        return tuple(sorted(ordered, key=lambda query: query.query_id))
    buckets: list[list[EvaluationQuery]] = [[] for _ in range(strata_count)]
    for index, query in enumerate(ordered):
        bucket_index = min(strata_count - 1, index * strata_count // len(ordered))
        buckets[bucket_index].append(query)
    randomizer = random.Random(seed)
    for bucket in buckets:
        randomizer.shuffle(bucket)
    selected: list[EvaluationQuery] = []
    while len(selected) < sample_size:
        added = False
        for bucket in buckets:
            if bucket and len(selected) < sample_size:
                selected.append(bucket.pop())
                added = True
        if not added:
            break
    return tuple(sorted(selected, key=lambda query: query.query_id))


def _deduplicate_documents(results: Sequence[RetrievalResult]) -> tuple[str, ...]:
    seen: set[str] = set()
    document_ids: list[str] = []
    for result in results:
        document_id = result.chunk.document_id
        if document_id not in seen:
            seen.add(document_id)
            document_ids.append(document_id)
    return tuple(document_ids)


def _query_record(
    query: EvaluationQuery,
    strategy: ChunkingStrategy,
    retriever: RetrievalConfiguration,
    relevant: set[str],
    latency_ms: float,
    results: Sequence[RetrievalResult],
    error: str | None = None,
) -> QueryEvaluation:
    retrieved = _deduplicate_documents(results)
    metrics = (
        evaluate_ranking(retrieved, relevant) if error is None else RetrievalMetrics(0, 0, 0, 0, 0)
    )
    return QueryEvaluation(
        query_id=query.query_id,
        strategy=strategy,
        retriever=retriever,
        latency_ms=latency_ms,
        retrieved_document_ids=retrieved,
        relevant_document_ids=tuple(sorted(relevant)),
        recall_at_1=metrics.recall_at_1,
        recall_at_5=metrics.recall_at_5,
        recall_at_10=metrics.recall_at_10,
        reciprocal_rank_at_10=metrics.reciprocal_rank_at_10,
        ndcg_at_10=metrics.ndcg_at_10,
        error=error,
    )


def _percentile(values: Sequence[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, int((len(ordered) - 1) * percentile + 0.5)))
    return ordered[rank]


def _aggregate(records: Sequence[QueryEvaluation]) -> LeaderboardEntry:
    if not records:
        raise EvaluationError("Cannot aggregate an empty evaluation result set")
    return LeaderboardEntry(
        strategy=records[0].strategy,
        retriever=records[0].retriever,
        evaluated_queries=len(records),
        failures=sum(record.error is not None for record in records),
        recall_at_1=fmean(record.recall_at_1 for record in records),
        recall_at_5=fmean(record.recall_at_5 for record in records),
        recall_at_10=fmean(record.recall_at_10 for record in records),
        mrr_at_10=fmean(record.reciprocal_rank_at_10 for record in records),
        ndcg_at_10=fmean(record.ndcg_at_10 for record in records),
        p50_latency_ms=_percentile([record.latency_ms for record in records], 0.50),
        p95_latency_ms=_percentile([record.latency_ms for record in records], 0.95),
    )


def _leaderboard(
    records: dict[tuple[str, str, str], QueryEvaluation],
    strategies: Sequence[ChunkingStrategy],
    retrievers: Sequence[RetrievalConfiguration],
    query_ids: Sequence[str],
) -> tuple[LeaderboardEntry, ...]:
    rows: list[LeaderboardEntry] = []
    for strategy in strategies:
        for retriever in retrievers:
            combination = [
                records[(query_id, strategy.value, retriever.value)] for query_id in query_ids
            ]
            rows.append(_aggregate(combination))
    return tuple(
        sorted(
            rows,
            key=lambda row: (
                -row.ndcg_at_10,
                -row.recall_at_10,
                -row.mrr_at_10,
                row.p95_latency_ms,
            ),
        )
    )


def _write_leaderboard(path: Path, rows: Sequence[LeaderboardEntry]) -> None:
    fieldnames = [
        "rank",
        "strategy",
        "retriever",
        "evaluated_queries",
        "failures",
        "recall_at_1",
        "recall_at_5",
        "recall_at_10",
        "mrr_at_10",
        "ndcg_at_10",
        "p50_latency_ms",
        "p95_latency_ms",
    ]
    with atomic_text_writer(path) as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        for rank, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "rank": rank,
                    "strategy": row.strategy.value,
                    "retriever": row.retriever.value,
                    "evaluated_queries": row.evaluated_queries,
                    "failures": row.failures,
                    "recall_at_1": f"{row.recall_at_1:.6f}",
                    "recall_at_5": f"{row.recall_at_5:.6f}",
                    "recall_at_10": f"{row.recall_at_10:.6f}",
                    "mrr_at_10": f"{row.mrr_at_10:.6f}",
                    "ndcg_at_10": f"{row.ndcg_at_10:.6f}",
                    "p50_latency_ms": f"{row.p50_latency_ms:.3f}",
                    "p95_latency_ms": f"{row.p95_latency_ms:.3f}",
                }
            )


def _summary_markdown(
    manifest: BenchmarkManifest,
    rows: Sequence[LeaderboardEntry],
) -> str:
    label = "Official full-corpus benchmark" if manifest.official else "VALIDATION ONLY"
    lines = [
        "# ResolveRAG Retrieval Benchmark",
        "",
        f"**Status:** {label}",
        f"**Index scope:** `{manifest.index_scope}`  ",
        f"**Evaluation split:** `{manifest.split.value}`  ",
        f"**Questions:** {len(manifest.sample_query_ids)}",
        "",
        "| Rank | Chunking | Retrieval | Recall@1 | Recall@5 | Recall@10 | "
        "MRR@10 | nDCG@10 | p95 ms | Failures |",
        "|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(rows, start=1):
        lines.append(
            f"| {rank} | {row.strategy.value} | {row.retriever.value} | "
            f"{row.recall_at_1:.3f} | {row.recall_at_5:.3f} | "
            f"{row.recall_at_10:.3f} | {row.mrr_at_10:.3f} | "
            f"{row.ndcg_at_10:.3f} | {row.p95_latency_ms:.1f} | {row.failures} |"
        )
    if not manifest.official:
        lines.extend(
            [
                "",
                "> Smoke-index results validate the evaluation machinery only and must not be "
                "reported as benchmark performance.",
            ]
        )
    return "\n".join(lines) + "\n"


def _load_existing_results(path: Path) -> dict[tuple[str, str, str], QueryEvaluation]:
    records: dict[tuple[str, str, str], QueryEvaluation] = {}
    if not path.is_file():
        return records
    try:
        with path.open(encoding="utf-8") as source:
            for line_number, line in enumerate(source, start=1):
                try:
                    record = QueryEvaluation.model_validate_json(line)
                except ValidationError as error:
                    raise EvaluationError(
                        f"Invalid resumable result at {path}:{line_number}: {error}"
                    ) from error
                records[(record.query_id, record.strategy.value, record.retriever.value)] = record
    except OSError as error:
        raise EvaluationError(f"Unable to read resumable results {path}: {error}") from error
    return records


def _append_result(destination: IO[str], record: QueryEvaluation) -> None:
    destination.write(record.model_dump_json(exclude_none=False) + "\n")
    destination.flush()


def run_retrieval_benchmark(
    config_path: Path,
    *,
    index_manifest_path: Path | None = None,
    sample_size: int | None = None,
    strategies: Sequence[ChunkingStrategy] | None = None,
    retrievers: Sequence[RetrievalConfiguration] | None = None,
    resume: bool = True,
    runtime_factory: RuntimeFactory = _default_runtime_factory,
) -> BenchmarkResult:
    """Execute or resume a retrieval matrix and generate its leaderboard."""

    config: EvaluationConfig = load_evaluation_config(config_path)
    retrieval_config_path = config.inputs.retrieval_config_file
    from resolverag.retrieval.config import load_retrieval_config

    retrieval_config = load_retrieval_config(retrieval_config_path)
    selected_manifest_path = index_manifest_path or retrieval_config.inputs.index_manifest_file
    try:
        index_manifest = IndexBuildManifest.model_validate_json(
            selected_manifest_path.read_text(encoding="utf-8")
        )
    except (OSError, ValidationError) as error:
        raise EvaluationError(
            f"Invalid index manifest {selected_manifest_path}: {error}"
        ) from error

    selected_strategies = tuple(strategies or config.matrix.chunking_strategies)
    selected_retrievers = tuple(retrievers or config.matrix.retrieval_configurations)
    available_strategies = {collection.strategy for collection in index_manifest.collections}
    missing = set(selected_strategies) - available_strategies
    if missing:
        names = ", ".join(sorted(strategy.value for strategy in missing))
        raise EvaluationError(f"Index manifest is missing strategies: {names}")

    queries = _load_jsonl(config.inputs.queries_file, EvaluationQuery)
    qrels = _load_jsonl(config.inputs.qrels_file, RelevanceJudgment)
    relevance: dict[str, set[str]] = defaultdict(set)
    for judgment in qrels:
        relevance[judgment.query_id].add(judgment.document_id)
    candidates = [
        query
        for query in queries
        if query.split is config.benchmark.split and relevance.get(query.query_id)
    ]
    chosen_sample_size = sample_size if sample_size is not None else config.benchmark.sample_size
    sampled_queries = _sample_queries(
        candidates,
        chosen_sample_size,
        seed=config.benchmark.random_seed,
        strata_count=config.benchmark.length_strata,
    )
    if not sampled_queries:
        raise EvaluationError("No answerable queries are available for the requested split")

    plan = {
        "evaluation_config": sha256_file(config_path),
        "retrieval_config": sha256_file(retrieval_config_path),
        "index_manifest": sha256_file(selected_manifest_path),
        "query_ids": [query.query_id for query in sampled_queries],
        "strategies": [strategy.value for strategy in selected_strategies],
        "retrievers": [retriever.value for retriever in selected_retrievers],
    }
    run_fingerprint = sha256_text(json.dumps(plan, sort_keys=True))
    output_directory = (
        config.outputs.directory / f"{index_manifest.scope}_{config.benchmark.split.value}"
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    manifest_path = output_directory / "benchmark_manifest.json"
    results_path = output_directory / "retrieval_results.jsonl"
    leaderboard_path = output_directory / "retrieval_leaderboard.csv"
    summary_path = output_directory / "benchmark_summary.md"
    winner_path = output_directory / "winning_configuration.yaml"

    started_at = datetime.now(UTC)
    if resume and manifest_path.is_file():
        existing_manifest = BenchmarkManifest.model_validate_json(
            manifest_path.read_text(encoding="utf-8")
        )
        if existing_manifest.run_fingerprint != run_fingerprint:
            raise EvaluationError(
                "Existing benchmark state has a different plan; use --no-resume to replace it"
            )
        started_at = existing_manifest.started_at
    official_run = (
        index_manifest.scope == "full"
        and set(selected_strategies) == set(config.matrix.chunking_strategies)
        and set(selected_retrievers) == set(config.matrix.retrieval_configurations)
        and chosen_sample_size == config.benchmark.sample_size
    )
    manifest = BenchmarkManifest(
        run_fingerprint=run_fingerprint,
        status="running",
        official=official_run,
        started_at=started_at,
        completed_at=None,
        index_scope=index_manifest.scope,
        split=config.benchmark.split,
        sample_query_ids=tuple(query.query_id for query in sampled_queries),
        strategies=selected_strategies,
        retrievers=selected_retrievers,
        evaluation_config_sha256=sha256_file(config_path),
        retrieval_config_sha256=sha256_file(retrieval_config_path),
        index_manifest_sha256=sha256_file(selected_manifest_path),
    )
    write_text(manifest_path, manifest.model_dump_json(indent=2) + "\n")

    if not resume:
        write_text(results_path, "")
    completed = _load_existing_results(results_path) if resume else {}
    with results_path.open("a", encoding="utf-8") as destination:
        for strategy in selected_strategies:
            for retriever in selected_retrievers:
                pending = [
                    query
                    for query in sampled_queries
                    if (query.query_id, strategy.value, retriever.value) not in completed
                ]
                if not pending:
                    continue
                runtime: EvaluationRuntime | None = None
                try:
                    runtime = runtime_factory(
                        retrieval_config_path,
                        strategy,
                        selected_manifest_path,
                    )
                    runtime.search(
                        config.benchmark.warmup_query,
                        retriever,
                        top_k=config.benchmark.retrieval_depth,
                    )
                    for query in pending:
                        started = perf_counter()
                        try:
                            results = runtime.search(
                                query.question,
                                retriever,
                                top_k=config.benchmark.retrieval_depth,
                            )
                            latency_ms = (perf_counter() - started) * 1000
                            record = _query_record(
                                query,
                                strategy,
                                retriever,
                                relevance[query.query_id],
                                latency_ms,
                                results,
                            )
                        except Exception as error:
                            latency_ms = (perf_counter() - started) * 1000
                            record = _query_record(
                                query,
                                strategy,
                                retriever,
                                relevance[query.query_id],
                                latency_ms,
                                (),
                                error=f"{type(error).__name__}: {error}",
                            )
                        _append_result(destination, record)
                        completed[(query.query_id, strategy.value, retriever.value)] = record
                except Exception as error:
                    message = f"{type(error).__name__}: {error}"
                    for query in pending:
                        record = _query_record(
                            query,
                            strategy,
                            retriever,
                            relevance[query.query_id],
                            0.0,
                            (),
                            error=message,
                        )
                        _append_result(destination, record)
                        completed[(query.query_id, strategy.value, retriever.value)] = record
                finally:
                    if runtime is not None:
                        runtime.close()

    query_ids = [query.query_id for query in sampled_queries]
    leaderboard = _leaderboard(
        completed,
        selected_strategies,
        selected_retrievers,
        query_ids,
    )
    _write_leaderboard(leaderboard_path, leaderboard)
    completed_manifest = manifest.model_copy(
        update={
            "status": (
                "completed_with_failures"
                if any(row.failures for row in leaderboard)
                else "completed"
            ),
            "completed_at": datetime.now(UTC),
        }
    )
    write_text(manifest_path, completed_manifest.model_dump_json(indent=2) + "\n")
    write_text(summary_path, _summary_markdown(completed_manifest, leaderboard))
    winner = leaderboard[0]
    write_text(
        winner_path,
        yaml.safe_dump(
            {
                "eligible_for_configuration_selection": (
                    completed_manifest.official and completed_manifest.split.value == "development"
                ),
                "strategy": winner.strategy.value,
                "retriever": winner.retriever.value,
                "metrics": winner.model_dump(mode="json", exclude={"schema_version"}),
                "run_fingerprint": run_fingerprint,
            },
            sort_keys=False,
        ),
    )
    return BenchmarkResult(
        output_directory=output_directory,
        manifest_path=manifest_path,
        results_path=results_path,
        leaderboard_path=leaderboard_path,
        summary_path=summary_path,
        winner_path=winner_path,
        manifest=completed_manifest,
        leaderboard=leaderboard,
    )
