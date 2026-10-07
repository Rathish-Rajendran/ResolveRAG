from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import NAMESPACE_URL, uuid5

import yaml

from resolverag.data.integrity import sha256_text
from resolverag.domain.models import (
    ArtifactManifest,
    Chunk,
    ChunkingStrategy,
    EvaluationQuery,
    LogicalSplit,
    RelevanceJudgment,
    RetrievalConfiguration,
    RetrievalResult,
)
from resolverag.evaluation.runner import run_retrieval_benchmark
from resolverag.indexing.models import (
    ChunkingStatistics,
    CollectionBuild,
    IndexBuildManifest,
)


class FakeEvaluationRuntime:
    def __init__(self, strategy: ChunkingStrategy) -> None:
        self.strategy = strategy
        self.closed = False

    def search(
        self,
        query: str,
        configuration: RetrievalConfiguration,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievalResult, ...]:
        document_id = query.rsplit(" ", maxsplit=1)[-1]
        if not document_id.endswith(".txt"):
            document_id = "warmup.txt"
        chunk = Chunk(
            chunk_id=str(uuid5(NAMESPACE_URL, f"{self.strategy}:{document_id}")),
            document_id=document_id,
            source_path=f"corpus/{document_id}",
            strategy=self.strategy,
            chunk_index=0,
            text=f"Evidence from {document_id}",
            context_text=f"Evidence from {document_id}",
            text_sha256=sha256_text(document_id),
            token_count=4,
        )
        return (
            RetrievalResult(
                retriever=configuration,
                rank=1,
                score=1.0,
                chunk=chunk,
                component_scores={"fake": 1.0},
            ),
        )

    def close(self) -> None:
        self.closed = True


def _collection(strategy: ChunkingStrategy) -> CollectionBuild:
    artifact = ArtifactManifest(
        filename=f"chunks/{strategy.value}.jsonl",
        byte_size=1,
        sha256="1" * 64,
        records=4,
    )
    statistics = ChunkingStatistics(
        strategy=strategy,
        documents=4,
        chunks=4,
        chunks_per_document=1.0,
        minimum_tokens=4,
        mean_tokens=4.0,
        p50_tokens=4,
        p95_tokens=4,
        maximum_tokens=4,
    )
    return CollectionBuild(
        strategy=strategy,
        collection_name=f"test_{strategy.value}",
        vector_count=4,
        build_seconds=1.0,
        chunk_artifact=artifact,
        statistics=statistics,
    )


def _benchmark_config(tmp_path: Path) -> tuple[Path, Path]:
    queries = [
        EvaluationQuery(
            query_id=f"TRAIN_Q{index:03}",
            question=f"Find doc-{index}.txt",
            reference_answer="answer",
            is_answerable=True,
            split=LogicalSplit.DEVELOPMENT,
        )
        for index in range(4)
    ]
    qrels = [
        RelevanceJudgment(query_id=query.query_id, document_id=f"doc-{index}.txt")
        for index, query in enumerate(queries)
    ]
    queries_path = tmp_path / "queries.jsonl"
    qrels_path = tmp_path / "qrels.jsonl"
    queries_path.write_text(
        "".join(query.model_dump_json() + "\n" for query in queries),
        encoding="utf-8",
    )
    qrels_path.write_text(
        "".join(judgment.model_dump_json() + "\n" for judgment in qrels),
        encoding="utf-8",
    )
    strategies = (ChunkingStrategy.FIXED_256, ChunkingStrategy.RECURSIVE)
    index_manifest = IndexBuildManifest(
        generated_at=datetime.now(UTC),
        scope="smoke_4",
        document_limit=4,
        configuration_sha256="2" * 64,
        dataset_manifest_sha256="3" * 64,
        documents_sha256="4" * 64,
        embedding_provider="ollama",
        embedding_model="fake",
        embedding_dimensions=3,
        qdrant_path=str(tmp_path / "qdrant"),
        collections=tuple(_collection(strategy) for strategy in strategies),
    )
    index_manifest_path = tmp_path / "index-manifest.json"
    index_manifest_path.write_text(index_manifest.model_dump_json(), encoding="utf-8")
    retrieval_config_path = tmp_path / "retrieval.yaml"
    retrieval_config_data: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "index_config_file": str(tmp_path / "index.yaml"),
            "index_manifest_file": str(index_manifest_path),
        },
        "retrieval": {
            "default_top_k": 10,
            "dense_candidate_k": 50,
            "sparse_candidate_k": 50,
            "hybrid_candidate_k": 50,
            "mmr_lambda": 0.65,
            "reciprocal_rank_constant": 60,
            "dense_weight": 0.6,
        },
        "reranker": {"model": "fake", "batch_size": 2},
    }
    retrieval_config_path.write_text(
        yaml.safe_dump(retrieval_config_data),
        encoding="utf-8",
    )
    evaluation_config_data: dict[str, Any] = {
        "schema_version": 1,
        "inputs": {
            "queries_file": str(queries_path),
            "qrels_file": str(qrels_path),
            "retrieval_config_file": str(retrieval_config_path),
        },
        "benchmark": {
            "split": "development",
            "sample_size": 4,
            "random_seed": 42,
            "length_strata": 2,
            "retrieval_depth": 10,
            "cutoffs": [1, 5, 10],
            "warmup_query": "warmup query",
        },
        "matrix": {
            "chunking_strategies": [strategy.value for strategy in strategies],
            "retrieval_configurations": ["bm25", "dense"],
        },
        "outputs": {"directory": str(tmp_path / "reports")},
    }
    evaluation_config_path = tmp_path / "evaluation.yaml"
    evaluation_config_path.write_text(
        yaml.safe_dump(evaluation_config_data),
        encoding="utf-8",
    )
    return evaluation_config_path, index_manifest_path


def test_benchmark_generates_leaderboard_and_resumes(tmp_path: Path) -> None:
    config_path, manifest_path = _benchmark_config(tmp_path)
    factory_calls: list[tuple[ChunkingStrategy, Path]] = []

    def factory(
        _config_path: Path,
        strategy: ChunkingStrategy,
        index_manifest_path: Path,
    ) -> FakeEvaluationRuntime:
        factory_calls.append((strategy, index_manifest_path))
        return FakeEvaluationRuntime(strategy)

    result = run_retrieval_benchmark(
        config_path,
        index_manifest_path=manifest_path,
        runtime_factory=factory,
    )

    assert len(result.leaderboard) == 4
    assert all(row.recall_at_1 == 1.0 for row in result.leaderboard)
    assert all(row.failures == 0 for row in result.leaderboard)
    assert result.manifest.status == "completed"
    assert not result.manifest.official
    assert len(result.results_path.read_text(encoding="utf-8").splitlines()) == 16
    assert "VALIDATION ONLY" in result.summary_path.read_text(encoding="utf-8")
    assert len(factory_calls) == 4

    factory_calls.clear()
    resumed = run_retrieval_benchmark(
        config_path,
        index_manifest_path=manifest_path,
        runtime_factory=factory,
    )

    assert resumed.manifest.run_fingerprint == result.manifest.run_fingerprint
    assert not factory_calls
