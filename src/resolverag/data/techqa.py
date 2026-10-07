"""TechQA-specific validation and canonicalization pipeline."""

import json
import subprocess
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path, PurePosixPath
from typing import cast
from zipfile import BadZipFile, ZipFile

from pydantic import BaseModel

from resolverag.config import DatasetConfig, load_dataset_config
from resolverag.data.downloader import download_source_files, validate_source_files
from resolverag.data.integrity import sha256_file, sha256_text
from resolverag.data.normalization import normalize_text
from resolverag.data.serialization import write_jsonl, write_text
from resolverag.domain.models import (
    ArtifactManifest,
    DatasetCounts,
    DatasetManifest,
    EvaluationQuery,
    LogicalSplit,
    RelevanceJudgment,
    SourceDocument,
    SourceFileManifest,
)
from resolverag.exceptions import DatasetValidationError

TRANSFORMATION_VERSION = "techqa-canonical-v1"


@dataclass(frozen=True)
class PreparationResult:
    """Paths and provenance returned by a successful dataset build."""

    documents_path: Path
    queries_path: Path
    qrels_path: Path
    manifest_path: Path
    manifest: DatasetManifest


@dataclass(frozen=True)
class _CorpusResult:
    documents: tuple[SourceDocument, ...]
    duplicate_count: int


def _required_field(record: Mapping[str, object], field: str, record_id: str) -> object:
    try:
        return record[field]
    except KeyError as error:
        raise DatasetValidationError(
            f"Record {record_id!r} is missing required field {field!r}"
        ) from error


def _required_string(record: Mapping[str, object], field: str, record_id: str) -> str:
    value = _required_field(record, field, record_id)
    if not isinstance(value, str):
        raise DatasetValidationError(f"Field {field!r} in record {record_id!r} must be a string")
    return value


def _load_corpus(config: DatasetConfig, archive_path: Path) -> _CorpusResult:
    documents_by_id: dict[str, SourceDocument] = {}
    duplicate_count = 0
    try:
        with ZipFile(archive_path) as archive:
            for member in archive.infolist():
                member_path = PurePosixPath(member.filename)
                if member.is_dir():
                    continue
                if member_path.is_absolute() or ".." in member_path.parts:
                    raise DatasetValidationError(
                        f"Unsafe path in corpus archive: {member.filename!r}"
                    )
                if member_path.suffix.lower() != ".txt":
                    raise DatasetValidationError(
                        f"Unexpected non-text file in corpus archive: {member.filename!r}"
                    )

                document_id = member_path.name.strip()
                if config.normalization.reject_empty_document_ids and not document_id:
                    raise DatasetValidationError("Corpus contains an empty document identifier")
                raw_text = archive.read(member).decode("utf-8")
                text = normalize_text(raw_text, config.normalization)
                if not text:
                    raise DatasetValidationError(f"Document {document_id!r} has empty text")
                document = SourceDocument(
                    document_id=document_id,
                    text=text,
                    source_path=member.filename,
                    text_sha256=sha256_text(text),
                )
                existing = documents_by_id.get(document_id)
                if existing is None:
                    documents_by_id[document_id] = document
                elif existing.text_sha256 == document.text_sha256:
                    duplicate_count += 1
                else:
                    raise DatasetValidationError(
                        f"Conflicting corpus entries share document ID {document_id!r}"
                    )
    except (BadZipFile, UnicodeDecodeError, OSError) as error:
        raise DatasetValidationError(
            f"Unable to read corpus archive {archive_path}: {error}"
        ) from error

    if not documents_by_id:
        raise DatasetValidationError("Corpus archive contains no documents")
    documents = tuple(sorted(documents_by_id.values(), key=lambda item: item.document_id))
    return _CorpusResult(documents=documents, duplicate_count=duplicate_count)


def _logical_split(query_id: str, config: DatasetConfig) -> LogicalSplit:
    if query_id.startswith(config.logical_splits.development.id_prefix):
        return LogicalSplit.DEVELOPMENT
    if query_id.startswith(config.logical_splits.held_out.id_prefix):
        return LogicalSplit.HELD_OUT
    raise DatasetValidationError(f"Question {query_id!r} has an unknown logical split")


def _load_question_rows(path: Path, expected_rows: int) -> list[Mapping[str, object]]:
    try:
        decoded: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise DatasetValidationError(f"Unable to read question file {path}: {error}") from error
    if not isinstance(decoded, list):
        raise DatasetValidationError("Question source must contain a JSON array")
    if len(decoded) != expected_rows:
        raise DatasetValidationError(
            f"Expected {expected_rows} question rows but observed {len(decoded)}"
        )
    rows: list[Mapping[str, object]] = []
    for position, value in enumerate(decoded):
        if not isinstance(value, dict):
            raise DatasetValidationError(f"Question row {position} must be a JSON object")
        rows.append(cast(Mapping[str, object], value))
    return rows


def _load_queries(
    config: DatasetConfig,
    question_path: Path,
    document_ids: set[str],
) -> tuple[tuple[EvaluationQuery, ...], tuple[RelevanceJudgment, ...]]:
    rows = _load_question_rows(question_path, config.validation.expected_question_rows)
    schema = config.source_schema
    queries: list[EvaluationQuery] = []
    qrels: list[RelevanceJudgment] = []
    observed_query_ids: set[str] = set()
    observed_qrels: set[tuple[str, str]] = set()

    for position, row in enumerate(rows):
        provisional_id = f"row {position}"
        for required_field in config.validation.required_question_fields:
            _required_field(row, required_field, provisional_id)

        query_id = _required_string(row, schema.question_id, provisional_id).strip()
        if not query_id:
            raise DatasetValidationError(f"Question row {position} has an empty ID")
        if query_id in observed_query_ids:
            raise DatasetValidationError(f"Duplicate question ID: {query_id!r}")
        observed_query_ids.add(query_id)

        question = normalize_text(
            _required_string(row, schema.question, query_id), config.normalization
        )
        if not question:
            raise DatasetValidationError(f"Question {query_id!r} has empty text")
        source_answer = normalize_text(
            _required_string(row, schema.reference_answer, query_id), config.normalization
        )
        unanswerable_value = _required_field(row, schema.is_unanswerable, query_id)
        if not isinstance(unanswerable_value, bool):
            raise DatasetValidationError(
                f"Field {schema.is_unanswerable!r} in {query_id!r} must be a boolean"
            )
        contexts_value = _required_field(row, schema.contexts, query_id)
        if not isinstance(contexts_value, list):
            raise DatasetValidationError(f"Contexts in {query_id!r} must be a list")

        if unanswerable_value:
            if contexts_value:
                raise DatasetValidationError(
                    f"Unanswerable question {query_id!r} unexpectedly has contexts"
                )
            if source_answer != config.normalization.unanswerable.source_answer:
                raise DatasetValidationError(
                    f"Unanswerable question {query_id!r} has unexpected source answer"
                )
            reference_answer: str | None = None
        else:
            if not contexts_value:
                raise DatasetValidationError(
                    f"Answerable question {query_id!r} does not have a context"
                )
            reference_answer = source_answer or None

        queries.append(
            EvaluationQuery(
                query_id=query_id,
                question=question,
                reference_answer=reference_answer,
                is_answerable=not unanswerable_value,
                split=_logical_split(query_id, config),
            )
        )

        for context_position, context_value in enumerate(contexts_value):
            if not isinstance(context_value, dict):
                raise DatasetValidationError(
                    f"Context {context_position} in {query_id!r} must be an object"
                )
            context = cast(Mapping[str, object], context_value)
            document_id = _required_string(context, schema.context.document_id, query_id).strip()
            if not document_id:
                raise DatasetValidationError(f"Question {query_id!r} has an empty document ID")
            _required_string(context, schema.context.text, query_id)
            if document_id not in document_ids:
                raise DatasetValidationError(
                    f"Question {query_id!r} refers to missing document {document_id!r}"
                )
            judgment_key = (query_id, document_id)
            if judgment_key not in observed_qrels:
                qrels.append(RelevanceJudgment(query_id=query_id, document_id=document_id))
                observed_qrels.add(judgment_key)

    return (
        tuple(sorted(queries, key=lambda item: item.query_id)),
        tuple(sorted(qrels, key=lambda item: (item.query_id, item.document_id))),
    )


def _package_versions() -> dict[str, str]:
    packages = ("datasets", "huggingface-hub", "pydantic", "pyyaml")
    versions = {"python": sys.version.split()[0]}
    for package in packages:
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "not-installed"
    return versions


def _git_state() -> tuple[str | None, bool | None]:
    try:
        revision_result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        status_result = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, None
    return revision_result.stdout.strip(), bool(status_result.stdout.strip())


def _artifact(path: Path, record_count: int) -> ArtifactManifest:
    return ArtifactManifest(
        filename=path.name,
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
        records=record_count,
    )


def _source_manifest(path: Path) -> SourceFileManifest:
    return SourceFileManifest(
        filename=path.name,
        byte_size=path.stat().st_size,
        sha256=sha256_file(path),
    )


def _write_records(path: Path, records: tuple[BaseModel, ...]) -> None:
    write_jsonl(path, records)


def prepare_dataset(config_path: Path, *, download: bool = True) -> PreparationResult:
    """Download, validate, normalize, and serialize the canonical TechQA dataset."""

    config = load_dataset_config(config_path)
    if config.provenance.checksum_algorithm.lower() != "sha256":
        raise DatasetValidationError("Only SHA-256 provenance checksums are supported")
    if download:
        download_source_files(config)
    question_path, archive_path = validate_source_files(config)

    corpus = _load_corpus(config, archive_path)
    document_ids = {document.document_id for document in corpus.documents}
    queries, qrels = _load_queries(config, question_path, document_ids)

    processed_directory = config.storage.processed_directory
    documents_path = processed_directory / "documents.jsonl"
    queries_path = processed_directory / "queries.jsonl"
    qrels_path = processed_directory / "qrels.jsonl"
    _write_records(documents_path, corpus.documents)
    _write_records(queries_path, queries)
    _write_records(qrels_path, qrels)

    answerable_count = sum(query.is_answerable for query in queries)
    development_count = sum(query.split is LogicalSplit.DEVELOPMENT for query in queries)
    counts = DatasetCounts(
        source_question_rows=len(queries),
        documents=len(corpus.documents),
        queries=len(queries),
        relevance_judgments=len(qrels),
        answerable_queries=answerable_count,
        unanswerable_queries=len(queries) - answerable_count,
        answerable_queries_without_reference_answer=sum(
            query.is_answerable and query.reference_answer is None for query in queries
        ),
        development_queries=development_count,
        held_out_queries=len(queries) - development_count,
        duplicate_documents_removed=corpus.duplicate_count,
    )
    code_revision, working_tree_dirty = _git_state()
    manifest = DatasetManifest(
        dataset_name=config.dataset.name,
        repository_id=config.dataset.repo_id,
        repository_revision=config.dataset.revision,
        transformation_version=TRANSFORMATION_VERSION,
        generated_at=datetime.now(UTC),
        configuration_sha256=sha256_file(config_path),
        source_files=tuple(_source_manifest(path) for path in (question_path, archive_path)),
        artifacts=(
            _artifact(documents_path, len(corpus.documents)),
            _artifact(queries_path, len(queries)),
            _artifact(qrels_path, len(qrels)),
        ),
        counts=counts,
        library_versions=_package_versions(),
        code_revision=code_revision,
        working_tree_dirty=working_tree_dirty,
    )
    write_text(
        config.storage.manifest_file,
        manifest.model_dump_json(indent=2, exclude_none=False) + "\n",
    )
    return PreparationResult(
        documents_path=documents_path,
        queries_path=queries_path,
        qrels_path=qrels_path,
        manifest_path=config.storage.manifest_file,
        manifest=manifest,
    )
