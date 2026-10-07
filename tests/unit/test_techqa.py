import json
from copy import deepcopy
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pytest
import yaml
from typer.testing import CliRunner

from resolverag.cli import app
from resolverag.config import load_dataset_config
from resolverag.data.downloader import download_source_files, validate_source_files
from resolverag.data.integrity import sha256_file
from resolverag.data.techqa import prepare_dataset
from resolverag.exceptions import DatasetValidationError


def _answerable_row() -> dict[str, object]:
    return {
        "id": "TRAIN_Q001",
        "question": "  How do I fix the service?\r\n",
        "answer": "Restart it.",
        "is_impossible": False,
        "contexts": [{"filename": "doc-a.txt", "text": "Document A"}],
    }


def _unanswerable_row() -> dict[str, object]:
    return {
        "id": "DEV_Q001",
        "question": "What is unsupported?",
        "answer": "-",
        "is_impossible": True,
        "contexts": [],
    }


def _config(
    tmp_path: Path,
    rows: list[dict[str, object]],
    *,
    documents: dict[str, str] | None = None,
) -> tuple[Path, Path, Path, dict[str, Any]]:
    raw_directory = tmp_path / "raw"
    processed_directory = tmp_path / "processed"
    raw_directory.mkdir()
    question_path = raw_directory / "train.json"
    archive_path = raw_directory / "corpus.zip"
    question_path.write_text(json.dumps(rows), encoding="utf-8")
    with ZipFile(archive_path, "w") as archive:
        for document_id, text in (documents or {"doc-a.txt": " Document A\r\n"}).items():
            archive.writestr(f"corpus/{document_id}", text)

    config_data: dict[str, Any] = {
        "schema_version": 1,
        "dataset": {
            "name": "techqa_test",
            "provider": "huggingface",
            "publisher": "NVIDIA",
            "repo_id": "nvidia/TechQA-RAG-Eval",
            "revision": "test-revision",
            "license": "Apache-2.0",
            "homepage": "https://example.test",
        },
        "upstream": {
            "name": "TechQA",
            "owner": "IBM Research",
            "repository": "https://example.test/repository",
            "publication": "https://example.test/publication",
        },
        "source_files": {
            "questions": "train.json",
            "corpus_archive": "corpus.zip",
        },
        "source_checksums": {
            "train.json": sha256_file(question_path),
            "corpus.zip": sha256_file(archive_path),
        },
        "storage": {
            "raw_directory": str(raw_directory),
            "interim_directory": str(tmp_path / "interim"),
            "processed_directory": str(processed_directory),
            "manifest_file": str(processed_directory / "manifest.json"),
        },
        "source_schema": {
            "question_id": "id",
            "question": "question",
            "reference_answer": "answer",
            "is_unanswerable": "is_impossible",
            "contexts": "contexts",
            "context": {"document_id": "filename", "text": "text"},
        },
        "logical_splits": {
            "development": {"id_prefix": "TRAIN_", "purpose": "development"},
            "held_out": {"id_prefix": "DEV_", "purpose": "evaluation"},
            "unknown_prefix_policy": "error",
        },
        "normalization": {
            "strip_outer_whitespace": True,
            "normalize_newlines": True,
            "reject_empty_document_ids": True,
            "unanswerable": {
                "source_answer": "-",
                "canonical_answer": None,
                "canonical_relevant_documents": [],
            },
        },
        "validation": {
            "expected_question_rows": len(rows),
            "required_question_fields": [
                "id",
                "question",
                "answer",
                "is_impossible",
                "contexts",
            ],
            "unique_fields": ["id"],
            "fail_on": [],
        },
        "provenance": {
            "preserve_raw_files": True,
            "checksum_algorithm": "sha256",
            "record_download_timestamp": True,
            "record_library_versions": True,
        },
    }
    config_path = tmp_path / "techqa.yaml"
    config_path.write_text(yaml.safe_dump(config_data), encoding="utf-8")
    return config_path, question_path, archive_path, config_data


def test_prepare_dataset_creates_separate_canonical_artifacts(tmp_path: Path) -> None:
    rows = [_answerable_row(), _unanswerable_row()]
    config_path, _, _, _ = _config(tmp_path, rows)

    result = prepare_dataset(config_path, download=False)

    assert result.manifest.counts.documents == 1
    assert result.manifest.counts.queries == 2
    assert result.manifest.counts.relevance_judgments == 1
    assert result.manifest.counts.development_queries == 1
    assert result.manifest.counts.held_out_queries == 1
    assert result.manifest.counts.answerable_queries_without_reference_answer == 0
    document = json.loads(result.documents_path.read_text(encoding="utf-8"))
    assert document["text"] == "Document A"
    queries = [
        json.loads(line) for line in result.queries_path.read_text(encoding="utf-8").splitlines()
    ]
    assert all("contexts" not in query for query in queries)
    assert {query["question"] for query in queries} == {
        "How do I fix the service?",
        "What is unsupported?",
    }
    assert json.loads(result.qrels_path.read_text(encoding="utf-8"))["document_id"] == ("doc-a.txt")
    assert len(result.manifest.artifacts) == 3
    assert result.manifest_path.is_file()


def test_download_reuses_verified_local_files(tmp_path: Path) -> None:
    config_path, question_path, archive_path, _ = _config(tmp_path, [_answerable_row()])
    config = load_dataset_config(config_path)

    assert download_source_files(config) == (question_path, archive_path)


def test_validate_source_files_rejects_checksum_mismatch(tmp_path: Path) -> None:
    config_path, question_path, _, _ = _config(tmp_path, [_answerable_row()])
    config = load_dataset_config(config_path)
    question_path.write_text("corrupt", encoding="utf-8")

    with pytest.raises(DatasetValidationError, match="SHA-256 mismatch"):
        validate_source_files(config)


def test_prepare_rejects_duplicate_question_id(tmp_path: Path) -> None:
    row = _answerable_row()
    config_path, _, _, _ = _config(tmp_path, [row, deepcopy(row)])

    with pytest.raises(DatasetValidationError, match="Duplicate question ID"):
        prepare_dataset(config_path, download=False)


def test_prepare_rejects_answerable_question_without_context(tmp_path: Path) -> None:
    row = _answerable_row()
    row["contexts"] = []
    config_path, _, _, _ = _config(tmp_path, [row])

    with pytest.raises(DatasetValidationError, match="does not have a context"):
        prepare_dataset(config_path, download=False)


def test_prepare_rejects_unknown_split(tmp_path: Path) -> None:
    row = _answerable_row()
    row["id"] = "TEST_Q001"
    config_path, _, _, _ = _config(tmp_path, [row])

    with pytest.raises(DatasetValidationError, match="unknown logical split"):
        prepare_dataset(config_path, download=False)


def test_prepare_rejects_reference_to_missing_document(tmp_path: Path) -> None:
    row = _answerable_row()
    row["contexts"] = [{"filename": "missing.txt", "text": "Missing"}]
    config_path, _, _, _ = _config(tmp_path, [row])

    with pytest.raises(DatasetValidationError, match="refers to missing document"):
        prepare_dataset(config_path, download=False)


def test_prepare_rejects_unanswerable_question_with_context(tmp_path: Path) -> None:
    row = _unanswerable_row()
    row["contexts"] = [{"filename": "doc-a.txt", "text": "Document A"}]
    config_path, _, _, _ = _config(tmp_path, [row])

    with pytest.raises(DatasetValidationError, match="unexpectedly has contexts"):
        prepare_dataset(config_path, download=False)


def test_prepare_rejects_missing_required_field(tmp_path: Path) -> None:
    row = _answerable_row()
    del row["question"]
    config_path, _, _, _ = _config(tmp_path, [row])

    with pytest.raises(DatasetValidationError, match="missing required field"):
        prepare_dataset(config_path, download=False)


def test_cli_reports_success_and_failure(tmp_path: Path) -> None:
    config_path, question_path, _, _ = _config(tmp_path, [_answerable_row()])
    runner = CliRunner()

    success = runner.invoke(
        app,
        ["dataset", "prepare", "--config", str(config_path), "--no-download"],
    )
    assert success.exit_code == 0
    assert "TechQA dataset prepared" in success.stdout

    question_path.write_text("corrupt", encoding="utf-8")
    failure = runner.invoke(
        app,
        ["dataset", "prepare", "--config", str(config_path), "--no-download"],
    )
    assert failure.exit_code == 1
    assert "Dataset preparation failed" in failure.stdout
