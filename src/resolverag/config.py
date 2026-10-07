"""Typed configuration loading for ResolveRAG."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from resolverag.exceptions import ConfigurationError


class StrictConfigModel(BaseModel):
    """Reject unknown settings so configuration drift fails loudly."""

    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetIdentity(StrictConfigModel):
    name: str
    provider: str
    publisher: str
    repo_id: str
    revision: str
    license: str
    homepage: str


class UpstreamIdentity(StrictConfigModel):
    name: str
    owner: str
    repository: str
    publication: str


class SourceFiles(StrictConfigModel):
    questions: str
    corpus_archive: str


class StorageConfig(StrictConfigModel):
    raw_directory: Path
    interim_directory: Path
    processed_directory: Path
    manifest_file: Path


class ContextSchema(StrictConfigModel):
    document_id: str
    text: str


class SourceSchema(StrictConfigModel):
    question_id: str
    question: str
    reference_answer: str
    is_unanswerable: str
    contexts: str
    context: ContextSchema


class LogicalSplitConfig(StrictConfigModel):
    id_prefix: str
    purpose: str


class LogicalSplitsConfig(StrictConfigModel):
    development: LogicalSplitConfig
    held_out: LogicalSplitConfig
    unknown_prefix_policy: str


class UnanswerableConfig(StrictConfigModel):
    source_answer: str
    canonical_answer: None = None
    canonical_relevant_documents: list[str] = Field(default_factory=list)


class NormalizationConfig(StrictConfigModel):
    strip_outer_whitespace: bool
    normalize_newlines: bool
    reject_empty_document_ids: bool
    unanswerable: UnanswerableConfig


class ValidationConfig(StrictConfigModel):
    expected_question_rows: int = Field(ge=0)
    required_question_fields: list[str]
    unique_fields: list[str]
    fail_on: list[str]


class ProvenanceConfig(StrictConfigModel):
    preserve_raw_files: bool
    checksum_algorithm: str
    record_download_timestamp: bool
    record_library_versions: bool


class DatasetConfig(StrictConfigModel):
    schema_version: int = Field(ge=1)
    dataset: DatasetIdentity
    upstream: UpstreamIdentity
    source_files: SourceFiles
    source_checksums: dict[str, str]
    storage: StorageConfig
    source_schema: SourceSchema
    logical_splits: LogicalSplitsConfig
    normalization: NormalizationConfig
    validation: ValidationConfig
    provenance: ProvenanceConfig


def load_dataset_config(path: Path) -> DatasetConfig:
    """Load and validate one dataset configuration file."""

    if not path.is_file():
        raise ConfigurationError(f"Dataset configuration does not exist: {path}")

    try:
        raw_config: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return DatasetConfig.model_validate(raw_config)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(f"Invalid dataset configuration {path}: {error}") from error
