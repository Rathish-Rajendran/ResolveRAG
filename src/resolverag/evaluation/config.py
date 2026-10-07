"""Typed configuration for retrieval benchmarking."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from resolverag.domain.models import (
    ChunkingStrategy,
    LogicalSplit,
    RetrievalConfiguration,
)
from resolverag.exceptions import ConfigurationError


class StrictEvaluationConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class EvaluationInputs(StrictEvaluationConfigModel):
    queries_file: Path
    qrels_file: Path
    retrieval_config_file: Path


class BenchmarkSettings(StrictEvaluationConfigModel):
    split: LogicalSplit
    sample_size: int | None = Field(default=None, gt=0)
    random_seed: int
    length_strata: int = Field(gt=0)
    retrieval_depth: int = Field(gt=0)
    cutoffs: list[int] = Field(min_length=1)
    warmup_query: str = Field(min_length=1)

    @model_validator(mode="after")
    def validate_cutoffs(self) -> "BenchmarkSettings":
        if any(cutoff <= 0 for cutoff in self.cutoffs):
            raise ValueError("Evaluation cutoffs must be greater than zero")
        if len(set(self.cutoffs)) != len(self.cutoffs):
            raise ValueError("Evaluation cutoffs must be unique")
        if set(self.cutoffs) != {1, 5, 10}:
            raise ValueError("ResolveRAG retrieval evaluation requires cutoffs 1, 5, and 10")
        if max(self.cutoffs) > self.retrieval_depth:
            raise ValueError("retrieval_depth must be at least the largest cutoff")
        return self


class EvaluationMatrix(StrictEvaluationConfigModel):
    chunking_strategies: list[ChunkingStrategy] = Field(min_length=1)
    retrieval_configurations: list[RetrievalConfiguration] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_unique_matrix(self) -> "EvaluationMatrix":
        if len(set(self.chunking_strategies)) != len(self.chunking_strategies):
            raise ValueError("Chunking strategies must be unique")
        if len(set(self.retrieval_configurations)) != len(self.retrieval_configurations):
            raise ValueError("Retrieval configurations must be unique")
        return self


class EvaluationOutputs(StrictEvaluationConfigModel):
    directory: Path


class EvaluationConfig(StrictEvaluationConfigModel):
    schema_version: int = Field(ge=1)
    inputs: EvaluationInputs
    benchmark: BenchmarkSettings
    matrix: EvaluationMatrix
    outputs: EvaluationOutputs


def load_evaluation_config(path: Path) -> EvaluationConfig:
    """Load one retrieval-benchmark configuration."""

    if not path.is_file():
        raise ConfigurationError(f"Evaluation configuration does not exist: {path}")
    try:
        raw_config: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return EvaluationConfig.model_validate(raw_config)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(f"Invalid evaluation configuration {path}: {error}") from error
