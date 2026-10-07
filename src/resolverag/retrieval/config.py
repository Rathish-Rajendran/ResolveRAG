"""Typed retrieval configuration."""

from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, ValidationError, model_validator

from resolverag.exceptions import ConfigurationError


class StrictRetrievalConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class RetrievalInputs(StrictRetrievalConfigModel):
    index_config_file: Path
    index_manifest_file: Path


class RetrievalSettings(StrictRetrievalConfigModel):
    default_top_k: int = Field(gt=0)
    dense_candidate_k: int = Field(gt=0)
    sparse_candidate_k: int = Field(gt=0)
    hybrid_candidate_k: int = Field(gt=0)
    mmr_lambda: float = Field(ge=0.0, le=1.0)
    reciprocal_rank_constant: int = Field(gt=0)
    dense_weight: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_candidate_depths(self) -> "RetrievalSettings":
        if self.dense_candidate_k < self.default_top_k:
            raise ValueError("dense_candidate_k must be at least default_top_k")
        if self.sparse_candidate_k < self.default_top_k:
            raise ValueError("sparse_candidate_k must be at least default_top_k")
        if self.hybrid_candidate_k < self.default_top_k:
            raise ValueError("hybrid_candidate_k must be at least default_top_k")
        return self


class RerankerSettings(StrictRetrievalConfigModel):
    model: str = Field(min_length=1)
    batch_size: int = Field(gt=0)


class RetrievalConfig(StrictRetrievalConfigModel):
    schema_version: int = Field(ge=1)
    inputs: RetrievalInputs
    retrieval: RetrievalSettings
    reranker: RerankerSettings


def load_retrieval_config(path: Path) -> RetrievalConfig:
    """Load the six-strategy retrieval configuration."""

    if not path.is_file():
        raise ConfigurationError(f"Retrieval configuration does not exist: {path}")
    try:
        raw_config: object = yaml.safe_load(path.read_text(encoding="utf-8"))
        return RetrievalConfig.model_validate(raw_config)
    except (OSError, yaml.YAMLError, ValidationError) as error:
        raise ConfigurationError(f"Invalid retrieval configuration {path}: {error}") from error
