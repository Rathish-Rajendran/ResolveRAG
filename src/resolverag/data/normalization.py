"""Deterministic text normalization."""

from resolverag.config import NormalizationConfig


def normalize_text(value: str, config: NormalizationConfig) -> str:
    """Normalize source text without destroying meaningful internal spacing."""

    normalized = value
    if config.normalize_newlines:
        normalized = normalized.replace("\r\n", "\n").replace("\r", "\n")
    if config.strip_outer_whitespace:
        normalized = normalized.strip()
    return normalized
