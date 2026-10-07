"""In-memory BM25 retrieval over one canonical chunk artifact."""

import re
from pathlib import Path
from typing import cast

import numpy as np
import numpy.typing as npt
from pydantic import ValidationError
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from resolverag.data.integrity import sha256_file
from resolverag.domain.models import (
    Chunk,
    RetrievalConfiguration,
    RetrievalResult,
)
from resolverag.exceptions import RetrievalError

_TERM = re.compile(r"[a-z0-9]+(?:[._/-][a-z0-9]+)*")


def tokenize(value: str) -> list[str]:
    """Tokenize technical prose while preserving common identifier punctuation."""

    return _TERM.findall(value.lower())


class BM25Retriever:
    """Rank chunks using Okapi BM25 lexical relevance."""

    def __init__(self, chunks_path: Path, expected_sha256: str) -> None:
        observed_sha256 = sha256_file(chunks_path)
        if observed_sha256 != expected_sha256:
            raise RetrievalError(
                f"Chunk artifact checksum mismatch for {chunks_path}: "
                f"expected {expected_sha256}, observed {observed_sha256}"
            )
        chunks: list[Chunk] = []
        try:
            with chunks_path.open(encoding="utf-8") as source:
                for line_number, line in enumerate(source, start=1):
                    try:
                        chunks.append(Chunk.model_validate_json(line))
                    except ValidationError as error:
                        raise RetrievalError(
                            f"Invalid chunk at {chunks_path}:{line_number}: {error}"
                        ) from error
        except OSError as error:
            raise RetrievalError(f"Unable to load chunk artifact {chunks_path}: {error}") from error
        if not chunks:
            raise RetrievalError(f"Chunk artifact is empty: {chunks_path}")
        self._chunks = chunks
        self._index = BM25Okapi([tokenize(chunk.text) for chunk in chunks])

    def retrieve(self, query: str, top_k: int) -> tuple[RetrievalResult, ...]:
        if top_k <= 0:
            raise RetrievalError("top_k must be greater than zero")
        query_terms = tokenize(query)
        if not query_terms:
            return ()
        scores = cast(npt.NDArray[np.float64], self._index.get_scores(query_terms))
        result_count = min(top_k, len(self._chunks))
        if result_count == len(self._chunks):
            candidate_indices = np.arange(len(self._chunks))
        else:
            candidate_indices = np.argpartition(scores, -result_count)[-result_count:]
        ranked_indices = sorted(
            candidate_indices.tolist(), key=lambda index: scores[index], reverse=True
        )
        return tuple(
            RetrievalResult(
                retriever=RetrievalConfiguration.BM25,
                rank=rank,
                score=float(scores[index]),
                chunk=self._chunks[index],
                component_scores={"bm25": float(scores[index])},
            )
            for rank, index in enumerate(ranked_indices, start=1)
        )
