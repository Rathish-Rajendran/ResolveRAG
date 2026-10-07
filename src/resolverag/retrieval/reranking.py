"""Cross-encoder reranking for hybrid retrieval candidates."""

from typing import cast

import numpy as np
import numpy.typing as npt
from sentence_transformers import CrossEncoder

from resolverag.domain.models import RetrievalConfiguration, RetrievalResult
from resolverag.exceptions import RetrievalError
from resolverag.retrieval.base import PassageReranker, Retriever


class CrossEncoderReranker:
    """Adapter around an open-source sentence-transformers cross-encoder."""

    def __init__(self, model: str, batch_size: int) -> None:
        self._model = CrossEncoder(model)
        self._batch_size = batch_size

    def score(self, query: str, passages: list[str]) -> list[float]:
        if not passages:
            return []
        predictions = self._model.predict(
            [(query, passage) for passage in passages],
            batch_size=self._batch_size,
            show_progress_bar=False,
        )
        values = cast(npt.NDArray[np.float32], np.asarray(predictions))
        return [float(value) for value in values.reshape(-1)]


class RerankingRetriever:
    """Rerank a hybrid candidate pool with query-passage relevance scores."""

    def __init__(
        self,
        base: Retriever,
        reranker: PassageReranker,
        *,
        candidate_k: int,
    ) -> None:
        self._base = base
        self._reranker = reranker
        self._candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int) -> tuple[RetrievalResult, ...]:
        candidates = self._base.retrieve(query, self._candidate_k)
        scores = self._reranker.score(query, [result.chunk.text for result in candidates])
        if len(scores) != len(candidates):
            raise RetrievalError(
                f"Reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )
        scored = sorted(
            zip(candidates, scores, strict=True),
            key=lambda item: item[1],
            reverse=True,
        )
        output: list[RetrievalResult] = []
        for rank, (result, score) in enumerate(scored[:top_k], start=1):
            components = dict(result.component_scores)
            components["cross_encoder"] = score
            output.append(
                result.model_copy(
                    update={
                        "retriever": RetrievalConfiguration.HYBRID_RERANK,
                        "rank": rank,
                        "score": score,
                        "component_scores": components,
                    }
                )
            )
        return tuple(output)
