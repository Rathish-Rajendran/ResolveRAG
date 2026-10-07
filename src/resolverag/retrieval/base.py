"""Application-owned interfaces for retrieval components."""

from typing import Protocol

from resolverag.domain.models import RetrievalResult


class QueryEmbedder(Protocol):
    def embed_query(self, text: str) -> list[float]: ...


class Retriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> tuple[RetrievalResult, ...]: ...


class PassageReranker(Protocol):
    def score(self, query: str, passages: list[str]) -> list[float]: ...
