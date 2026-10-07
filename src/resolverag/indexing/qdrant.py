"""Qdrant client construction shared by indexing and retrieval."""

from qdrant_client import QdrantClient

from resolverag.indexing.config import QdrantSettings


def create_qdrant_client(settings: QdrantSettings) -> QdrantClient:
    """Create the configured local or server-backed Qdrant client."""

    if settings.mode == "local":
        return QdrantClient(path=settings.location)
    return QdrantClient(
        url=settings.location,
        prefer_grpc=settings.prefer_grpc,
        timeout=settings.timeout_seconds,
    )
