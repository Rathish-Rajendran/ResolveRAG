"""Validated runtime wiring for all six retrieval configurations."""

from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from qdrant_client import QdrantClient

from resolverag.data.integrity import sha256_file
from resolverag.domain.models import (
    ChunkingStrategy,
    RetrievalConfiguration,
    RetrievalResult,
)
from resolverag.exceptions import RetrievalError
from resolverag.indexing.config import IndexConfig, load_index_config
from resolverag.indexing.models import CollectionBuild, IndexBuildManifest
from resolverag.retrieval.base import PassageReranker, QueryEmbedder, Retriever
from resolverag.retrieval.config import RetrievalConfig, load_retrieval_config
from resolverag.retrieval.dense import DenseRetriever
from resolverag.retrieval.fusion import (
    ReciprocalRankFusionRetriever,
    WeightedScoreFusionRetriever,
)
from resolverag.retrieval.reranking import CrossEncoderReranker, RerankingRetriever
from resolverag.retrieval.sparse import BM25Retriever


class _CachedQueryEmbedder:
    def __init__(self, delegate: QueryEmbedder) -> None:
        self._delegate = delegate
        self._cache: dict[str, list[float]] = {}

    def embed_query(self, text: str) -> list[float]:
        if text not in self._cache:
            self._cache[text] = self._delegate.embed_query(text)
        return list(self._cache[text])


class RetrievalRuntime:
    """Own validated artifacts and lazily construct retrieval components."""

    def __init__(
        self,
        config_path: Path,
        strategy: ChunkingStrategy,
        *,
        index_manifest_path: Path | None = None,
        embedder: QueryEmbedder | None = None,
        reranker: PassageReranker | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        self.config: RetrievalConfig = load_retrieval_config(config_path)
        self.index_config: IndexConfig = load_index_config(self.config.inputs.index_config_file)
        self.index_manifest_path = index_manifest_path or self.config.inputs.index_manifest_file
        self.index_manifest = self._load_manifest(self.index_manifest_path)
        self._validate_index_provenance()
        self.strategy = strategy
        self.collection = self._find_collection(strategy)
        self.chunks_path = self.index_manifest_path.parent / self.collection.chunk_artifact.filename
        self._provided_embedder = embedder
        self._provided_reranker = reranker
        self._provided_client = client
        self._embedder: _CachedQueryEmbedder | None = None
        self._reranker: PassageReranker | None = None
        self._client: QdrantClient | None = None
        self._owns_client = False
        self._sparse: BM25Retriever | None = None

    @staticmethod
    def _load_manifest(path: Path) -> IndexBuildManifest:
        if not path.is_file():
            raise RetrievalError(f"Index manifest does not exist: {path}")
        try:
            return IndexBuildManifest.model_validate_json(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as error:
            raise RetrievalError(f"Invalid index manifest {path}: {error}") from error

    def _validate_index_provenance(self) -> None:
        config_checksum = sha256_file(self.config.inputs.index_config_file)
        if config_checksum != self.index_manifest.configuration_sha256:
            raise RetrievalError(
                "Index configuration changed after the selected collections were built"
            )
        if self.index_manifest.embedding_model != self.index_config.embedding.model:
            raise RetrievalError("Index manifest and embedding configuration use different models")
        if (
            self.index_manifest.embedding_dimensions
            != self.index_config.embedding.expected_dimensions
        ):
            raise RetrievalError("Index manifest and configuration disagree on vector dimensions")
        if self.index_manifest.qdrant_path != str(self.index_config.qdrant.path):
            raise RetrievalError("Index manifest points to a different Qdrant database")

    def _find_collection(self, strategy: ChunkingStrategy) -> CollectionBuild:
        for collection in self.index_manifest.collections:
            if collection.strategy is strategy:
                return collection
        raise RetrievalError(
            f"Index manifest {self.index_manifest_path} does not contain {strategy.value}"
        )

    def _get_sparse(self) -> BM25Retriever:
        if self._sparse is None:
            self._sparse = BM25Retriever(
                self.chunks_path,
                self.collection.chunk_artifact.sha256,
            )
        return self._sparse

    def _get_embedder(self) -> _CachedQueryEmbedder:
        if self._embedder is None:
            delegate = self._provided_embedder or OllamaEmbeddings(
                model=self.index_config.embedding.model,
                base_url=self.index_config.embedding.base_url,
                validate_model_on_init=True,
            )
            self._embedder = _CachedQueryEmbedder(delegate)
        return self._embedder

    def _get_client(self) -> QdrantClient:
        if self._client is None:
            if self._provided_client is not None:
                self._client = self._provided_client
            else:
                self._client = QdrantClient(path=str(self.index_config.qdrant.path))
                self._owns_client = True
            if not self._client.collection_exists(self.collection.collection_name):
                raise RetrievalError(
                    f"Qdrant collection does not exist: {self.collection.collection_name}"
                )
        return self._client

    def _get_dense(self, *, mmr: bool = False) -> DenseRetriever:
        settings = self.config.retrieval
        return DenseRetriever(
            self._get_client(),
            self.collection.collection_name,
            self._get_embedder(),
            query_prefix=self.index_config.embedding.query_prefix,
            candidate_k=settings.dense_candidate_k,
            mmr_lambda=settings.mmr_lambda if mmr else None,
        )

    def _get_rrf(self) -> ReciprocalRankFusionRetriever:
        settings = self.config.retrieval
        return ReciprocalRankFusionRetriever(
            self._get_sparse(),
            self._get_dense(),
            sparse_candidate_k=settings.sparse_candidate_k,
            dense_candidate_k=settings.dense_candidate_k,
            rank_constant=settings.reciprocal_rank_constant,
        )

    def _get_weighted(self) -> WeightedScoreFusionRetriever:
        settings = self.config.retrieval
        return WeightedScoreFusionRetriever(
            self._get_sparse(),
            self._get_dense(),
            sparse_candidate_k=settings.sparse_candidate_k,
            dense_candidate_k=settings.dense_candidate_k,
            dense_weight=settings.dense_weight,
        )

    def _get_reranker(self) -> PassageReranker:
        if self._reranker is None:
            self._reranker = self._provided_reranker or CrossEncoderReranker(
                self.config.reranker.model,
                self.config.reranker.batch_size,
            )
        return self._reranker

    def _retriever(self, configuration: RetrievalConfiguration) -> Retriever:
        if configuration is RetrievalConfiguration.BM25:
            return self._get_sparse()
        if configuration is RetrievalConfiguration.DENSE:
            return self._get_dense()
        if configuration is RetrievalConfiguration.DENSE_MMR:
            return self._get_dense(mmr=True)
        if configuration is RetrievalConfiguration.HYBRID_RRF:
            return self._get_rrf()
        if configuration is RetrievalConfiguration.HYBRID_WEIGHTED:
            return self._get_weighted()
        return RerankingRetriever(
            self._get_rrf(),
            self._get_reranker(),
            candidate_k=self.config.retrieval.hybrid_candidate_k,
        )

    def search(
        self,
        query: str,
        configuration: RetrievalConfiguration,
        *,
        top_k: int | None = None,
    ) -> tuple[RetrievalResult, ...]:
        """Run one configured retriever and return traceable ranked chunks."""

        normalized_query = query.strip()
        if not normalized_query:
            raise RetrievalError("Query must not be empty")
        result_count = top_k or self.config.retrieval.default_top_k
        if result_count <= 0:
            raise RetrievalError("top_k must be greater than zero")
        if result_count > self.config.retrieval.hybrid_candidate_k:
            raise RetrievalError("top_k cannot exceed hybrid_candidate_k")
        return self._retriever(configuration).retrieve(normalized_query, result_count)

    def close(self) -> None:
        """Release a locally owned Qdrant database handle."""

        if self._owns_client and self._client is not None:
            self._client.close()
            self._client = None
            self._owns_client = False

    def __enter__(self) -> "RetrievalRuntime":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
