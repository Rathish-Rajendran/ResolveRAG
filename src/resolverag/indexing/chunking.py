"""Deterministic implementations of the five chunking strategies."""

import re
from collections.abc import Sequence
from typing import Protocol
from uuid import NAMESPACE_URL, uuid5

from langchain_text_splitters import RecursiveCharacterTextSplitter, TokenTextSplitter
from tiktoken import Encoding, get_encoding

from resolverag.data.integrity import sha256_text
from resolverag.domain.models import Chunk, ChunkingStrategy, SourceDocument
from resolverag.exceptions import ConfigurationError
from resolverag.indexing.config import StrategySettings

_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class DocumentChunker(Protocol):
    """Application-owned interface implemented by every chunker."""

    strategy: ChunkingStrategy

    def split(self, document: SourceDocument) -> tuple[Chunk, ...]: ...


def _stable_identifier(*parts: object) -> str:
    identity = ":".join(str(part) for part in parts)
    return str(uuid5(NAMESPACE_URL, f"resolverag:{identity}"))


def _required_int(value: int | None, name: str) -> int:
    if value is None:
        raise ConfigurationError(f"Missing chunking setting: {name}")
    return value


class _BaseChunker:
    strategy: ChunkingStrategy

    def __init__(self, strategy: ChunkingStrategy, encoding: Encoding) -> None:
        self.strategy = strategy
        self._encoding = encoding

    def _chunk(
        self,
        document: SourceDocument,
        index: int,
        text: str,
        *,
        context_text: str | None = None,
        parent_chunk_id: str | None = None,
    ) -> Chunk:
        normalized_text = text.strip()
        retrieval_context = (context_text or normalized_text).strip()
        text_digest = sha256_text(normalized_text)
        return Chunk(
            chunk_id=_stable_identifier(
                self.strategy.value,
                document.document_id,
                index,
                text_digest,
            ),
            document_id=document.document_id,
            source_path=document.source_path,
            strategy=self.strategy,
            chunk_index=index,
            text=normalized_text,
            context_text=retrieval_context,
            parent_chunk_id=parent_chunk_id,
            text_sha256=text_digest,
            token_count=len(self._encoding.encode(normalized_text, disallowed_special=())),
        )


class SplitterChunker(_BaseChunker):
    """Adapt a LangChain text splitter to ResolveRAG's canonical chunks."""

    def __init__(
        self,
        strategy: ChunkingStrategy,
        encoding: Encoding,
        splitter: TokenTextSplitter | RecursiveCharacterTextSplitter,
    ) -> None:
        super().__init__(strategy, encoding)
        self._splitter = splitter

    def split(self, document: SourceDocument) -> tuple[Chunk, ...]:
        texts = [text for text in self._splitter.split_text(document.text) if text.strip()]
        return tuple(self._chunk(document, index, text) for index, text in enumerate(texts))


class SentenceWindowChunker(_BaseChunker):
    """Group neighboring sentences while retaining a small sliding overlap."""

    def __init__(
        self,
        strategy: ChunkingStrategy,
        encoding: Encoding,
        *,
        window_size: int,
        overlap: int,
        max_sentence_characters: int,
    ) -> None:
        super().__init__(strategy, encoding)
        self._window_size = window_size
        self._step = window_size - overlap
        self._long_sentence_splitter = RecursiveCharacterTextSplitter(
            chunk_size=max_sentence_characters,
            chunk_overlap=0,
            separators=["\n", ". ", " ", ""],
            strip_whitespace=True,
        )

    def _sentences(self, text: str) -> list[str]:
        sentences: list[str] = []
        for candidate in _SENTENCE_BOUNDARY.split(text):
            candidate = candidate.strip()
            if candidate:
                sentences.extend(
                    part.strip()
                    for part in self._long_sentence_splitter.split_text(candidate)
                    if part.strip()
                )
        return sentences

    def split(self, document: SourceDocument) -> tuple[Chunk, ...]:
        sentences = self._sentences(document.text)
        chunks: list[Chunk] = []
        for start in range(0, len(sentences), self._step):
            window = sentences[start : start + self._window_size]
            if not window:
                continue
            chunks.append(self._chunk(document, len(chunks), " ".join(window)))
            if start + self._window_size >= len(sentences):
                break
        return tuple(chunks)


class ParentChildChunker(_BaseChunker):
    """Embed small child chunks while retaining their larger parent context."""

    def __init__(
        self,
        strategy: ChunkingStrategy,
        encoding: Encoding,
        *,
        parent_size: int,
        parent_overlap: int,
        child_size: int,
        child_overlap: int,
    ) -> None:
        super().__init__(strategy, encoding)
        separators = ["\n\n", "\n", ". ", " ", ""]
        self._parent_splitter = RecursiveCharacterTextSplitter(
            chunk_size=parent_size,
            chunk_overlap=parent_overlap,
            separators=separators,
            strip_whitespace=True,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=child_size,
            chunk_overlap=child_overlap,
            separators=separators,
            strip_whitespace=True,
        )

    def split(self, document: SourceDocument) -> tuple[Chunk, ...]:
        chunks: list[Chunk] = []
        parents = [text for text in self._parent_splitter.split_text(document.text) if text.strip()]
        for parent_index, parent_text in enumerate(parents):
            parent_id = _stable_identifier(
                self.strategy.value,
                document.document_id,
                "parent",
                parent_index,
                sha256_text(parent_text),
            )
            children = [
                text for text in self._child_splitter.split_text(parent_text) if text.strip()
            ]
            for child_text in children:
                chunks.append(
                    self._chunk(
                        document,
                        len(chunks),
                        child_text,
                        context_text=parent_text,
                        parent_chunk_id=parent_id,
                    )
                )
        return tuple(chunks)


def _validate_kind(strategy: ChunkingStrategy, kind: str) -> None:
    expected_kinds = {
        ChunkingStrategy.FIXED_256: "token",
        ChunkingStrategy.FIXED_512: "token",
        ChunkingStrategy.RECURSIVE: "recursive",
        ChunkingStrategy.SENTENCE_WINDOW: "sentence_window",
        ChunkingStrategy.PARENT_CHILD: "parent_child",
    }
    if kind != expected_kinds[strategy]:
        raise ConfigurationError(
            f"Strategy {strategy.value!r} requires kind {expected_kinds[strategy]!r}, not {kind!r}"
        )


def build_chunker(
    strategy: ChunkingStrategy,
    settings: StrategySettings,
    *,
    encoding_name: str,
) -> DocumentChunker:
    """Construct one configured chunker behind an application-owned interface."""

    _validate_kind(strategy, settings.kind)
    encoding = get_encoding(encoding_name)
    if settings.kind == "token":
        splitter = TokenTextSplitter(
            encoding_name=encoding_name,
            chunk_size=_required_int(settings.chunk_size, "chunk_size"),
            chunk_overlap=_required_int(settings.chunk_overlap, "chunk_overlap"),
            strip_whitespace=True,
            disallowed_special=(),
        )
        return SplitterChunker(strategy, encoding, splitter)
    if settings.kind == "recursive":
        recursive_splitter = RecursiveCharacterTextSplitter(
            chunk_size=_required_int(settings.chunk_size, "chunk_size"),
            chunk_overlap=_required_int(settings.chunk_overlap, "chunk_overlap"),
            separators=settings.separators,
            strip_whitespace=True,
        )
        return SplitterChunker(strategy, encoding, recursive_splitter)
    if settings.kind == "sentence_window":
        return SentenceWindowChunker(
            strategy,
            encoding,
            window_size=_required_int(settings.window_size, "window_size"),
            overlap=_required_int(settings.sentence_overlap, "sentence_overlap"),
            max_sentence_characters=_required_int(
                settings.max_sentence_characters,
                "max_sentence_characters",
            ),
        )
    return ParentChildChunker(
        strategy,
        encoding,
        parent_size=_required_int(settings.parent_chunk_size, "parent_chunk_size"),
        parent_overlap=_required_int(settings.parent_chunk_overlap, "parent_chunk_overlap"),
        child_size=_required_int(settings.child_chunk_size, "child_chunk_size"),
        child_overlap=_required_int(settings.child_chunk_overlap, "child_chunk_overlap"),
    )


def strategy_names(strategies: Sequence[ChunkingStrategy]) -> str:
    """Return stable, human-readable names for logs and errors."""

    return ", ".join(strategy.value for strategy in strategies)
