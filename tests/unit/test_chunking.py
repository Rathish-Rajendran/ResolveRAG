from collections.abc import Iterable

import pytest

from resolverag.data.integrity import sha256_text
from resolverag.domain.models import Chunk, ChunkingStrategy, SourceDocument
from resolverag.exceptions import ConfigurationError
from resolverag.indexing.chunking import build_chunker
from resolverag.indexing.config import StrategySettings


def _document(text: str) -> SourceDocument:
    return SourceDocument(
        document_id="doc-1.txt",
        text=text,
        source_path="corpus/doc-1.txt",
        text_sha256=sha256_text(text),
    )


def _texts(chunks: Iterable[Chunk]) -> list[str]:
    return [chunk.text for chunk in chunks]


@pytest.mark.parametrize(
    ("strategy", "size", "overlap"),
    [
        (ChunkingStrategy.FIXED_256, 12, 2),
        (ChunkingStrategy.FIXED_512, 20, 4),
    ],
)
def test_fixed_token_chunkers_are_stable(
    strategy: ChunkingStrategy,
    size: int,
    overlap: int,
) -> None:
    settings = StrategySettings(kind="token", chunk_size=size, chunk_overlap=overlap)
    chunker = build_chunker(strategy, settings, encoding_name="cl100k_base")
    document = _document(" ".join(f"token-{index}" for index in range(80)))

    first = chunker.split(document)
    second = chunker.split(document)

    assert len(first) > 1
    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert all(chunk.strategy is strategy for chunk in first)
    assert all(chunk.context_text == chunk.text for chunk in first)


def test_recursive_chunker_preserves_document_traceability() -> None:
    settings = StrategySettings(
        kind="recursive",
        chunk_size=55,
        chunk_overlap=5,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunker = build_chunker(
        ChunkingStrategy.RECURSIVE,
        settings,
        encoding_name="cl100k_base",
    )

    chunks = chunker.split(_document("Paragraph one is here.\n\nParagraph two is longer here."))

    assert chunks
    assert {chunk.document_id for chunk in chunks} == {"doc-1.txt"}
    assert [chunk.chunk_index for chunk in chunks] == list(range(len(chunks)))


def test_sentence_window_uses_neighboring_sentences() -> None:
    settings = StrategySettings(
        kind="sentence_window",
        window_size=3,
        sentence_overlap=1,
        max_sentence_characters=100,
    )
    chunker = build_chunker(
        ChunkingStrategy.SENTENCE_WINDOW,
        settings,
        encoding_name="cl100k_base",
    )

    chunks = chunker.split(_document("One. Two. Three. Four. Five."))

    assert _texts(chunks) == ["One. Two. Three.", "Three. Four. Five."]


def test_parent_child_embeds_children_and_returns_parent_context() -> None:
    settings = StrategySettings(
        kind="parent_child",
        parent_chunk_size=80,
        parent_chunk_overlap=10,
        child_chunk_size=30,
        child_chunk_overlap=5,
    )
    chunker = build_chunker(
        ChunkingStrategy.PARENT_CHILD,
        settings,
        encoding_name="cl100k_base",
    )
    document = _document(
        "First paragraph has enough words to split into children. "
        "Second sentence adds more detail. Third sentence finishes it."
    )

    chunks = chunker.split(document)

    assert len(chunks) > 1
    assert all(chunk.parent_chunk_id is not None for chunk in chunks)
    assert all(chunk.text in chunk.context_text for chunk in chunks)
    assert any(chunk.context_text != chunk.text for chunk in chunks)


def test_chunker_rejects_strategy_kind_mismatch() -> None:
    settings = StrategySettings(kind="token", chunk_size=10, chunk_overlap=1)

    with pytest.raises(ConfigurationError, match="requires kind"):
        build_chunker(
            ChunkingStrategy.RECURSIVE,
            settings,
            encoding_name="cl100k_base",
        )
