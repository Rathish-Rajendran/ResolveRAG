"""Atomic serialization of canonical artifacts."""

import os
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import IO, Protocol


class JsonSerializable(Protocol):
    """The small serialization surface required from canonical models."""

    def model_dump_json(self, *, exclude_none: bool = ...) -> str: ...


@contextmanager
def atomic_text_writer(path: Path) -> Iterator[IO[str]]:
    """Yield a temporary UTF-8 writer and replace the target only on success."""

    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as destination:
            temporary_path = Path(destination.name)
            yield destination
            destination.flush()
            os.fsync(destination.fileno())
        temporary_path.replace(path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def write_jsonl(path: Path, records: Iterable[JsonSerializable]) -> None:
    """Atomically write canonical models as newline-delimited JSON."""

    with atomic_text_writer(path) as destination:
        for record in records:
            destination.write(record.model_dump_json(exclude_none=False))
            destination.write("\n")


def write_text(path: Path, content: str) -> None:
    """Atomically write a UTF-8 text artifact."""

    with atomic_text_writer(path) as destination:
        destination.write(content)
