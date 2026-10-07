"""File integrity helpers."""

from hashlib import sha256
from pathlib import Path


def sha256_file(path: Path, *, block_size: int = 1024 * 1024) -> str:
    """Return a streaming SHA-256 digest for a file."""

    digest = sha256()
    with path.open("rb") as source:
        while block := source.read(block_size):
            digest.update(block)
    return digest.hexdigest()


def sha256_text(value: str) -> str:
    """Return the SHA-256 digest of UTF-8 text."""

    return sha256(value.encode("utf-8")).hexdigest()
