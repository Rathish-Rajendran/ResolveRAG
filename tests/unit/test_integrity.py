from pathlib import Path

from resolverag.data.integrity import sha256_file, sha256_text


def test_sha256_helpers_produce_the_same_digest(tmp_path: Path) -> None:
    path = tmp_path / "value.txt"
    path.write_text("ResolveRAG", encoding="utf-8")

    assert sha256_file(path) == sha256_text("ResolveRAG")
