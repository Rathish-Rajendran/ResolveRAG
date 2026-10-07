from pathlib import Path

import pytest

from resolverag.data.serialization import atomic_text_writer


def test_atomic_text_writer_does_not_replace_target_on_failure(tmp_path: Path) -> None:
    path = tmp_path / "artifact.jsonl"
    path.write_text("original\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="stop"), atomic_text_writer(path) as destination:
        destination.write("partial\n")
        raise RuntimeError("stop")

    assert path.read_text(encoding="utf-8") == "original\n"
    assert not list(tmp_path.glob("*.tmp"))
