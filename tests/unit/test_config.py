from pathlib import Path

import pytest

from resolverag.config import load_dataset_config
from resolverag.exceptions import ConfigurationError


def test_load_dataset_config_rejects_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigurationError, match="does not exist"):
        load_dataset_config(tmp_path / "missing.yaml")


def test_load_dataset_config_rejects_invalid_shape(tmp_path: Path) -> None:
    config_path = tmp_path / "invalid.yaml"
    config_path.write_text("dataset: unexpected\n", encoding="utf-8")

    with pytest.raises(ConfigurationError, match="Invalid dataset configuration"):
        load_dataset_config(config_path)
