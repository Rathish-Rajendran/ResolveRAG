"""Pinned Hugging Face dataset acquisition."""

from pathlib import Path

from huggingface_hub import hf_hub_download

from resolverag.config import DatasetConfig
from resolverag.data.integrity import sha256_file
from resolverag.exceptions import DatasetValidationError


def download_source_files(config: DatasetConfig) -> tuple[Path, ...]:
    """Download all configured files from the exact repository revision."""

    config.storage.raw_directory.mkdir(parents=True, exist_ok=True)
    filenames = (
        config.source_files.questions,
        config.source_files.corpus_archive,
    )
    paths: list[Path] = []
    for filename in filenames:
        destination = config.storage.raw_directory / filename
        expected_checksum = config.source_checksums.get(filename)
        if (
            destination.is_file()
            and expected_checksum is not None
            and sha256_file(destination) == expected_checksum
        ):
            paths.append(destination)
            continue
        downloaded = hf_hub_download(
            repo_id=config.dataset.repo_id,
            repo_type="dataset",
            revision=config.dataset.revision,
            filename=filename,
            local_dir=config.storage.raw_directory,
            force_download=destination.is_file(),
        )
        paths.append(Path(downloaded))
    return tuple(paths)


def validate_source_files(config: DatasetConfig) -> tuple[Path, ...]:
    """Ensure source files exist and match checksums in the dataset contract."""

    filenames = (
        config.source_files.questions,
        config.source_files.corpus_archive,
    )
    paths: list[Path] = []
    for filename in filenames:
        path = config.storage.raw_directory / filename
        if not path.is_file():
            raise DatasetValidationError(f"Required source file is missing: {path}")
        expected_checksum = config.source_checksums.get(filename)
        if expected_checksum is None:
            raise DatasetValidationError(f"No expected SHA-256 configured for {filename}")
        observed_checksum = sha256_file(path)
        if observed_checksum != expected_checksum:
            raise DatasetValidationError(
                f"SHA-256 mismatch for {filename}: expected {expected_checksum}, "
                f"observed {observed_checksum}"
            )
        paths.append(path)
    return tuple(paths)
