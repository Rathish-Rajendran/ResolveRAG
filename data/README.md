# ResolveRAG Data

This directory contains the local data used by ResolveRAG for ingestion, retrieval, and evaluation experiments.

## Dataset

ResolveRAG uses **NVIDIA TechQA-RAG-Eval**, a RAG-ready derivative of the **IBM Research TechQA** benchmark.

TechQA is based on real technical-support questions and IBM technical documentation (Technotes), making it suitable for evaluating enterprise retrieval-augmented generation systems.

- **Dataset:** TechQA-RAG-Eval
- **Publisher:** NVIDIA
- **Upstream dataset:** IBM Research TechQA
- **Hugging Face repository:** `nvidia/TechQA-RAG-Eval`
- **Pinned revision:** `0b5bbc84b7f07d6d09d063130e90b716d8d4a32a`
- **License:** Apache-2.0
- **Dataset configuration:** `configs/datasets/techqa.yaml`

The dataset revision is pinned so that experiments can be reproduced against the same source data even if the upstream repository changes.

## Directory Structure

```text
data/
├── raw/
│   └── techqa/
├── interim/
│   └── techqa/
├── processed/
│   └── techqa/
│       ├── documents.jsonl
│       ├── queries.jsonl
│       ├── qrels.jsonl
│       ├── indexes/
│       │   ├── smoke_N/
│       │   └── full/
│       └── manifest.json
└── README.md
```

### `raw/`

Contains the original files downloaded from the external dataset source.

Raw files are treated as **immutable**. They must never be manually edited or corrected.

Any cleaning or transformation must be performed through reproducible code so that the processed dataset can always be regenerated from the original source.

### `interim/`

Contains intermediate artifacts produced while extracting, validating, cleaning, deduplicating, or normalizing the raw dataset.

These files are implementation artifacts and may be regenerated at any time.

### `processed/`

Contains the canonical representation of the dataset consumed by ResolveRAG.

Processed records are created from the raw dataset through versioned transformation logic. Downstream components such as chunking, retrieval, indexing, and evaluation should consume the canonical processed representation rather than depend directly on the external dataset schema.

Each index-build scope contains one chunk JSONL file per strategy, chunking
statistics, and an index manifest. Vector data is stored separately under
`qdrant_storage/`; both locations are reproducible local artifacts ignored by
Git.

## Data Provenance

Each processed dataset will include a `manifest.json` describing the data that was actually processed.

The manifest will record information such as:

- Dataset and source revision
- Source file checksums
- Raw and processed record counts
- Processing timestamp
- Transformation/schema version
- Relevant library versions

SHA-256 checksums are used to verify that experiments are operating on identical source artifacts.

The dataset YAML describes **what ResolveRAG expects**, while the generated manifest records **what ResolveRAG actually observed and processed**.

## Evaluation Split Policy

The packaged dataset contains records originating from the TechQA `TRAIN_` and `DEV_` groups.

ResolveRAG treats them as separate logical splits:

- `TRAIN_` — used for development, experimentation, and configuration selection.
- `DEV_` — held out for final evaluation.

The held-out `DEV_` records must not be used to select chunk sizes, retrieval strategies, rerankers, `top_k` values, prompts, or other RAG configuration parameters.

This separation helps prevent evaluation leakage and provides a more reliable estimate of how the selected pipeline performs on unseen questions.

## Unanswerable Questions

TechQA contains questions for which the available evidence does not provide a valid answer.

These records are intentionally preserved rather than discarded.

They will be used to evaluate whether ResolveRAG can recognize insufficient evidence and abstain instead of generating an unsupported answer.

## Git Policy

Downloaded datasets and generated artifacts are intentionally excluded from Git.

The repository stores:

- Dataset configuration
- Ingestion and transformation code
- Validation rules
- Evaluation logic
- Documentation

The repository does **not** store large downloaded or generated datasets.

The following directories are therefore ignored by Git:

```text
data/raw/
data/interim/
data/processed/
```

This keeps the repository lightweight while allowing the data to be reproduced from its pinned source revision.

## Regenerating the Dataset

Prepare the complete canonical dataset with:

```bash
uv run resolverag dataset prepare --config configs/datasets/techqa.yaml
```

The command downloads the pinned files when necessary, verifies their SHA-256
checksums, validates the source schema, and produces `documents.jsonl`,
`queries.jsonl`, `qrels.jsonl`, and `manifest.json` under
`data/processed/techqa/`.

Use `--no-download` to require that verified raw files already exist locally.
Downloaded and generated data must not be manually modified or committed.
