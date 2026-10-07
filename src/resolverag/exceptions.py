"""Application-specific exceptions."""


class ConfigurationError(ValueError):
    """Raised when a ResolveRAG configuration file is invalid."""


class DatasetValidationError(ValueError):
    """Raised when source data violates the configured dataset contract."""


class IndexBuildError(RuntimeError):
    """Raised when chunking, embedding, or index persistence fails."""


class RetrievalError(RuntimeError):
    """Raised when a retriever cannot load or rank its candidates."""


class EvaluationError(RuntimeError):
    """Raised when a benchmark plan or its persisted state is invalid."""
