import pytest

from resolverag.evaluation.metrics import evaluate_ranking, ndcg_at_k


def test_retrieval_metrics_use_document_level_ranks() -> None:
    metrics = evaluate_ranking(["irrelevant.txt", "relevant.txt"], {"relevant.txt"})

    assert metrics.recall_at_1 == 0.0
    assert metrics.recall_at_5 == 1.0
    assert metrics.recall_at_10 == 1.0
    assert metrics.reciprocal_rank_at_10 == 0.5
    assert metrics.ndcg_at_10 == pytest.approx(0.6309297536)


def test_ndcg_supports_multiple_relevant_documents() -> None:
    score = ndcg_at_k(["a", "x", "b"], {"a", "b"}, 3)

    assert 0.0 < score < 1.0
