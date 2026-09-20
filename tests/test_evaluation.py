from cognition_ake.evaluation import document_counts, evaluate_micro
from cognition_ake.normalization import normalize_phrase, unique_ranked


def test_normalization_and_deduplication():
    assert normalize_phrase("  Neural   Networks. ") == "neural networks"
    assert unique_ranked(["A", "a", "B"], k=2) == ["a", "b"]


def test_dataset_micro_f1():
    records = [
        {"doc_id": "a", "predictions": [["x", 1.0], ["z", 0.5]], "gold_present": ["x", "y"]},
        {"doc_id": "b", "predictions": [["m", 1.0]], "gold_present": ["m"]},
    ]
    metrics = evaluate_micro(records, 5)
    assert metrics["tp"] == 2
    assert metrics["predicted"] == 3
    assert metrics["gold"] == 3
    assert abs(metrics["f1"] - 2 / 3) < 1e-12


def test_top_k_after_rank_preserving_deduplication():
    row = {"predictions": [["A", 1], ["a", 0.9], ["B", 0.8]], "gold_present": ["a", "b"]}
    assert document_counts(row, 2) == {"tp": 2, "predicted": 2, "gold": 2}

