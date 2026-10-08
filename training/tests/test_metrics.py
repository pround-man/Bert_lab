import pytest


def test_routing_counts_wrong_single_and_missing_second_label():
    from watch_intents_training.metrics import compute_metrics
    # Exact, zero->single, multi->single, wrong single, zero->zero.
    truth = [[1, 0], [0, 0], [1, 1], [1, 0], [0, 0]]
    pred = [[1, 0], [1, 0], [1, 0], [0, 1], [0, 0]]
    m = compute_metrics(truth, pred, ["A", "B"])
    assert m["exact_set_accuracy"]["value"] == 0.4
    assert m["simple_error_rate"]["value"] == 0.75
    assert m["simple_coverage"]["value"] == 0.8
    assert m["correct_simple_coverage"]["value"] == 0.2
    assert m["simple_request_recall"]["value"] == 0.5
    assert m["zero_to_simple"]["value"] == 0.5
    assert m["multi_to_simple"]["value"] == 1
    assert m["cardinality_confusion"] == [[1, 1, 0], [0, 2, 0], [0, 1, 0]]
    assert m["micro_f1"] == pytest.approx(0.5)


def test_undefined_metrics_are_null_not_perfect():
    from watch_intents_training.metrics import compute_metrics
    m = compute_metrics([[0, 0]], [[0, 0]], ["A", "B"])
    assert m["exact_set_accuracy"]["value"] == 1
    assert m["simple_error_rate"]["value"] is None
    assert m["simple_error_rate"]["ci95"] is None
    assert m["micro_f1"] is None
    assert m["macro_f1"] is None
    assert m["per_label"][0]["precision"] is None


def test_threshold_inclusive_and_ties_prefer_half():
    from watch_intents_training.metrics import calibrate, predict_labels
    assert predict_labels([[0.5, 0.49]], 0.5).tolist() == [[1, 0]]
    result = calibrate([[1, 0]], [[0.99, 0.01]], ["A", "B"])
    assert result["threshold"] == 0.5
    assert len(result["curve"]) == 81


def test_slice_reports_do_not_hide_worst_language():
    from watch_intents_training.metrics import stratified_metrics
    rows = [{"locale": "en", "kind": "single", "scenario": "direct"},
            {"locale": "th", "kind": "single", "scenario": "withdrawal"}]
    result = stratified_metrics(rows, [[1], [1]], [[1], [0]], ["A"])
    assert result["language_equal_mean_exact"] == 0.5
    assert result["worst_language_exact"] == {"locale": "th", "value": 0.0}


def test_nonfinite_probabilities_are_rejected():
    from watch_intents_training.metrics import predict_labels
    with pytest.raises(ValueError):
        predict_labels([[float("nan")]], 0.5)
