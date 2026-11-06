import pytest

from netra.metrics import iou, precision_recall


def test_iou_basic():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [20, 20, 30, 30]) == 0.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)


def test_precision_recall():
    truth = [{"label": "person", "box": [0, 0, 10, 10]}, {"label": "car", "box": [20, 20, 40, 40]}]
    pred = [{"label": "person", "box": [1, 1, 10, 10], "confidence": 0.9},
            {"label": "dog", "box": [50, 50, 60, 60], "confidence": 0.8}]
    r = precision_recall(pred, truth)
    assert (r["tp"], r["fp"], r["fn"]) == (1, 1, 1)
    assert r["precision"] == 0.5 and r["recall"] == 0.5
