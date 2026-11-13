"""
Unit Tests for Classification Metrics.
"""

import pytest
from src.metrics.classification_metrics import compute_metrics, render_ascii_confusion_matrix


def test_perfect_classification():
    y_true = ["card_arrival", "lost_or_stolen_card", "exchange_rate"]
    y_pred = ["card_arrival", "lost_or_stolen_card", "exchange_rate"]
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["macro_precision"] == 1.0
    assert metrics["macro_recall"] == 1.0


def test_imperfect_classification():
    y_true = ["card_arrival", "card_arrival", "exchange_rate", "exchange_rate"]
    y_pred = ["card_arrival", "exchange_rate", "exchange_rate", "exchange_rate"]
    metrics = compute_metrics(y_true, y_pred)

    assert metrics["accuracy"] == 0.75
    assert metrics["total_samples"] == 4
    assert 0.0 < metrics["macro_f1"] < 1.0


def test_confusion_matrix_rendering():
    y_true = ["card_arrival", "lost_or_stolen_card"]
    y_pred = ["card_arrival", "card_arrival"]
    metrics = compute_metrics(y_true, y_pred, labels=["card_arrival", "lost_or_stolen_card"])
    table_str = render_ascii_confusion_matrix(metrics)
    assert "Actual \\ Pred" in table_str
    assert "card_arrival" in table_str
