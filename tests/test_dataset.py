"""
Unit Tests for Banking77 Dataset Loader and Splits.
"""

import pytest
from data.banking77_subsets.dataset_loader import get_dataset


def test_dataset_loading():
    dataset = get_dataset()
    assert dataset is not None
    intents = dataset.get_intents()
    assert len(intents) == 10
    assert "card_arrival" in intents
    assert "wrong_amount_additional_charge" in intents


def test_dataset_splits_balance():
    dataset = get_dataset()
    train_data = dataset.get_train_examples()
    val_data = dataset.get_val_examples()
    test_data = dataset.get_test_examples()

    assert len(train_data) == 120  # 12 per intent
    assert len(val_data) == 40     # 4 per intent
    assert len(test_data) == 80    # 8 per intent

    # Check intent distribution in test split
    test_counts = {}
    for item in test_data:
        test_counts[item["intent"]] = test_counts.get(item["intent"], 0) + 1

    for intent in dataset.get_intents():
        assert test_counts[intent] == 8


def test_few_shot_demonstrations():
    dataset = get_dataset()
    demos = dataset.get_few_shot_demonstrations(n_shots_per_intent=1)
    assert len(demos) == 10
    intents_represented = set(d["intent"] for d in demos)
    assert len(intents_represented) == 10
