"""
Unit Tests for Manual Zero-Shot and Few-Shot Classifiers.
"""

import pytest
from src.llm_backend.mock_llm import MockLLM
from src.baselines.zero_shot import ManualZeroShotClassifier
from src.baselines.few_shot import ManualFewShotClassifier


def test_zero_shot_classifier():
    llm = MockLLM()
    clf = ManualZeroShotClassifier(llm=llm)
    pred = clf.predict("When will my new debit card arrive in the mail?")
    assert pred == "card_arrival"

    stats = clf.get_prompt_stats()
    assert stats["char_length"] > 100
    assert stats["estimated_tokens"] > 50


def test_few_shot_classifier():
    llm = MockLLM()
    clf = ManualFewShotClassifier(llm=llm, shots_per_intent=1)
    pred = clf.predict("I dropped my wallet and need to freeze my card immediately.")
    assert pred == "lost_or_stolen_card"

    stats = clf.get_prompt_stats()
    assert stats["estimated_tokens"] > clf.get_prompt_stats()["estimated_tokens"] * 0.5
