"""
Unit Tests for DSPy Modules and Optimizers.
"""

import pytest
from src.llm_backend.mock_llm import MockLLM
from src.optimizers.dspy_engine import DSPyIntentClassifier, BootstrapFewShot, COPROOptimizer


def test_dspy_bootstrap_optimizer():
    llm = MockLLM()
    clf = DSPyIntentClassifier(llm=llm, optimizer_name="bootstrap")
    clf.optimize()
    assert clf.is_compiled
    assert len(clf.module.demos) > 0

    pred = clf.predict("How do I link an external debit card to my app account?")
    assert pred == "card_linking"


def test_dspy_copro_optimizer():
    llm = MockLLM()
    clf = DSPyIntentClassifier(llm=llm, optimizer_name="copro")
    clf.optimize()
    assert clf.is_compiled

    pred = clf.predict("What foreign exchange rate do you use for converting EUR to USD?")
    assert pred == "exchange_rate"
