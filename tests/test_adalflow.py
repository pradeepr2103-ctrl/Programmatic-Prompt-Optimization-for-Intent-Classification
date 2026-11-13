"""
Unit Tests for AdalFlow Declarative Pipeline and Auto-Optimizer.
"""

import pytest
from src.llm_backend.mock_llm import MockLLM
from src.optimizers.adalflow_engine import AdalFlowClassifier, PromptTemplate, OutputParser


def test_adalflow_prompt_template_and_parser():
    template = PromptTemplate(
        template_str="Input: {query}\nOutput:",
        instructions="Test instructions",
        demos=[{"text": "example 1", "intent": "card_arrival"}],
    )
    rendered = template.render(query="test query")
    assert "Test instructions" in rendered
    assert "example 1" in rendered
    assert "test query" in rendered

    parser = OutputParser(valid_intents=["card_arrival", "card_linking"])
    assert parser("Intent: card_arrival") == "card_arrival"
    assert parser("Unknown random text") == "card_arrival"


def test_adalflow_classifier():
    llm = MockLLM()
    clf = AdalFlowClassifier(llm=llm)
    clf.optimize()
    assert clf.is_optimized

    pred = clf.predict("How do I complete identity verification in the app?")
    assert pred == "verify_identity"
