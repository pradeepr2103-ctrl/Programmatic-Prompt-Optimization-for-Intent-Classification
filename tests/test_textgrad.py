"""
Unit Tests for TextGrad Engine and Optimization.
"""

import pytest
from src.llm_backend.mock_llm import MockLLM
from src.optimizers.textgrad_engine import Variable, TextualLossCritic, TextualGradientDescent, TextGradClassifier


def test_textgrad_variable_and_gradients():
    var = Variable(value="Initial Prompt", role_description="Test Prompt")
    assert var.value == "Initial Prompt"
    assert len(var.gradients) == 0

    var.add_gradient("Gradient critique 1")
    assert len(var.gradients) == 1
    var.zero_grad()
    assert len(var.gradients) == 0


def test_textgrad_classifier():
    llm = MockLLM()
    clf = TextGradClassifier(llm=llm)
    clf.optimize(epochs=1, batch_size=4)

    pred = clf.predict("Why was my top-up payment cancelled and reverted?")
    assert pred == "top_up_reverted"

    stats = clf.get_prompt_stats()
    assert stats["estimated_tokens"] > 50
