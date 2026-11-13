"""Baselines package."""
from src.baselines.zero_shot import ManualZeroShotClassifier
from src.baselines.few_shot import ManualFewShotClassifier

__all__ = ["ManualZeroShotClassifier", "ManualFewShotClassifier"]
