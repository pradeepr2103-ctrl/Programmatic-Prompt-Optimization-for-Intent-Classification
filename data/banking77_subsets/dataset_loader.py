"""
Dataset Loader for Banking77 (10 Intent Subset)
Provides clean access to train, validation, and test datasets, intent labels,
and stratified sampling helpers.
"""

import json
import os
from typing import Dict, List, Optional, Tuple


class Banking77Subset:
    """Manages loading and querying of the Banking77 10-intent subset."""

    def __init__(self, data_path: Optional[str] = None):
        if data_path is None:
            data_path = os.path.join(
                os.path.dirname(__file__), "banking77_10intents.json"
            )
        self.data_path = data_path
        with open(self.data_path, "r", encoding="utf-8") as f:
            self.data = json.load(f)

        meta_path = os.path.join(os.path.dirname(__file__), "intents_metadata.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}

        self.intents: List[str] = self.data["metadata"]["intents"]
        self.train_data: List[Dict[str, str]] = self.data["train"]
        self.val_data: List[Dict[str, str]] = self.data["val"]
        self.test_data: List[Dict[str, str]] = self.data["test"]

    def get_train_examples(self) -> List[Dict[str, str]]:
        return list(self.train_data)

    def get_val_examples(self) -> List[Dict[str, str]]:
        return list(self.val_data)

    def get_test_examples(self) -> List[Dict[str, str]]:
        return list(self.test_data)

    def get_intents(self) -> List[str]:
        return list(self.intents)

    def get_intent_description(self, intent: str) -> str:
        if intent in self.metadata:
            return self.metadata[intent].get("description", "")
        return intent.replace("_", " ")

    def get_few_shot_demonstrations(self, n_shots_per_intent: int = 1) -> List[Dict[str, str]]:
        """Extracts n balanced demonstrations per intent from the training split."""
        demos = []
        intent_counts: Dict[str, int] = {i: 0 for i in self.intents}
        for item in self.train_data:
            intent = item["intent"]
            if intent_counts[intent] < n_shots_per_intent:
                demos.append(item)
                intent_counts[intent] += 1
        return demos


_DEFAULT_DATASET: Optional[Banking77Subset] = None


def get_dataset() -> Banking77Subset:
    """Singleton getter for the dataset."""
    global _DEFAULT_DATASET
    if _DEFAULT_DATASET is None:
        _DEFAULT_DATASET = Banking77Subset()
    return _DEFAULT_DATASET
