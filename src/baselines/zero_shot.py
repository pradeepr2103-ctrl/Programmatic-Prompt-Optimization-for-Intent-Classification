"""
Manual Zero-Shot Baseline Intent Classifier.
Uses a manually engineered zero-shot instruction prompt with intent descriptions.
"""

import re
from typing import Dict, List, Optional
from src.llm_backend.base import BaseLLM
from src.llm_backend.tracker import estimate_tokens
from data.banking77_subsets.dataset_loader import get_dataset


class ManualZeroShotClassifier:
    """Manual zero-shot intent classifier."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.dataset = get_dataset()
        self.intents = self.dataset.get_intents()
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        intent_lines = []
        for intent in self.intents:
            desc = self.dataset.get_intent_description(intent)
            intent_lines.append(f"- {intent}: {desc}")
        intents_block = "\n".join(intent_lines)

        return (
            "You are a helpful customer support intent classification assistant for a retail bank.\n"
            "Your task is to classify incoming customer inquiries into exactly one of the following banking intents:\n\n"
            f"{intents_block}\n\n"
            "INSTRUCTIONS:\n"
            "1. Read the customer inquiry carefully.\n"
            "2. Select the single best matching intent from the list above.\n"
            "3. Return ONLY the exact intent label identifier (e.g. 'card_arrival'). Do not include explanation or markdown."
        )

    def format_prompt(self, utterance: str) -> str:
        return f"Customer Query: \"{utterance}\"\nIntent:"

    def predict(self, utterance: str) -> str:
        prompt = self.format_prompt(utterance)
        raw_output = self.llm(prompt=prompt, system_prompt=self.system_prompt, temperature=0.0)
        return self._clean_prediction(raw_output)

    def predict_batch(self, utterances: List[str]) -> List[str]:
        return [self.predict(u) for u in utterances]

    def _clean_prediction(self, raw_pred: str) -> str:
        text = raw_pred.strip().lower()
        # Direct match check
        for intent in self.intents:
            if intent in text:
                return intent
        # Match with regex
        match = re.search(r"([a-z_]{4,35})", text)
        if match and match.group(1) in self.intents:
            return match.group(1)
        # Fallback to closest match
        for intent in self.intents:
            parts = intent.split("_")
            if any(p in text for p in parts if len(p) > 3):
                return intent
        return self.intents[0]

    def get_prompt_text(self) -> str:
        """Returns the full prompt definition for length accounting."""
        sample_query = self.format_prompt("When will my replacement card arrive?")
        return f"{self.system_prompt}\n\n{sample_query}"

    def get_prompt_stats(self) -> Dict[str, int]:
        full_text = self.get_prompt_text()
        return {
            "char_length": len(full_text),
            "word_count": len(full_text.split()),
            "estimated_tokens": estimate_tokens(full_text),
        }
