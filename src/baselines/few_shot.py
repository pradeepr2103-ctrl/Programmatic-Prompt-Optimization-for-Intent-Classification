"""
Manual Few-Shot Baseline Intent Classifier.
Uses manually curated in-context demonstrations (1-shot or 2-shot per intent).
"""

import re
from typing import Dict, List, Optional
from src.llm_backend.base import BaseLLM
from src.llm_backend.tracker import estimate_tokens
from data.banking77_subsets.dataset_loader import get_dataset


class ManualFewShotClassifier:
    """Manual Few-Shot intent classifier using static human-curated examples."""

    def __init__(self, llm: BaseLLM, shots_per_intent: int = 1):
        self.llm = llm
        self.dataset = get_dataset()
        self.intents = self.dataset.get_intents()
        self.shots_per_intent = shots_per_intent
        self.demonstrations = self.dataset.get_few_shot_demonstrations(shots_per_intent)
        self.system_prompt = self._build_system_prompt()

    def _build_system_prompt(self) -> str:
        intent_lines = []
        for intent in self.intents:
            desc = self.dataset.get_intent_description(intent)
            intent_lines.append(f"- {intent}: {desc}")
        intents_block = "\n".join(intent_lines)

        demo_lines = []
        for idx, demo in enumerate(self.demonstrations, 1):
            demo_lines.append(f"Example {idx}:\nCustomer: \"{demo['text']}\"\nIntent: {demo['intent']}\n")
        demos_block = "\n".join(demo_lines)

        return (
            "You are an expert customer support intent classifier for a banking platform.\n"
            "Your task is to classify incoming customer inquiries into exactly one of the following 10 intents:\n\n"
            f"{intents_block}\n\n"
            "Here are verified reference examples demonstrating the correct mapping:\n\n"
            f"{demos_block}\n"
            "INSTRUCTIONS:\n"
            "1. Analyze the customer inquiry against the examples and intent definitions.\n"
            "2. Output ONLY the exact intent label."
        )

    def format_prompt(self, utterance: str) -> str:
        return f"Customer: \"{utterance}\"\nIntent:"

    def predict(self, utterance: str) -> str:
        prompt = self.format_prompt(utterance)
        raw_output = self.llm(prompt=prompt, system_prompt=self.system_prompt, temperature=0.0)
        return self._clean_prediction(raw_output)

    def predict_batch(self, utterances: List[str]) -> List[str]:
        return [self.predict(u) for u in utterances]

    def _clean_prediction(self, raw_pred: str) -> str:
        text = raw_pred.strip().lower()
        for intent in self.intents:
            if intent in text:
                return intent
        match = re.search(r"([a-z_]{4,35})", text)
        if match and match.group(1) in self.intents:
            return match.group(1)
        return self.intents[0]

    def get_prompt_text(self) -> str:
        sample_query = self.format_prompt("When will my replacement card arrive?")
        return f"{self.system_prompt}\n\n{sample_query}"

    def get_prompt_stats(self) -> Dict[str, int]:
        full_text = self.get_prompt_text()
        return {
            "char_length": len(full_text),
            "word_count": len(full_text.split()),
            "estimated_tokens": estimate_tokens(full_text),
        }
