"""
Abstract Base LLM Interface with instrumentation.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List, Optional
from src.llm_backend.tracker import TokenTracker, estimate_tokens


@dataclass
class LLMResponse:
    text: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    raw_response: Optional[dict] = None


class BaseLLM(ABC):
    """Abstract class for all language model backends."""

    def __init__(self, model_name: str = "mock-gpt4", tracker: Optional[TokenTracker] = None):
        self.model_name = model_name
        self.tracker = tracker if tracker is not None else TokenTracker(name=model_name)

    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> LLMResponse:
        """Generates a text completion for a given prompt and system prompt."""
        pass

    def __call__(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> str:
        res = self.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature)
        return res.text
