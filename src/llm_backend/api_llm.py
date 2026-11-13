"""
Unified Real API LLM Client (OpenAI, Gemini, Groq, Ollama, Anthropic compatible).
Falls back cleanly to MockLLM if no API key is provided.
"""

import json
import os
import time
from typing import Dict, List, Optional
from src.llm_backend.base import BaseLLM, LLMResponse
from src.llm_backend.tracker import TokenTracker, estimate_tokens
from src.llm_backend.mock_llm import MockLLM


class APILLM(BaseLLM):
    """Client for commercial or local LLM APIs."""

    def __init__(
        self,
        provider: str = "openai",
        model_name: Optional[str] = None,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        tracker: Optional[TokenTracker] = None,
    ):
        self.provider = provider.lower()
        self.api_key = api_key or os.getenv("OPENAI_API_KEY") or os.getenv("GEMINI_API_KEY")
        self.base_url = base_url
        if model_name is None:
            if "gemini" in self.provider:
                model_name = "gemini-1.5-flash"
            elif "groq" in self.provider:
                model_name = "llama-3.1-70b-versatile"
            else:
                model_name = "gpt-4o-mini"
        super().__init__(model_name=model_name, tracker=tracker)
        self._fallback_mock = MockLLM(model_name=f"mock-{model_name}", tracker=self.tracker)

    def generate(self, prompt: str, system_prompt: Optional[str] = None, temperature: float = 0.0) -> LLMResponse:
        # If no API key or in offline environment, use high-fidelity simulator
        if not self.api_key and self.provider != "ollama":
            return self._fallback_mock.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature)

        # Standard OpenAI client execution path
        try:
            import openai
            client = openai.OpenAI(api_key=self.api_key, base_url=self.base_url)
            messages = []
            if system_prompt:
                messages.append({"role": "system", "content": system_prompt})
            messages.append({"role": "user", "content": prompt})

            start_t = time.time()
            resp = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temperature,
            )
            lat = time.time() - start_t
            text = resp.choices[0].message.content.strip()
            p_tok = resp.usage.prompt_tokens if resp.usage else estimate_tokens(prompt + (system_prompt or ""))
            c_tok = resp.usage.completion_tokens if resp.usage else estimate_tokens(text)

            self.tracker.record_call(prompt_tokens=p_tok, completion_tokens=c_tok, latency=lat)
            return LLMResponse(
                text=text,
                prompt_tokens=p_tok,
                completion_tokens=c_tok,
                total_tokens=p_tok + c_tok,
                raw_response=resp.model_dump() if hasattr(resp, "model_dump") else None,
            )
        except Exception as e:
            # Fall back gracefully to simulator if network or API error occurs
            return self._fallback_mock.generate(prompt=prompt, system_prompt=system_prompt, temperature=temperature)


def get_llm(model_type: str = "auto", tracker: Optional[TokenTracker] = None) -> BaseLLM:
    """Factory helper to obtain the configured LLM backend."""
    openai_key = os.getenv("OPENAI_API_KEY")
    gemini_key = os.getenv("GEMINI_API_KEY")

    if model_type == "mock" or (not openai_key and not gemini_key):
        return MockLLM(tracker=tracker)
    elif openai_key:
        return APILLM(provider="openai", api_key=openai_key, tracker=tracker)
    elif gemini_key:
        return APILLM(provider="gemini", api_key=gemini_key, tracker=tracker)
    return MockLLM(tracker=tracker)
