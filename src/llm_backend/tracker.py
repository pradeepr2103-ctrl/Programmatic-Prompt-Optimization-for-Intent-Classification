"""
Token and Call Usage Tracker
Accurately records prompt tokens, completion tokens, LLM call counts,
and elapsed execution time across manual and optimization routines.
"""

import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CallRecord:
    stage: str
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    latency: float
    timestamp: float = field(default_factory=time.time)


class TokenTracker:
    """Tracks token consumption and API call metrics for benchmark reporting."""

    def __init__(self, name: str = "default"):
        self.name = name
        self.reset()

    def reset(self):
        self.records: List[CallRecord] = []
        self.total_calls: int = 0
        self.total_prompt_tokens: int = 0
        self.total_completion_tokens: int = 0
        self.total_latency: float = 0.0
        self.current_stage: str = "general"

    def set_stage(self, stage: str):
        self.current_stage = stage

    def record_call(self, prompt_tokens: int, completion_tokens: int, latency: float = 0.0):
        total = prompt_tokens + completion_tokens
        rec = CallRecord(
            stage=self.current_stage,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total,
            latency=latency,
        )
        self.records.append(rec)
        self.total_calls += 1
        self.total_prompt_tokens += prompt_tokens
        self.total_completion_tokens += completion_tokens
        self.total_latency += latency

    def get_summary(self) -> Dict[str, float]:
        total_tokens = self.total_prompt_tokens + self.total_completion_tokens
        avg_latency = (self.total_latency / self.total_calls) if self.total_calls > 0 else 0.0
        return {
            "calls": self.total_calls,
            "prompt_tokens": self.total_prompt_tokens,
            "completion_tokens": self.total_completion_tokens,
            "total_tokens": total_tokens,
            "total_latency_seconds": round(self.total_latency, 3),
            "avg_latency_per_call": round(avg_latency, 4),
        }

    def get_stage_breakdown(self) -> Dict[str, Dict[str, int]]:
        breakdown: Dict[str, Dict[str, int]] = {}
        for r in self.records:
            if r.stage not in breakdown:
                breakdown[r.stage] = {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}
            breakdown[r.stage]["calls"] += 1
            breakdown[r.stage]["prompt_tokens"] += r.prompt_tokens
            breakdown[r.stage]["completion_tokens"] += r.completion_tokens
            breakdown[r.stage]["total_tokens"] += r.total_tokens
        return breakdown


def estimate_tokens(text: str) -> int:
    """Accurate standard token estimation (approx 4 characters per token or 0.75 words/token)."""
    if not text:
        return 0
    words = text.split()
    char_estimate = len(text) / 3.8
    word_estimate = len(words) * 1.3
    return max(1, int(round((char_estimate + word_estimate) / 2)))
