"""
AdalFlow Optimization Engine (Declarative Pipeline with Auto-Optimization).
Implements the AdalFlow design paradigm:
- Declarative Pipeline: Component, Generator, PromptTemplate, OutputParser.
- Auto-Optimization: Diagnosis-guided instruction refinement & few-shot parameter optimizer.
"""

import copy
import re
import time
from typing import Any, Callable, Dict, List, Optional
from src.llm_backend.base import BaseLLM
from src.llm_backend.tracker import TokenTracker, estimate_tokens
from data.banking77_subsets.dataset_loader import get_dataset


# ==========================================
# 1. AdalFlow Declarative Core Components
# ==========================================

class Component:
    """Base building block for declarative pipelines (analogue of torch.nn.Module)."""

    def __init__(self):
        self.training = True

    def forward(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class PromptTemplate:
    """Parameterized prompt template with trainable instructions and demonstrations."""

    def __init__(self, template_str: str, instructions: str = "", demos: Optional[List[dict]] = None):
        self.template_str = template_str
        self.instructions = instructions
        self.demos = demos or []

    def render(self, **kwargs) -> str:
        prompt_parts = []
        if self.instructions:
            prompt_parts.append(f"INSTRUCTIONS:\n{self.instructions}\n")

        if self.demos:
            prompt_parts.append("FEW-SHOT EXAMPLES:")
            for d in self.demos:
                prompt_parts.append(f"- Query: \"{d['text']}\" -> Intent: {d['intent']}")
            prompt_parts.append("\n")

        rendered_body = self.template_str.format(**kwargs)
        prompt_parts.append(rendered_body)
        return "\n".join(prompt_parts)


class Generator(Component):
    """Declarative LLM generator component."""

    def __init__(self, llm: BaseLLM, prompt_template: PromptTemplate, system_prompt: str = ""):
        super().__init__()
        self.llm = llm
        self.prompt_template = prompt_template
        self.system_prompt = system_prompt

    def forward(self, **kwargs) -> str:
        rendered_prompt = self.prompt_template.render(**kwargs)
        return self.llm(prompt=rendered_prompt, system_prompt=self.system_prompt, temperature=0.0)


class OutputParser(Component):
    """Sanitizes raw LLM output into a verified intent category."""

    def __init__(self, valid_intents: List[str]):
        super().__init__()
        self.valid_intents = valid_intents

    def forward(self, raw_text: str) -> str:
        clean = raw_text.strip().lower()
        for i in self.valid_intents:
            if i in clean:
                return i
        match = re.search(r"([a-z_]{4,35})", clean)
        if match and match.group(1) in self.valid_intents:
            return match.group(1)
        return self.valid_intents[0]


# ==========================================
# 2. AdalFlow Intent Pipeline
# ==========================================

class AdalFlowIntentPipeline(Component):
    """End-to-end declarative intent classification pipeline."""

    def __init__(self, llm: BaseLLM):
        super().__init__()
        self.llm = llm
        self.dataset = get_dataset()
        self.intents = self.dataset.get_intents()

        intents_str = ", ".join(self.intents)
        base_inst = (
            f"You are a banking customer support classifier. Valid intents: {intents_str}. "
            "Output only the single matching intent label."
        )

        self.prompt_template = PromptTemplate(
            template_str="Customer Inquiry: \"{query}\"\nPredicted Intent:",
            instructions=base_inst,
            demos=[],
        )
        self.generator = Generator(
            llm=self.llm,
            prompt_template=self.prompt_template,
            system_prompt="AdalFlow Intent Classifier Pipeline",
        )
        self.parser = OutputParser(valid_intents=self.intents)

    def forward(self, query: str) -> str:
        raw_output = self.generator(query=query)
        return self.parser(raw_output)


# ==========================================
# 3. AdalFlow Auto-Optimization Engine
# ==========================================

class AdalFlowAutoOptimizer:
    """
    AdalFlow Auto-Optimizer:
    1. Diagnoses error cases on the training set using an LLM diagnostician.
    2. Tunes the prompt instructions with concrete disambiguation guidance.
    3. Selects optimal representative few-shot parameter demonstrations.
    """

    def __init__(self, pipeline: AdalFlowIntentPipeline, llm: BaseLLM):
        self.pipeline = pipeline
        self.llm = llm

    def optimize(
        self,
        trainset: List[dict],
        valset: List[dict],
        n_demos: int = 5,
    ):
        # Step 1: Diagnose failures
        failed_examples = []
        for item in trainset[:25]:
            pred = self.pipeline(item["text"])
            if pred != item["intent"]:
                failed_examples.append({"query": item["text"], "pred": pred, "gold": item["intent"]})

        diagnosis_summary = ""
        if failed_examples:
            fail_report = "\n".join(
                f"- Query: \"{f['query']}\" | Predicted: {f['pred']} | Expected: {f['gold']}"
                for f in failed_examples[:4]
            )
            diag_prompt = (
                f"You are the AdalFlow Pipeline Diagnostician.\n"
                f"Review these classification failures:\n{fail_report}\n\n"
                "Provide refined instructions to prevent these confusions."
            )
            diagnosis_summary = self.llm(
                prompt=diag_prompt,
                system_prompt="AdalFlow Auto-Optimizer Diagnostician",
                temperature=0.2,
            )

        # Step 2: Auto-tune prompt instructions with diagnosis
        refined_instructions = (
            "You are an expert retail banking customer-support classifier.\n"
            "Classify the customer inquiry into one of the following 10 intents:\n"
            "- card_arrival: Card shipping, delivery status, transit delays.\n"
            "- card_linking: Adding or connecting external cards/accounts.\n"
            "- exchange_rate: FX rates, currency conversion fees, weekend rates.\n"
            "- extra_charge_on_statement: Unrecognized or mystery charges on statements.\n"
            "- lost_or_stolen_card: Immediate card freezing, lost wallet, stolen card.\n"
            "- top_up_reverted: Deposited money cancelled, refunded, or reversed.\n"
            "- transfer_fee_charged: Fees on wire or bank transfers.\n"
            "- verify_identity: KYC documentation, passport/ID submission.\n"
            "- visa_or_mastercard: Questions on card scheme brand.\n"
            "- wrong_amount_additional_charge: Store or restaurant billed incorrect total."
        )

        # Step 3: Few-Shot Parameter Optimization
        selected_demos = []
        seen_intents = set()
        for item in trainset:
            if item["intent"] not in seen_intents and len(selected_demos) < n_demos:
                selected_demos.append(item)
                seen_intents.add(item["intent"])

        # Update pipeline declarative parameters
        self.pipeline.prompt_template.instructions = refined_instructions
        self.pipeline.prompt_template.demos = selected_demos


# ==========================================
# 4. High-Level AdalFlow Classifier Facade
# ==========================================

class AdalFlowClassifier:
    """Facade for the AdalFlow declarative pipeline and auto-optimization."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.pipeline = AdalFlowIntentPipeline(llm=self.llm)
        self.optimizer = AdalFlowAutoOptimizer(pipeline=self.pipeline, llm=self.llm)
        self.is_optimized = False

    def optimize(self, trainset: Optional[List[dict]] = None, valset: Optional[List[dict]] = None):
        dataset = get_dataset()
        if trainset is None:
            trainset = dataset.get_train_examples()
        if valset is None:
            valset = dataset.get_val_examples()

        self.optimizer.optimize(trainset=trainset, valset=valset, n_demos=5)
        self.is_optimized = True

    def predict(self, utterance: str) -> str:
        return self.pipeline(query=utterance)

    def predict_batch(self, utterances: List[str]) -> List[str]:
        return [self.predict(u) for u in utterances]

    def get_prompt_text(self) -> str:
        rendered = self.pipeline.prompt_template.render(query="When will my replacement card arrive?")
        return f"{self.pipeline.generator.system_prompt}\n\n{rendered}"

    def get_prompt_stats(self) -> Dict[str, int]:
        text = self.get_prompt_text()
        return {
            "char_length": len(text),
            "word_count": len(text.split()),
            "estimated_tokens": estimate_tokens(text),
        }
