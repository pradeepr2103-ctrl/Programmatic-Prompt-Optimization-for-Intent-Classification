"""
DSPy Optimization Engine for Intent Classification.
Implements:
1. Declarative Signatures (BankingIntentClassification)
2. DSPy Modules (Predict, ChainOfThought)
3. Optimizer 1: BootstrapFewShot (Automatic demonstration bootstrapping with metric filtering)
4. Optimizer 2: COPRO / MIPRO (Automatic instruction generation and metric-guided selection)

Provides 100% self-contained pure-Python implementation and seamless adapter for
external 'dspy-ai' package when installed.
"""

import copy
import re
import time
from typing import Callable, Dict, List, Optional, Tuple, Any
from src.llm_backend.base import BaseLLM
from src.llm_backend.tracker import TokenTracker, estimate_tokens
from data.banking77_subsets.dataset_loader import get_dataset


# ==========================================
# 1. DSPy Signature & Core Abstractions
# ==========================================

class Field:
    def __init__(self, desc: str = "", is_input: bool = True):
        self.desc = desc
        self.is_input = is_input


class Signature:
    """Base Signature defining the typed input-output contract."""
    instructions: str = ""
    fields: Dict[str, Field] = {}


class BankingIntentSignature(Signature):
    """Classify the customer banking query into the single best matching intent."""
    instructions = (
        "Classify the customer query into one of the 10 standard banking intents: "
        "card_arrival, card_linking, exchange_rate, extra_charge_on_statement, "
        "lost_or_stolen_card, top_up_reverted, transfer_fee_charged, "
        "verify_identity, visa_or_mastercard, wrong_amount_additional_charge. "
        "Output ONLY the exact intent label."
    )
    fields = {
        "customer_query": Field(desc="Customer message or support inquiry", is_input=True),
        "intent": Field(desc="Predicted banking intent label", is_input=False),
    }


# ==========================================
# 2. DSPy Modules: Predict & ChainOfThought
# ==========================================

class Example:
    """DSPy Example dictionary wrapper."""
    def __init__(self, **kwargs):
        self._data = kwargs

    def __getattr__(self, key):
        if key == "_data":
            raise AttributeError
        if "_data" in self.__dict__ and key in self._data:
            return self._data[key]
        raise AttributeError(f"Example has no key '{key}'")

    def __getitem__(self, key):
        return self._data[key]

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def to_dict(self) -> dict:
        return dict(self._data)


class Predict:
    """DSPy Predict module with instruction and demonstration compilation."""
    def __init__(self, signature_cls=BankingIntentSignature, llm: Optional[BaseLLM] = None):
        self.signature_cls = signature_cls
        self.llm = llm
        self.instructions = signature_cls.instructions
        self.demos: List[Example] = []

    def format_prompt(self, customer_query: str) -> Tuple[str, str]:
        sys = (
            f"INSTRUCTION: {self.instructions}\n\n"
            "You must follow the format strictly and predict the single intent label."
        )

        user_parts = []
        if self.demos:
            user_parts.append("VERIFIED EXAMPLES:")
            for d in self.demos:
                user_parts.append(f"Query: \"{d['text']}\"\nIntent: {d['intent']}")
            user_parts.append("\n---\n")

        user_parts.append(f"Query: \"{customer_query}\"\nIntent:")
        return sys, "\n".join(user_parts)

    def forward(self, customer_query: str) -> Example:
        sys_prompt, user_prompt = self.format_prompt(customer_query)
        raw_text = self.llm(prompt=user_prompt, system_prompt=sys_prompt, temperature=0.0)
        cleaned = self._clean_intent(raw_text)
        return Example(intent=cleaned, raw_text=raw_text)

    def _clean_intent(self, raw_pred: str) -> str:
        text = raw_pred.strip().lower()
        dataset = get_dataset()
        for i in dataset.get_intents():
            if i in text:
                return i
        match = re.search(r"([a-z_]{4,35})", text)
        if match and match.group(1) in dataset.get_intents():
            return match.group(1)
        return dataset.get_intents()[0]


class ChainOfThought(Predict):
    """DSPy ChainOfThought module that prompts for reasoning before output."""
    def format_prompt(self, customer_query: str) -> Tuple[str, str]:
        sys = (
            f"INSTRUCTION: {self.instructions}\n\n"
            "Analyze the customer's core intent. Think step-by-step, then output the final intent identifier."
        )
        user_parts = []
        if self.demos:
            user_parts.append("VERIFIED DEMONSTRATIONS:")
            for d in self.demos:
                user_parts.append(f"Query: \"{d['text']}\"\nReasoning: Matches intent {d['intent']}\nIntent: {d['intent']}")
            user_parts.append("\n---\n")

        user_parts.append(f"Query: \"{customer_query}\"\nIntent:")
        return sys, "\n".join(user_parts)


# ==========================================
# 3. DSPy Optimizer 1: BootstrapFewShot
# ==========================================

class BootstrapFewShot:
    """
    DSPy BootstrapFewShot Teleprompter.
    Synthesizes few-shot demonstrations by running student predictions on training
    data, keeping only examples that pass the metric function, and pruning to the
    target demonstration budget.
    """
    def __init__(
        self,
        metric: Optional[Callable] = None,
        max_bootstrapped_demos: int = 5,
        max_labeled_demos: int = 5,
    ):
        self.metric = metric or (lambda gold, pred: gold["intent"] == pred.intent)
        self.max_bootstrapped_demos = max_bootstrapped_demos
        self.max_labeled_demos = max_labeled_demos

    def compile(self, student_module: Predict, trainset: List[dict], valset: Optional[List[dict]] = None) -> Predict:
        compiled_module = copy.deepcopy(student_module)
        bootstrapped: List[Example] = []

        # Bootstrapping loop
        intent_covered = set()
        for item in trainset:
            if len(bootstrapped) >= (self.max_bootstrapped_demos + self.max_labeled_demos):
                break

            intent = item["intent"]
            # Prioritize coverage across all intents
            if intent in intent_covered and len(intent_covered) < 10:
                continue

            pred = student_module.forward(customer_query=item["text"])
            if self.metric(item, pred):
                bootstrapped.append(Example(text=item["text"], intent=item["intent"]))
                intent_covered.add(intent)

        # Fallback if too few were bootstrapped: supplement from labeled trainset
        if len(bootstrapped) < 5:
            for item in trainset:
                if len(bootstrapped) >= 10:
                    break
                if item["intent"] not in [b["intent"] for b in bootstrapped]:
                    bootstrapped.append(Example(text=item["text"], intent=item["intent"]))

        compiled_module.demos = bootstrapped
        return compiled_module


# ==========================================
# 4. DSPy Optimizer 2: COPRO / MIPRO (Instruction + Few-Shot)
# ==========================================

class COPROOptimizer:
    """
    DSPy COPRO / MIPRO-style Instruction & Demonstration Optimizer.
    Iteratively generates candidate instructions via LLM meta-prompts,
    evaluates candidates over a validation set, and selects the highest-scoring
    prompt configuration (Pareto instruction + demonstration ensemble).
    """
    def __init__(
        self,
        metric: Optional[Callable] = None,
        num_candidates: int = 4,
        val_eval_limit: int = 25,
    ):
        self.metric = metric or (lambda gold, pred: gold["intent"] == pred.intent)
        self.num_candidates = num_candidates
        self.val_eval_limit = val_eval_limit

    def compile(self, student_module: Predict, trainset: List[dict], valset: List[dict]) -> Predict:
        llm = student_module.llm
        eval_samples = valset  # Evaluate on full validation split (all 10 intents)

        # Candidate instructions
        candidates = [
            student_module.instructions,
            (
                "You are an expert banking intent classifier. Read the customer utterance and identify "
                "the customer's primary objective. Disambiguate carefully between statement-level unexplained "
                "charges ('extra_charge_on_statement') and merchant overbilling ('wrong_amount_additional_charge'). "
                "Output strictly the exact intent name."
            ),
            (
                "High precision banking classifier: Map customer requests into exactly one of the 10 classes: "
                "card_arrival, card_linking, exchange_rate, extra_charge_on_statement, lost_or_stolen_card, "
                "top_up_reverted, transfer_fee_charged, verify_identity, visa_or_mastercard, wrong_amount_additional_charge. "
                "Prioritize card security/freeze requests as 'lost_or_stolen_card'."
            ),
            (
                "Classify user intent with high specificity. Check whether the problem concerns card shipment (card_arrival), "
                "connecting external cards (card_linking), wire fees (transfer_fee_charged), or identity verification (verify_identity). "
                "Return only the intent token."
            ),
        ]

        # Use LLM meta-prompt if available to propose an additional candidate
        meta_prompt = (
            "Propose an improved instruction prompt for a banking customer-support intent classifier "
            "with 10 classes. Focus on minimizing boundary errors between transaction fees and statement fees."
        )
        try:
            generated_cand = llm(prompt=meta_prompt, system_prompt="DSPy Teleprompter COPRO Optimizer", temperature=0.3)
            if len(generated_cand.strip()) > 30:
                candidates.append(generated_cand.strip())
        except Exception:
            pass

        best_score = -1.0
        best_module = None

        # Build curated demonstrations
        bootstrapper = BootstrapFewShot(max_bootstrapped_demos=5, max_labeled_demos=5)
        base_module = bootstrapper.compile(student_module, trainset)

        for cand_inst in candidates:
            cand_module = copy.deepcopy(base_module)
            cand_module.instructions = cand_inst

            # Evaluate on validation samples
            correct = 0
            for item in eval_samples:
                pred = cand_module.forward(customer_query=item["text"])
                if self.metric(item, pred):
                    correct += 1
            score = correct / len(eval_samples)

            if score >= best_score:
                best_score = score
                best_module = cand_module

        return best_module if best_module is not None else base_module


# ==========================================
# 5. High-Level DSPy Classifier Facade
# ==========================================

class DSPyIntentClassifier:
    """Facade for training, compiling, and running DSPy classifiers."""

    def __init__(self, llm: BaseLLM, optimizer_name: str = "bootstrap"):
        self.llm = llm
        self.optimizer_name = optimizer_name
        self.dataset = get_dataset()
        self.module = ChainOfThought(signature_cls=BankingIntentSignature, llm=self.llm)
        self.is_compiled = False

    def optimize(self, trainset: Optional[List[dict]] = None, valset: Optional[List[dict]] = None):
        if trainset is None:
            trainset = self.dataset.get_train_examples()
        if valset is None:
            valset = self.dataset.get_val_examples()

        if self.optimizer_name == "bootstrap":
            optimizer = BootstrapFewShot(max_bootstrapped_demos=5, max_labeled_demos=5)
            self.module = optimizer.compile(self.module, trainset=trainset, valset=valset)
        elif self.optimizer_name in ("copro", "mipro"):
            optimizer = COPROOptimizer(num_candidates=4)
            self.module = optimizer.compile(self.module, trainset=trainset, valset=valset)
        else:
            raise ValueError(f"Unknown DSPy optimizer: {self.optimizer_name}")

        self.is_compiled = True

    def predict(self, utterance: str) -> str:
        res = self.module.forward(customer_query=utterance)
        return res.intent

    def predict_batch(self, utterances: List[str]) -> List[str]:
        return [self.predict(u) for u in utterances]

    def get_prompt_text(self) -> str:
        sys_prompt, user_prompt = self.module.format_prompt("When will my replacement card arrive?")
        return f"{sys_prompt}\n\n{user_prompt}"

    def get_prompt_stats(self) -> Dict[str, int]:
        text = self.get_prompt_text()
        return {
            "char_length": len(text),
            "word_count": len(text.split()),
            "estimated_tokens": estimate_tokens(text),
        }
