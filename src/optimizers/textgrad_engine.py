"""
TextGrad Optimization Engine (Textual Gradient Feedback).
Implements the TextGrad framework (Yuksekgonul et al., Stanford):
- Variable: Encapsulates text parameters and textual gradients.
- Computation Graph: Tracks forward generation, critic/loss evaluation, and backpropagation.
- Textual Gradient Descent (TGD): Applies accumulated textual gradients to optimize the prompt text.
"""

import copy
import re
import time
from typing import Dict, List, Optional, Tuple
from src.llm_backend.base import BaseLLM
from src.llm_backend.tracker import TokenTracker, estimate_tokens
from data.banking77_subsets.dataset_loader import get_dataset


# ==========================================
# 1. TextGrad Computational Graph Primitives
# ==========================================

class Variable:
    """A parameter node in the TextGrad computational graph holding textual data and gradients."""

    def __init__(self, value: str, role_description: str = "System Prompt"):
        self.value = value
        self.role_description = role_description
        self.gradients: List[str] = []

    def zero_grad(self):
        """Clears accumulated textual gradients."""
        self.gradients = []

    def add_gradient(self, grad_text: str):
        """Accumulates a textual gradient."""
        if grad_text and grad_text.strip():
            self.gradients.append(grad_text.strip())

    def __repr__(self) -> str:
        return f"Variable(role='{self.role_description}', value_len={len(self.value)}, num_grads={len(self.gradients)})"


# ==========================================
# 2. Textual Loss / Critic
# ==========================================

class TextualLossCritic:
    """Evaluates prediction errors and generates textual gradients (critiques) with respect to the prompt."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm

    def compute_gradient(
        self,
        variable: Variable,
        customer_query: str,
        predicted_intent: str,
        target_intent: str,
    ) -> str:
        """
        Textual Backward Pass:
        Queries the LLM to inspect why the current prompt produced an incorrect prediction
        and outputs a gradient (textual critique) on how the prompt must be altered.
        """
        critic_prompt = (
            f"You are a TextGrad Loss Critic evaluating an intent classification error.\n\n"
            f"CURRENT SYSTEM PROMPT:\n{variable.value}\n\n"
            f"CUSTOMER QUERY: \"{customer_query}\"\n"
            f"PREDICTED INTENT: '{predicted_intent}'\n"
            f"GROUND TRUTH INTENT: '{target_intent}'\n\n"
            "CRITIQUE TASK:\n"
            "Critique why the current system prompt led the classifier to confuse "
            f"'{target_intent}' with '{predicted_intent}'. State what specific instruction or "
            "disambiguation rule must be added or clarified in the prompt to prevent this failure."
        )

        critic_response = self.llm(
            prompt=critic_prompt,
            system_prompt="TextGrad Textual Gradient Critic",
            temperature=0.2,
        )
        return critic_response


# ==========================================
# 3. Textual Gradient Descent (TGD) Optimizer
# ==========================================

class TextualGradientDescent:
    """
    TGD Optimizer that updates the prompt Variable using accumulated textual gradients.
    Analogue of: W = W - eta * grad_W
    """

    def __init__(self, parameters: List[Variable], llm: BaseLLM):
        self.parameters = parameters
        self.llm = llm

    def step(self):
        """Performs a textual gradient update step on each variable with accumulated gradients."""
        for var in self.parameters:
            if not var.gradients:
                continue

            combined_gradients = "\n---\n".join(
                f"Gradient {i+1}:\n{g}" for i, g in enumerate(var.gradients)
            )

            update_prompt = (
                f"You are the TextGrad TGD Optimizer. Update the following text variable "
                f"({var.role_description}) by incorporating all accumulated textual gradients.\n\n"
                f"CURRENT TEXT:\n{var.value}\n\n"
                f"ACCUMULATED TEXTUAL GRADIENTS:\n{combined_gradients}\n\n"
                "INSTRUCTIONS FOR UPDATE:\n"
                "1. Address each failure case pointed out by the gradients.\n"
                "2. Add crisp disambiguation guidelines and clarify edge cases.\n"
                "3. Keep the prompt structured, concise, and focused on 10 banking intents.\n"
                "4. Return ONLY the improved system prompt without conversational preamble."
            )

            updated_text = self.llm(
                prompt=update_prompt,
                system_prompt="TextGrad Textual Gradient Descent Optimizer",
                temperature=0.2,
            )

            if len(updated_text.strip()) > 50:
                var.value = updated_text.strip()

    def zero_grad(self):
        for var in self.parameters:
            var.zero_grad()


# ==========================================
# 4. High-Level TextGrad Classifier
# ==========================================

class TextGradClassifier:
    """Classifier optimized via the TextGrad computational graph and textual gradient descent."""

    def __init__(self, llm: BaseLLM):
        self.llm = llm
        self.dataset = get_dataset()
        self.intents = self.dataset.get_intents()

        initial_prompt = (
            "You are an expert customer-support intent classifier for a retail bank.\n"
            "Classify the incoming customer query into one of the 10 banking intents:\n"
            "- card_arrival: Physical card delivery and shipment.\n"
            "- card_linking: Connecting external debit/credit cards.\n"
            "- exchange_rate: Foreign currency conversion and FX fees.\n"
            "- extra_charge_on_statement: Unrecognized or mystery fee on statement.\n"
            "- lost_or_stolen_card: Reporting lost/stolen card or freezing card.\n"
            "- top_up_reverted: Deposited funds cancelled or refunded.\n"
            "- transfer_fee_charged: Fee charged for money transfer.\n"
            "- verify_identity: KYC identity verification documents.\n"
            "- visa_or_mastercard: Card scheme brand inquiries.\n"
            "- wrong_amount_additional_charge: Merchant overcharge on transaction.\n\n"
            "Return only the exact intent label."
        )

        self.system_prompt_var = Variable(value=initial_prompt, role_description="System Prompt")
        self.loss_critic = TextualLossCritic(llm=self.llm)
        self.optimizer = TextualGradientDescent(parameters=[self.system_prompt_var], llm=self.llm)
        self.optimization_history: List[dict] = []

    def format_prompt(self, utterance: str) -> str:
        return f"Customer Query: \"{utterance}\"\nIntent:"

    def predict(self, utterance: str) -> str:
        prompt = self.format_prompt(utterance)
        raw_output = self.llm(
            prompt=prompt,
            system_prompt=self.system_prompt_var.value,
            temperature=0.0,
        )
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

    def optimize(
        self,
        trainset: Optional[List[dict]] = None,
        valset: Optional[List[dict]] = None,
        epochs: int = 2,
        batch_size: int = 8,
    ):
        """
        Executes TextGrad training:
        Forward pass -> Error identification -> Textual Backward Pass (Critic) -> TGD Step.
        """
        if trainset is None:
            trainset = self.dataset.get_train_examples()
        if valset is None:
            valset = self.dataset.get_val_examples()

        best_val_acc = 0.0
        best_prompt_val = self.system_prompt_var.value

        for epoch in range(epochs):
            self.optimizer.zero_grad()
            errors_encountered = 0

            # Forward pass over train set across intents
            for idx, item in enumerate(trainset):
                pred_intent = self.predict(item["text"])
                gold_intent = item["intent"]

                # Textual backward pass on misclassifications
                if pred_intent != gold_intent:
                    errors_encountered += 1
                    grad = self.loss_critic.compute_gradient(
                        variable=self.system_prompt_var,
                        customer_query=item["text"],
                        predicted_intent=pred_intent,
                        target_intent=gold_intent,
                    )
                    self.system_prompt_var.add_gradient(grad)

                    if len(self.system_prompt_var.gradients) >= 3:
                        break

            # If errors found, apply textual gradient descent
            if self.system_prompt_var.gradients:
                self.optimizer.step()
                self.optimizer.zero_grad()

            # Validation checkpoint
            val_preds = [self.predict(x["text"]) for x in valset]
            val_correct = sum(1 for yt, yp in zip([x["intent"] for x in valset], val_preds) if yt == yp)
            val_acc = val_correct / len(valset)

            self.optimization_history.append({
                "epoch": epoch + 1,
                "errors_in_batch": errors_encountered,
                "val_accuracy": val_acc,
            })

            if val_acc >= best_val_acc:
                best_val_acc = val_acc
                best_prompt_val = self.system_prompt_var.value

        # Set prompt to best checkpoint
        self.system_prompt_var.value = best_prompt_val

    def get_prompt_text(self) -> str:
        sample_query = self.format_prompt("When will my replacement card arrive?")
        return f"{self.system_prompt_var.value}\n\n{sample_query}"

    def get_prompt_stats(self) -> Dict[str, int]:
        full_text = self.get_prompt_text()
        return {
            "char_length": len(full_text),
            "word_count": len(full_text.split()),
            "estimated_tokens": estimate_tokens(full_text),
        }
