"""
Standardized Evaluation Engine.
Executes the full comparative benchmark across manual baselines and
programmatically optimized classifiers under identical holdout test conditions.
"""

import time
from typing import Dict, List, Optional
from src.llm_backend.base import BaseLLM
from src.llm_backend.tracker import TokenTracker
from src.llm_backend.mock_llm import MockLLM
from src.metrics.classification_metrics import compute_metrics
from src.baselines.zero_shot import ManualZeroShotClassifier
from src.baselines.few_shot import ManualFewShotClassifier
from src.optimizers.dspy_engine import DSPyIntentClassifier
from src.optimizers.textgrad_engine import TextGradClassifier
from src.optimizers.adalflow_engine import AdalFlowClassifier
from data.banking77_subsets.dataset_loader import get_dataset


class BenchmarkRunner:
    """Orchestrates comparative benchmarking across all classifiers."""

    def __init__(self, llm_factory_fn=None):
        self.llm_factory = llm_factory_fn or (lambda tracker: MockLLM(tracker=tracker))
        self.dataset = get_dataset()
        self.train_data = self.dataset.get_train_examples()
        self.val_data = self.dataset.get_val_examples()
        self.test_data = self.dataset.get_test_examples()
        self.test_texts = [x["text"] for x in self.test_data]
        self.test_labels = [x["intent"] for x in self.test_data]

    def run_all(self) -> Dict[str, dict]:
        """Runs the complete suite of manual and programmatic methods."""
        results = {}

        # 1. Manual Zero-Shot
        print("\n[1/6] Running Manual Zero-Shot Classifier...")
        results["Manual Zero-Shot"] = self._run_zero_shot()

        # 2. Manual Few-Shot
        print("\n[2/6] Running Manual Few-Shot Classifier...")
        results["Manual Few-Shot"] = self._run_few_shot()

        # 3. DSPy BootstrapFewShot
        print("\n[3/6] Running DSPy (BootstrapFewShot Optimizer)...")
        results["DSPy (BootstrapFewShot)"] = self._run_dspy_bootstrap()

        # 4. DSPy COPRO / MIPRO
        print("\n[4/6] Running DSPy (COPRO Instruction Optimizer)...")
        results["DSPy (COPRO/MIPRO)"] = self._run_dspy_copro()

        # 5. TextGrad (Textual-Gradient Feedback)
        print("\n[5/6] Running TextGrad (Textual-Gradient Feedback)...")
        results["TextGrad (TGD)"] = self._run_textgrad()

        # 6. AdalFlow (Declarative Auto-Optimization)
        print("\n[6/6] Running AdalFlow (Auto-Optimization Pipeline)...")
        results["AdalFlow (Auto-Optimizer)"] = self._run_adalflow()

        print("\nBenchmark complete!")
        return results

    def _run_zero_shot(self) -> dict:
        tracker = TokenTracker(name="ZeroShotTracker")
        llm = self.llm_factory(tracker)
        clf = ManualZeroShotClassifier(llm=llm)

        start_eval = time.time()
        tracker.set_stage("evaluation")
        preds = clf.predict_batch(self.test_texts)
        eval_time = time.time() - start_eval

        metrics = compute_metrics(y_true=self.test_labels, y_pred=preds, labels=self.dataset.get_intents())
        prompt_stats = clf.get_prompt_stats()
        usage = tracker.get_summary()

        return {
            "method": "Manual Zero-Shot",
            "category": "Baseline",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "prompt_length_chars": prompt_stats["char_length"],
            "prompt_length_tokens": prompt_stats["estimated_tokens"],
            "opt_llm_calls": 0,
            "opt_prompt_tokens": 0,
            "opt_completion_tokens": 0,
            "opt_total_tokens": 0,
            "opt_runtime_sec": 0.0,
            "test_eval_time_sec": round(eval_time, 3),
            "total_calls": usage["calls"],
            "total_tokens": usage["total_tokens"],
            "metrics": metrics,
        }

    def _run_few_shot(self) -> dict:
        tracker = TokenTracker(name="FewShotTracker")
        llm = self.llm_factory(tracker)
        clf = ManualFewShotClassifier(llm=llm, shots_per_intent=1)

        start_eval = time.time()
        tracker.set_stage("evaluation")
        preds = clf.predict_batch(self.test_texts)
        eval_time = time.time() - start_eval

        metrics = compute_metrics(y_true=self.test_labels, y_pred=preds, labels=self.dataset.get_intents())
        prompt_stats = clf.get_prompt_stats()
        usage = tracker.get_summary()

        return {
            "method": "Manual Few-Shot",
            "category": "Baseline",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "prompt_length_chars": prompt_stats["char_length"],
            "prompt_length_tokens": prompt_stats["estimated_tokens"],
            "opt_llm_calls": 0,
            "opt_prompt_tokens": 0,
            "opt_completion_tokens": 0,
            "opt_total_tokens": 0,
            "opt_runtime_sec": 0.0,
            "test_eval_time_sec": round(eval_time, 3),
            "total_calls": usage["calls"],
            "total_tokens": usage["total_tokens"],
            "metrics": metrics,
        }

    def _run_dspy_bootstrap(self) -> dict:
        tracker = TokenTracker(name="DSPyBootstrapTracker")
        llm = self.llm_factory(tracker)
        clf = DSPyIntentClassifier(llm=llm, optimizer_name="bootstrap")

        tracker.set_stage("optimization")
        start_opt = time.time()
        clf.optimize(trainset=self.train_data, valset=self.val_data)
        opt_time = time.time() - start_opt

        opt_breakdown = tracker.get_stage_breakdown().get("optimization", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        tracker.set_stage("evaluation")
        start_eval = time.time()
        preds = clf.predict_batch(self.test_texts)
        eval_time = time.time() - start_eval

        metrics = compute_metrics(y_true=self.test_labels, y_pred=preds, labels=self.dataset.get_intents())
        prompt_stats = clf.get_prompt_stats()
        usage = tracker.get_summary()

        return {
            "method": "DSPy (BootstrapFewShot)",
            "category": "Programmatic Optimizer",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "prompt_length_chars": prompt_stats["char_length"],
            "prompt_length_tokens": prompt_stats["estimated_tokens"],
            "opt_llm_calls": opt_breakdown["calls"],
            "opt_prompt_tokens": opt_breakdown["prompt_tokens"],
            "opt_completion_tokens": opt_breakdown["completion_tokens"],
            "opt_total_tokens": opt_breakdown["total_tokens"],
            "opt_runtime_sec": round(opt_time, 3),
            "test_eval_time_sec": round(eval_time, 3),
            "total_calls": usage["calls"],
            "total_tokens": usage["total_tokens"],
            "metrics": metrics,
        }

    def _run_dspy_copro(self) -> dict:
        tracker = TokenTracker(name="DSPyCOPROTracker")
        llm = self.llm_factory(tracker)
        clf = DSPyIntentClassifier(llm=llm, optimizer_name="copro")

        tracker.set_stage("optimization")
        start_opt = time.time()
        clf.optimize(trainset=self.train_data, valset=self.val_data)
        opt_time = time.time() - start_opt

        opt_breakdown = tracker.get_stage_breakdown().get("optimization", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        tracker.set_stage("evaluation")
        start_eval = time.time()
        preds = clf.predict_batch(self.test_texts)
        eval_time = time.time() - start_eval

        metrics = compute_metrics(y_true=self.test_labels, y_pred=preds, labels=self.dataset.get_intents())
        prompt_stats = clf.get_prompt_stats()
        usage = tracker.get_summary()

        return {
            "method": "DSPy (COPRO/MIPRO)",
            "category": "Programmatic Optimizer",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "prompt_length_chars": prompt_stats["char_length"],
            "prompt_length_tokens": prompt_stats["estimated_tokens"],
            "opt_llm_calls": opt_breakdown["calls"],
            "opt_prompt_tokens": opt_breakdown["prompt_tokens"],
            "opt_completion_tokens": opt_breakdown["completion_tokens"],
            "opt_total_tokens": opt_breakdown["total_tokens"],
            "opt_runtime_sec": round(opt_time, 3),
            "test_eval_time_sec": round(eval_time, 3),
            "total_calls": usage["calls"],
            "total_tokens": usage["total_tokens"],
            "metrics": metrics,
        }

    def _run_textgrad(self) -> dict:
        tracker = TokenTracker(name="TextGradTracker")
        llm = self.llm_factory(tracker)
        clf = TextGradClassifier(llm=llm)

        tracker.set_stage("optimization")
        start_opt = time.time()
        clf.optimize(trainset=self.train_data, valset=self.val_data, epochs=2, batch_size=8)
        opt_time = time.time() - start_opt

        opt_breakdown = tracker.get_stage_breakdown().get("optimization", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        tracker.set_stage("evaluation")
        start_eval = time.time()
        preds = clf.predict_batch(self.test_texts)
        eval_time = time.time() - start_eval

        metrics = compute_metrics(y_true=self.test_labels, y_pred=preds, labels=self.dataset.get_intents())
        prompt_stats = clf.get_prompt_stats()
        usage = tracker.get_summary()

        return {
            "method": "TextGrad (TGD)",
            "category": "Programmatic Optimizer",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "prompt_length_chars": prompt_stats["char_length"],
            "prompt_length_tokens": prompt_stats["estimated_tokens"],
            "opt_llm_calls": opt_breakdown["calls"],
            "opt_prompt_tokens": opt_breakdown["prompt_tokens"],
            "opt_completion_tokens": opt_breakdown["completion_tokens"],
            "opt_total_tokens": opt_breakdown["total_tokens"],
            "opt_runtime_sec": round(opt_time, 3),
            "test_eval_time_sec": round(eval_time, 3),
            "total_calls": usage["calls"],
            "total_tokens": usage["total_tokens"],
            "metrics": metrics,
        }

    def _run_adalflow(self) -> dict:
        tracker = TokenTracker(name="AdalFlowTracker")
        llm = self.llm_factory(tracker)
        clf = AdalFlowClassifier(llm=llm)

        tracker.set_stage("optimization")
        start_opt = time.time()
        clf.optimize(trainset=self.train_data, valset=self.val_data)
        opt_time = time.time() - start_opt

        opt_breakdown = tracker.get_stage_breakdown().get("optimization", {"calls": 0, "prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

        tracker.set_stage("evaluation")
        start_eval = time.time()
        preds = clf.predict_batch(self.test_texts)
        eval_time = time.time() - start_eval

        metrics = compute_metrics(y_true=self.test_labels, y_pred=preds, labels=self.dataset.get_intents())
        prompt_stats = clf.get_prompt_stats()
        usage = tracker.get_summary()

        return {
            "method": "AdalFlow (Auto-Optimizer)",
            "category": "Programmatic Optimizer",
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "prompt_length_chars": prompt_stats["char_length"],
            "prompt_length_tokens": prompt_stats["estimated_tokens"],
            "opt_llm_calls": opt_breakdown["calls"],
            "opt_prompt_tokens": opt_breakdown["prompt_tokens"],
            "opt_completion_tokens": opt_breakdown["completion_tokens"],
            "opt_total_tokens": opt_breakdown["total_tokens"],
            "opt_runtime_sec": round(opt_time, 3),
            "test_eval_time_sec": round(eval_time, 3),
            "total_calls": usage["calls"],
            "total_tokens": usage["total_tokens"],
            "metrics": metrics,
        }
