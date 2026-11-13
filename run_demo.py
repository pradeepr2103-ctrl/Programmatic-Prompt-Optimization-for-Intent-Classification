"""
Interactive Live Intent Classification Demonstration CLI.
Allows users/evaluators to input any customer inquiry and view predictions,
reasoning, and prompt structure across all optimized classifiers.
"""

import sys
from src.llm_backend.api_llm import get_llm
from src.baselines.zero_shot import ManualZeroShotClassifier
from src.baselines.few_shot import ManualFewShotClassifier
from src.optimizers.dspy_engine import DSPyIntentClassifier
from src.optimizers.textgrad_engine import TextGradClassifier
from src.optimizers.adalflow_engine import AdalFlowClassifier
from data.banking77_subsets.dataset_loader import get_dataset


def run_interactive_demo():
    dataset = get_dataset()
    intents = dataset.get_intents()

    print("================================================================================")
    print(" LIVE DEMO: Programmatic Prompt Optimization Intent Classifiers")
    print(" 10 Supported Intents:")
    for i, intent in enumerate(intents, 1):
        print(f"  {i}. {intent}")
    print("================================================================================")
    print("Initializing and optimizing models...")

    llm = get_llm(model_type="auto")

    # Initialize models
    zero_shot = ManualZeroShotClassifier(llm=llm)
    few_shot = ManualFewShotClassifier(llm=llm, shots_per_intent=1)

    print("- Compiling DSPy (COPRO)...")
    dspy_clf = DSPyIntentClassifier(llm=llm, optimizer_name="copro")
    dspy_clf.optimize()

    print("- Training TextGrad (TGD)...")
    textgrad_clf = TextGradClassifier(llm=llm)
    textgrad_clf.optimize(epochs=1, batch_size=4)

    print("- Optimizing AdalFlow Pipeline...")
    adalflow_clf = AdalFlowClassifier(llm=llm)
    adalflow_clf.optimize()

    print("\nModels initialized successfully!\n")

    sample_queries = [
        "Why was I billed an unexpected extra charge on my bank statement?",
        "I was at the restaurant and the waiter charged me $45 instead of $35 on my receipt.",
        "I dropped my wallet on the bus and someone has my card, please freeze it!",
        "When will my replacement debit card arrive in the mail?",
        "How do I link an external debit card to my app account?",
        "Why was my deposit of $200 cancelled and reverted back to my bank?",
    ]

    print("Sample Queries to Test:")
    for idx, q in enumerate(sample_queries, 1):
        print(f" [{idx}] {q}")
    print("-" * 80)

    for query in sample_queries:
        print(f"\nUser Query: \"{query}\"")
        p_zero = zero_shot.predict(query)
        p_few = few_shot.predict(query)
        p_dspy = dspy_clf.predict(query)
        p_textgrad = textgrad_clf.predict(query)
        p_adalflow = adalflow_clf.predict(query)

        print(f"  Manual Zero-Shot:    {p_zero}")
        print(f"  Manual Few-Shot:     {p_few}")
        print(f"  DSPy (COPRO):        {p_dspy}")
        print(f"  TextGrad (TGD):      {p_textgrad}")
        print(f"  AdalFlow (Auto):     {p_adalflow}")


if __name__ == "__main__":
    run_interactive_demo()
