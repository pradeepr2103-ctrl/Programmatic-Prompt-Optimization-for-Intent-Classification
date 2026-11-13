"""
Main Experiment Benchmark Execution Script.
Runs the complete programmatic prompt optimization suite across:
1. Manual Zero-Shot Baseline
2. Manual Few-Shot Baseline
3. DSPy (BootstrapFewShot)
4. DSPy (COPRO/MIPRO)
5. TextGrad (Textual-Gradient Feedback)
6. AdalFlow (Declarative Pipeline with Auto-Optimization)

Exports structured tables, JSON metrics, comprehensive Markdown report, and HTML dashboard.
"""

import argparse
import sys
from src.llm_backend.api_llm import get_llm
from src.evaluation.evaluator import BenchmarkRunner
from src.evaluation.report_generator import ReportGenerator
from src.visualization.charts import generate_html_dashboard


def main():
    parser = argparse.ArgumentParser(description="Run Programmatic Prompt Optimization Benchmark")
    parser.add_argument(
        "--model-type",
        type=str,
        default="auto",
        choices=["auto", "mock", "openai", "gemini"],
        help="LLM backend type (default: auto, falls back gracefully to offline simulator)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results",
        help="Directory where reports, tables and charts will be stored",
    )
    args = parser.parse_args()

    print("================================================================================")
    print(" PROGRAMMATIC PROMPT OPTIMIZATION FOR INTENT CLASSIFICATION")
    print(" Comparative Study: Manual Baselines vs DSPy, TextGrad & AdalFlow")
    print(" Dataset: Banking77 (10-Intent Customer Support Subset)")
    print("================================================================================")

    # Initialize benchmark runner
    runner = BenchmarkRunner(llm_factory_fn=lambda tracker: get_llm(model_type=args.model_type, tracker=tracker))

    # Execute all benchmarks
    results = runner.run_all()

    # Generate multi-format reports
    print("\nGenerating comprehensive benchmark reports...")
    report_gen = ReportGenerator(results=results, output_dir=args.output_dir)
    report_gen.export_all()

    # Generate visual HTML dashboard
    dashboard_path = generate_html_dashboard(results=results, output_path=f"{args.output_dir}/benchmark_dashboard.html")
    print(f"Interactive visual dashboard created at: {dashboard_path}")
    print("All tasks finished successfully!\n")


if __name__ == "__main__":
    main()
