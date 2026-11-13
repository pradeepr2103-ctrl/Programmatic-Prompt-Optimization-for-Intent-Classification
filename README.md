<div align="center">

# 🧠 Programmatic Prompt Optimization for Intent Classification

### A Comparative Study of DSPy, TextGrad & AdalFlow on Banking77

[![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![Tests](https://img.shields.io/badge/Tests-14%20Passed-brightgreen?style=for-the-badge&logo=pytest&logoColor=white)](tests/)
[![Dataset](https://img.shields.io/badge/Dataset-Banking77-orange?style=for-the-badge&logo=databricks&logoColor=white)](data/)
[![License](https://img.shields.io/badge/License-MIT-yellow?style=for-the-badge)](LICENSE)
[![Framework](https://img.shields.io/badge/DSPy-Stanford-6c63ff?style=for-the-badge)](https://github.com/stanfordnlp/dspy)
[![Framework](https://img.shields.io/badge/TextGrad-Stanford-ff6b6b?style=for-the-badge)](https://github.com/zou-group/textgrad)
[![Framework](https://img.shields.io/badge/AdalFlow-Auto--Opt-00d4aa?style=for-the-badge)](https://github.com/SylphAI-Inc/AdalFlow)

<br/>

> **Mini Project #6 — Prompt Engineering | NLP | Large Language Models**
>
> Replacing manually written prompts for a customer-support intent classifier with
> programmatically optimized prompts using **DSPy**, **TextGrad**, and **AdalFlow**,
> and quantifying the improvement on a 10-intent subset of the **Banking77** dataset.

<br/>

```
╔══════════════════════════════════════════════════════════════════════╗
║           BENCHMARK RESULTS — HOLDOUT TEST SET (80 SAMPLES)         ║
╠══════════════════════════════╦═══════════╦══════════╦═══════════════╣
║  Classifier Method           ║  Accuracy ║ Macro F1 ║  Prompt Toks  ║
╠══════════════════════════════╬═══════════╬══════════╬═══════════════╣
║  Manual Zero-Shot (Baseline) ║   81.25%  ║  0.7821  ║  420 tokens   ║
║  Manual Few-Shot  (Baseline) ║   90.00%  ║  0.8962  ║  659 tokens   ║
║  DSPy  BootstrapFewShot      ║   90.00%  ║  0.8962  ║  404 tokens   ║
║  DSPy  COPRO / MIPRO       ⭐║   97.50%  ║  0.9749  ║  404 tokens   ║
║  TextGrad  TGD             ⭐║   97.50%  ║  0.9749  ║  309 tokens ✓ ║
║  AdalFlow  Auto-Optimizer  ⭐║   97.50%  ║  0.9749  ║  315 tokens   ║
╚══════════════════════════════╩═══════════╩══════════╩═══════════════╝
         +16.25% accuracy gain │ 53% shorter prompts │ 0.9749 Macro F1
```

</div>

---

## 📌 Table of Contents

1. [Project Overview](#-project-overview)
2. [Key Results](#-key-results)
3. [Dataset](#-dataset--banking77-10-intent-subset)
4. [Methodology](#-methodology)
   - [Manual Baselines](#1-manual-baselines)
   - [DSPy — BootstrapFewShot & COPRO](#2-dspy--signatures-modules--two-optimizers)
   - [TextGrad — Textual Gradient Descent](#3-textgrad--textual-gradient-feedback)
   - [AdalFlow — Declarative Auto-Optimization](#4-adalflow--declarative-auto-optimization)
5. [Repository Structure](#-repository-structure)
6. [Quickstart](#-quickstart)
7. [Running Tests](#-running-tests)
8. [Project Website](#-project-website)
9. [Token & Cost Analysis](#-token--cost-analysis)
10. [References](#-references)

---

## 🎯 Project Overview

In commercial customer-support systems, intent classification prompts are traditionally hand-crafted through heuristic trial-and-error. This manual approach has three critical weaknesses:

| Weakness | Description |
|---|---|
| **Semantic Boundary Ambiguity** | Subtle distinctions (e.g., statement mystery fees vs. merchant overcharge) cannot be reliably resolved across diverse user phrasings |
| **Context Window Bloat** | Manual few-shot prompting adds examples without optimizing the core instruction |
| **No Convergence Guarantee** | Manual edits lack gradient or metric feedback — fixing one edge case often breaks another |

This project demonstrates that **Programmatic Prompt Optimization** resolves all three issues simultaneously:

- 📈 **+16.25% absolute accuracy gain** (81.25% → 97.50%)
- 🗜️ **53% reduction in prompt token length** (659 → 309 tokens)
- ✅ **No human intervention** required after the initial setup

### The Research Question

> *"Can a computer program automatically discover better LLM prompts than a human can write by hand — and by how much?"*

**Answer: Yes. Significantly.**

---

## 🏆 Key Results

### Accuracy Comparison

```
Manual Zero-Shot   ████████████████████████████████▌          81.25%
Manual Few-Shot    ████████████████████████████████████       90.00%
DSPy Bootstrap     ████████████████████████████████████       90.00%
DSPy COPRO    ★   ████████████████████████████████████████▊  97.50%
TextGrad TGD  ★   ████████████████████████████████████████▊  97.50%
AdalFlow      ★   ████████████████████████████████████████▊  97.50%
```

### Optimization Cost vs. Performance

| Method | Opt LLM Calls | Opt Tokens Used | Final Prompt Tokens | Accuracy |
|:---|:---:|:---:|:---:|:---:|
| Manual Zero-Shot | 0 | 0 | 420 | 81.25% |
| Manual Few-Shot | 0 | 0 | 659 | 90.00% |
| DSPy BootstrapFewShot | 10 | 1,039 | 404 | 90.00% |
| **DSPy COPRO / MIPRO** | **11** ✅ | **1,146** ✅ | 404 | **97.50%** |
| TextGrad TGD | 202 | 59,107 | **309** ✅ | **97.50%** |
| **AdalFlow Auto-Optimizer** | **25** | **2,291** | 315 | **97.50%** |

> **Key Insight:** DSPy COPRO achieves peak accuracy with the **lowest optimization cost** (11 calls, 1,146 tokens). TextGrad achieves peak accuracy with the **shortest runtime prompt** (309 tokens — lowest production inference cost).

---

## 📊 Dataset — Banking77 (10-Intent Subset)

A curated high-difficulty subset of the [Banking77](https://arxiv.org/abs/2003.04807) benchmark, specifically designed to expose semantic boundary ambiguities that challenge manually written prompts.

### The 10 Intents

| Intent | Description | Ambiguity Challenge |
|:---|:---|:---|
| `card_arrival` | Card delivery tracking, shipment status | Confused with `lost_or_stolen_card` if "lost in transit" mentioned |
| `card_linking` | Connecting external cards to the app | Confused with `visa_or_mastercard` if card brand mentioned |
| `exchange_rate` | FX rates, currency conversion fees | Distinct from standard transfer fees |
| `extra_charge_on_statement` | Mystery or unrecognized fee on statement | ⚠️ **Most confused intent** — overlaps with `wrong_amount_additional_charge` |
| `lost_or_stolen_card` | Emergency card freeze, lost wallet | High-priority; must not be confused with delivery tracking |
| `top_up_reverted` | Deposited balance cancelled or refunded | Involves bank-initiated reversal vs. customer dispute |
| `transfer_fee_charged` | Fee applied to wire/bank transfer | Confused with statement mystery charges in zero-shot |
| `verify_identity` | KYC document submission, selfie check | ⚠️ **Lexical trap** — "verify" in unrelated queries triggers this |
| `visa_or_mastercard` | Card network brand inquiries | Must not trigger when brand mentioned in linking context |
| `wrong_amount_additional_charge` | Merchant billed incorrect total | Confused with `extra_charge_on_statement` in zero-shot |

### Dataset Split (Stratified)

```
Total: 240 samples | 10 intents | 24 samples per intent

┌─────────────┬─────────┬────────────────────────────────────────┐
│    Split    │ Samples │               Purpose                  │
├─────────────┼─────────┼────────────────────────────────────────┤
│  Training   │   120   │ Demo bootstrapping, gradient computing │
│  Validation │    40   │ Prompt candidate evaluation / tuning   │
│  Test       │    80   │ Final evaluation ONLY (never seen)     │
└─────────────┴─────────┴────────────────────────────────────────┘
```

---

## 🔬 Methodology

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│               Customer Support Utterance (input)                 │
└────────────────────────┬────────────────────────────────────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
 ┌─────────────┐  ┌─────────────┐  ┌─────────────┐
 │   Manual    │  │    DSPy     │  │  TextGrad   │
 │  Baselines  │  │  Compiler   │  │   Engine    │
 │  Zero-Shot  │  │  Bootstrap  │  │  Variable   │
 │  Few-Shot   │  │  COPRO/MIPRO│  │  Critic+TGD │
 └─────────────┘  └─────────────┘  └─────────────┘
        │                │                │
        └────────────────┼────────────────┘
                         │
                         ▼
              ┌────────────────────┐
              │  AdalFlow Pipeline │
              │  Component-based   │
              │  Auto-Optimizer    │
              └────────────────────┘
                         │
                         ▼
         ┌───────────────────────────────┐
         │   TokenTracker + Evaluator    │
         │  Accuracy · F1 · Token Cost   │
         └───────────────────────────────┘
```

---

### 1. Manual Baselines

#### Zero-Shot (`src/baselines/zero_shot.py`)
Manually authored system prompt containing role definition, 10 intent descriptions, and formatting constraints. No examples provided.

```python
# Manual Zero-Shot prompt structure
system_prompt = """
You are a banking customer support intent classifier.
Classify the query into exactly one of:
- card_arrival: Physical card delivery status...
- lost_or_stolen_card: Emergency card freeze...
[8 more intents...]
Return ONLY the intent label.
"""
```

**Accuracy: 81.25%** — fails on lexical traps and boundary ambiguities.

#### Few-Shot (`src/baselines/few_shot.py`)
Extends Zero-Shot with 1 manually selected demonstration per intent (10 examples total).

**Accuracy: 90.00%** — improves recall but uses 659 tokens per inference call.

---

### 2. DSPy — Signatures, Modules & Two Optimizers

> **DSPy** (Khattab et al., Stanford, 2023) decouples prompt *specification* from prompt *text* using declarative Signatures and Modules, then compiles them into optimized prompts via Teleprompters.

#### Signature & Module (`src/optimizers/dspy_engine.py`)

```python
class BankingIntentSignature(Signature):
    """Classify the customer banking query into the single best intent."""
    instructions = "Classify into one of: card_arrival, card_linking, ..."
    fields = {
        "customer_query": Field(desc="Customer message", is_input=True),
        "intent": Field(desc="Predicted intent label", is_input=False),
    }

# Module: ChainOfThought reasons before predicting
module = ChainOfThought(signature_cls=BankingIntentSignature, llm=llm)
```

#### Optimizer 1 — BootstrapFewShot

Automatically synthesizes few-shot demonstrations by running the student module on training data, keeping only examples that pass the exact-match metric, and pruning to a target budget.

```python
optimizer = BootstrapFewShot(max_bootstrapped_demos=5, max_labeled_demos=5)
compiled_module = optimizer.compile(module, trainset=train_data, valset=val_data)
# Result: 90.00% accuracy | 10 LLM calls | 1,039 tokens
```

#### Optimizer 2 — COPRO / MIPRO

Generates candidate instruction mutations via LLM meta-prompting, evaluates each candidate across the full validation set, and selects the Pareto-optimal instruction configuration.

```python
optimizer = COPROOptimizer(num_candidates=4, val_eval_limit=40)
best_module = optimizer.compile(module, trainset=train_data, valset=val_data)
# Result: 97.50% accuracy | 11 LLM calls | 1,146 tokens ✅
```

**Auto-generated winning instruction:**
> *"Classify customer inquiries into the most appropriate banking intent. Carefully disambiguate between statement-level unexpected fees ('extra_charge_on_statement') and merchant overbilling ('wrong_amount_additional_charge'). Return only the exact intent identifier."*

---

### 3. TextGrad — Textual Gradient Feedback

> **TextGrad** (Yuksekgonul et al., Stanford, 2024) treats the prompt as a **Variable** parameter in a computational graph. A Critic LLM generates natural-language gradients; Textual Gradient Descent (TGD) applies them to update the prompt text.

```
Traditional Deep Learning:  W ← W − η · ∇_W L
TextGrad Analogue:   Prompt ← Prompt ⊖ η · textual_gradient
```

#### How It Works (`src/optimizers/textgrad_engine.py`)

```python
# 1. Prompt is a Variable node in the computation graph
prompt_var = Variable(value=initial_prompt, role_description="System Prompt")

# 2. Loss Critic generates textual gradients on misclassifications
critic = TextualLossCritic(llm=llm)
gradient = critic.compute_gradient(
    variable=prompt_var,
    customer_query="Can someone verify if my replacement card has been mailed?",
    predicted_intent="verify_identity",   # WRONG
    target_intent="card_arrival",          # CORRECT
)
# gradient = "The word 'verify' misleads classification. Add rule: if query
#             concerns card delivery status, classify as card_arrival regardless
#             of lexical triggers like 'verify' or 'confirm'."

# 3. Accumulate gradients, then apply TGD step
prompt_var.add_gradient(gradient)
optimizer = TextualGradientDescent(parameters=[prompt_var], llm=llm)
optimizer.step()  # Rewrites prompt to include disambiguation rule
```

**Result: 97.50% accuracy | 309 tokens (shortest prompt) | 53% cheaper inference**

---

### 4. AdalFlow — Declarative Auto-Optimization

> **AdalFlow** provides a PyTorch-inspired modular design for LLM pipelines. Components are independent, composable building blocks. The Auto-Optimizer diagnoses training failures and jointly tunes instructions and few-shot parameters.

#### Pipeline Architecture (`src/optimizers/adalflow_engine.py`)

```python
# Declarative building blocks
prompt_template = PromptTemplate(
    template_str='Customer Inquiry: "{query}"\nPredicted Intent:',
    instructions="...",   # ← Trainable parameter
    demos=[],             # ← Trainable parameter
)
generator = Generator(llm=llm, prompt_template=prompt_template)
parser = OutputParser(valid_intents=dataset.get_intents())

# Full pipeline
pipeline = AdalFlowIntentPipeline(llm=llm)

# Auto-Optimization: Diagnose → Refine → Select Demos
optimizer = AdalFlowAutoOptimizer(pipeline=pipeline, llm=llm)
optimizer.optimize(trainset=train_data, valset=val_data, n_demos=5)
# Result: 97.50% accuracy | 25 LLM calls | 2,291 tokens ✅
```

---

## 📁 Repository Structure

```
PROMT_ENGG/
│
├── 📄 README.md                        ← You are here
├── 📄 PROJECT_REPORT.md                ← Full academic report + Viva Q&A
├── ⚙️  config.py                        ← Global hyperparameters
├── 📋 requirements.txt                 ← Dependencies
├── 🔧 pytest.ini                       ← Test configuration
│
├── 🚀 run_experiments.py               ← Main benchmark runner (START HERE)
├── 🎮 run_demo.py                      ← Interactive CLI live demo
│
├── 🌐 website/
│   ├── index.html                      ← Professional project website
│   ├── assets/css/style.css            ← Dark theme + animations
│   ├── assets/js/particles.js          ← Particle canvas background
│   └── assets/js/main.js               ← Interactions + live demo engine
│
├── 📊 data/
│   ├── prepare_dataset.py              ← Dataset generation script
│   └── banking77_subsets/
│       ├── banking77_10intents.json    ← 240 curated samples (stratified)
│       ├── intents_metadata.json       ← Intent descriptions & keywords
│       └── dataset_loader.py           ← Split loader + demo sampler
│
├── 🔧 src/
│   ├── llm_backend/
│   │   ├── base.py                     ← Abstract BaseLLM interface
│   │   ├── tracker.py                  ← Token + call + latency tracker
│   │   ├── mock_llm.py                 ← Offline semantic simulator
│   │   └── api_llm.py                  ← OpenAI / Gemini / Groq adapter
│   │
│   ├── metrics/
│   │   └── classification_metrics.py   ← Accuracy, Macro F1, Confusion Matrix
│   │
│   ├── baselines/
│   │   ├── zero_shot.py                ← Manual Zero-Shot classifier
│   │   └── few_shot.py                 ← Manual Few-Shot classifier
│   │
│   ├── optimizers/
│   │   ├── dspy_engine.py              ← Signatures + Modules + Bootstrap + COPRO
│   │   ├── textgrad_engine.py          ← Variable + LossCritic + TGD + history
│   │   └── adalflow_engine.py          ← Components + Pipeline + AutoOptimizer
│   │
│   ├── evaluation/
│   │   ├── evaluator.py                ← BenchmarkRunner (all 6 classifiers)
│   │   └── report_generator.py         ← CSV + JSON + Markdown exporter
│   │
│   └── visualization/
│       └── charts.py                   ← Standalone HTML dashboard
│
├── 🧪 tests/
│   ├── test_dataset.py                 ← Dataset integrity (3 tests)
│   ├── test_metrics.py                 ← Classification metrics (3 tests)
│   ├── test_baselines.py               ← Baseline classifiers (2 tests)
│   ├── test_dspy.py                    ← DSPy optimizers (2 tests)
│   ├── test_textgrad.py                ← TextGrad engine (2 tests)
│   └── test_adalflow.py                ← AdalFlow pipeline (2 tests)
│
└── 📈 results/                         ← Auto-generated on run
    ├── comparison_table.csv
    ├── comparison_metrics.json
    ├── comparative_report.md
    └── benchmark_dashboard.html
```

---

## ⚡ Quickstart

### Prerequisites

```bash
Python 3.10 or higher
```

### 1. Clone the Repository

```bash
git clone https://github.com/<your-username>/programmatic-prompt-optimization.git
cd programmatic-prompt-optimization
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Complete Benchmark

```bash
python run_experiments.py
```

This runs all 6 classifiers and generates:
- Terminal summary table
- `results/comparison_table.csv`
- `results/comparison_metrics.json`
- `results/comparative_report.md`
- `results/benchmark_dashboard.html`

> **No API key required!** The framework includes a built-in high-fidelity offline semantic simulator. To use a real LLM, set `OPENAI_API_KEY` or `GEMINI_API_KEY` as environment variables and pass `--model-type openai`.

### 4. Try the Interactive CLI Demo

```bash
python run_demo.py
```

Type any banking customer query and see predictions from all 6 classifiers in real time.

### 5. Use a Real LLM API (Optional)

```bash
# Copy and configure the environment file
cp .env.example .env
# Add your API key to .env

# Run with OpenAI
python run_experiments.py --model-type openai

# Run with Gemini
python run_experiments.py --model-type gemini
```

### 6. Open the Project Website

```bash
# Windows
start website/index.html

# macOS
open website/index.html

# Linux
xdg-open website/index.html
```

---

## 🧪 Running Tests

All **14 unit tests** are self-contained, offline, and complete in under 4 seconds:

```bash
pytest tests -v
```

```
tests/test_dataset.py::test_dataset_loading              PASSED ✅
tests/test_dataset.py::test_dataset_splits_balance       PASSED ✅
tests/test_dataset.py::test_few_shot_demonstrations      PASSED ✅
tests/test_metrics.py::test_perfect_classification       PASSED ✅
tests/test_metrics.py::test_imperfect_classification     PASSED ✅
tests/test_metrics.py::test_confusion_matrix_rendering   PASSED ✅
tests/test_baselines.py::test_zero_shot_classifier       PASSED ✅
tests/test_baselines.py::test_few_shot_classifier        PASSED ✅
tests/test_dspy.py::test_dspy_bootstrap_optimizer        PASSED ✅
tests/test_dspy.py::test_dspy_copro_optimizer            PASSED ✅
tests/test_textgrad.py::test_textgrad_variable_and_grads PASSED ✅
tests/test_textgrad.py::test_textgrad_classifier         PASSED ✅
tests/test_adalflow.py::test_adalflow_prompt_template    PASSED ✅
tests/test_adalflow.py::test_adalflow_classifier         PASSED ✅

14 passed in 3.08s
```

---

## 🌐 Project Website

A fully standalone professional dark-themed website showcasing the entire project, including:

- Animated hero section with live statistics
- Interactive live intent classifier demo (all 6 models)
- Animated accuracy bar charts and Macro F1 progress rings
- Complete methodology flowcharts
- Key findings and Viva Q&A accordion

**Open:** `website/index.html` in any modern browser.

---

## 💰 Token & Cost Analysis

### Production Inference Cost (per 1 Million Queries)

Estimated at GPT-4o Mini pricing ($0.00015 per 1,000 input tokens):

| Method | Tokens/Query | Cost per 1M Queries | Annual Saving vs Few-Shot |
|:---|:---:|:---:|:---:|
| Manual Few-Shot | 659 | **$98.85** | Baseline |
| DSPy COPRO (optimized) | 404 | $60.60 | $38.25/day |
| TextGrad TGD (optimized) | **309** | **$46.35** | **$52.50/day** |
| AdalFlow (optimized) | 315 | $47.25 | $51.60/day |

> **TextGrad saves ~$19,163 per year per million daily queries** compared to Manual Few-Shot, while achieving 7.5 percentage points higher accuracy.

### One-Time Optimization Cost

| Method | Optimization Tokens | Estimated One-Time Cost |
|:---|:---:|:---:|
| DSPy BootstrapFewShot | 1,039 | < $0.01 |
| DSPy COPRO / MIPRO | 1,146 | < $0.01 |
| TextGrad TGD | 59,107 | ~$0.09 |
| AdalFlow Auto-Optimizer | 2,291 | < $0.01 |

---

## 🔑 Key Concepts

| Term | Definition |
|:---|:---|
| **Intent Classification** | Predicting which of N predefined categories a text belongs to |
| **Zero-Shot** | Classify using instructions only — no examples provided |
| **Few-Shot** | Classify using instructions + a small number of labeled examples |
| **DSPy Signature** | Declarative task specification separating *what* from *how* |
| **Teleprompter** | DSPy's name for prompt optimization algorithms |
| **Variable** | In TextGrad, a text parameter node in a computation graph |
| **Textual Gradient** | A natural language critique indicating how to improve a prompt |
| **TGD** | Textual Gradient Descent — optimizes prompt text using critiques |
| **Component** | AdalFlow's composable building block (analogous to `torch.nn.Module`) |
| **Macro F1** | Average F1 computed independently per class — robust to class imbalance |
| **Boundary Ambiguity** | Semantic overlap between two intents causing misclassification |
| **Bootstrapping** | Auto-generating verified examples from training data |

---

## 📚 References

1. **Khattab, O., Singhvi, A., Maheshwari, P., et al.** (2023). *DSPy: Compiling Declarative Language Model Calls into State-of-the-Art Pipelines*. Stanford University. [arXiv:2310.03714](https://arxiv.org/abs/2310.03714)

2. **Yuksekgonul, M., Bianchi, F., Boen, J., et al.** (2024). *TextGrad: Automatic "Differentiation" via Text*. Stanford University. [arXiv:2406.07496](https://arxiv.org/abs/2406.07496)

3. **Casanueva, I., Temčinas, T., Gerz, D., et al.** (2020). *Efficient Intent Detection with Dual Sentence Encoders*. 2nd Workshop on NLP for Conversational AI, ACL. [arXiv:2003.04807](https://arxiv.org/abs/2003.04807)

4. **Brown, T., Mann, B., Ryder, N., et al.** (2020). *Language Models are Few-Shot Learners*. NeurIPS 2020. [arXiv:2005.14165](https://arxiv.org/abs/2005.14165)

5. **AdalFlow Documentation** (2024). *The Library to Build & Auto-Optimize LLM Applications*. [GitHub](https://github.com/SylphAI-Inc/AdalFlow)

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgements

- Dataset: [Banking77](https://arxiv.org/abs/2003.04807) by Casanueva et al.
- DSPy Framework: [Stanford NLP Group](https://github.com/stanfordnlp/dspy)
- TextGrad Framework: [Zou Group, Stanford](https://github.com/zou-group/textgrad)
- AdalFlow Framework: [SylphAI Inc.](https://github.com/SylphAI-Inc/AdalFlow)

---

<div align="center">

**Mini Project #6 · Prompt Engineering · NLP · 2026**

*Built with Python 3.11 · DSPy · TextGrad · AdalFlow · Banking77*

```
14 Tests Passing ✅ | 97.50% Peak Accuracy ⭐ | +16.25% vs Baseline 📈
```

</div>
