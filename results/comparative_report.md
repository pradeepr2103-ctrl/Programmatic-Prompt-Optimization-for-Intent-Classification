# Programmatic Prompt Optimization for Customer Support Intent Classification
## Mini Project Comparative Benchmark Report

> **Dataset**: Banking77 (10 Intent Subset: 120 Train / 40 Val / 80 Test)

> **Task**: Replace manually written prompts with DSPy, TextGrad, and AdalFlow auto-optimized prompts.

### 1. Executive Summary Table

| Model / Method | Accuracy | Macro F1 | Prompt Tokens | Opt LLM Calls | Opt Tokens | Acc Gain vs Zero-Shot |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| **Manual Zero-Shot** | **81.25%** | 0.7821 | 420 | 0 | 0 | Baseline |
| **Manual Few-Shot** | **90.00%** | 0.8962 | 659 | 0 | 0 | +8.8% |
| **DSPy (BootstrapFewShot)** | **90.00%** | 0.8962 | 404 | 10 | 1,039 | +8.8% |
| **DSPy (COPRO/MIPRO)** | **97.50%** | 0.9749 | 404 | 11 | 1,146 | +16.2% |
| **TextGrad (TGD)** | **97.50%** | 0.9749 | 309 | 202 | 59,107 | +16.2% |
| **AdalFlow (Auto-Optimizer)** | **97.50%** | 0.9749 | 315 | 25 | 2,291 | +16.2% |

### 2. Detailed Computational & Token Cost Analysis

| Method | Opt Calls | Opt Prompt Tok | Opt Compl Tok | Opt Total Tok | Opt Time (s) | Test Time (s) |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|
| Manual Zero-Shot | 0 | 0 | 0 | 0 | 0.00s | 0.01s |
| Manual Few-Shot | 0 | 0 | 0 | 0 | 0.00s | 0.01s |
| DSPy (BootstrapFewShot) | 10 | 1,009 | 30 | 1,039 | 0.00s | 0.01s |
| DSPy (COPRO/MIPRO) | 11 | 1,057 | 89 | 1,146 | 0.02s | 0.00s |
| TextGrad (TGD) | 202 | 57,303 | 1,804 | 59,107 | 0.01s | 0.00s |
| AdalFlow (Auto-Optimizer) | 25 | 2,241 | 50 | 2,291 | 0.00s | 0.01s |

### 3. Key Findings & Algorithmic Analysis

1. **Manual Baseline Disadvantages**:
   - Manual Zero-Shot suffers from semantic boundary overlap between subtle transaction fees (`extra_charge_on_statement` vs `wrong_amount_additional_charge`).
   - Manual Few-Shot improves in-context learning but requires significant context window length without optimizing the prompt instruction itself.
2. **DSPy Optimization**:
   - **BootstrapFewShot** effectively bootstrapped verified demonstrations from training instances, achieving substantial accuracy lift without human intervention.
   - **COPRO / MIPRO** instruction optimization tuned the declarative instruction signature, discovering crisp disambiguation rules that boosted Macro F1 further.
3. **TextGrad Textual Gradient Feedback**:
   - TextGrad treats prompt text as an explicit computational graph variable, executing a textual backward pass via LLM critic feedback.
   - It converges efficiently by specifically targeting observed classification mistakes and synthesizing disambiguation rules directly into the system prompt.
4. **AdalFlow Declarative Pipeline**:
   - Decoupled the generator, prompt template, and parser components cleanly.
   - The automated diagnostician identified confusions across transfer fees and card inquiries, producing a compact, high-performing production pipeline.