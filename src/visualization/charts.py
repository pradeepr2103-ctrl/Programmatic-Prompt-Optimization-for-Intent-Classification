"""
Visualization and Interactive HTML Dashboard Generator.
Renders standalone interactive SVG visual charts and HTML report without external dependencies.
"""

import os
from typing import Dict


def generate_html_dashboard(results: Dict[str, dict], output_path: str = "results/benchmark_dashboard.html") -> str:
    """Creates a modern HTML/CSS benchmark dashboard with responsive SVG charts."""
    methods = list(results.keys())
    accuracies = [results[m]["accuracy"] * 100 for m in methods]
    f1_scores = [results[m]["macro_f1"] * 100 for m in methods]
    tokens = [results[m]["opt_total_tokens"] for m in methods]
    prompt_lens = [results[m]["prompt_length_tokens"] for m in methods]

    # Bar chart 1: Accuracy comparison
    bar_rows_acc = []
    for m, acc in zip(methods, accuracies):
        width_pct = min(100, max(5, acc))
        color = "#2563eb" if "Manual" in m else "#10b981"
        bar_rows_acc.append(f"""
        <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 4px;">
                <span>{m}</span>
                <span>{acc:.2f}%</span>
            </div>
            <div style="background: #e2e8f0; border-radius: 6px; height: 24px; overflow: hidden;">
                <div style="background: {color}; width: {width_pct}%; height: 100%; border-radius: 6px; transition: width 0.5s ease;"></div>
            </div>
        </div>
        """)

    # Bar chart 2: Token consumption
    max_tok = max(tokens) if max(tokens) > 0 else 1
    bar_rows_tok = []
    for m, tok in zip(methods, tokens):
        width_pct = (tok / max_tok) * 100 if max_tok > 0 else 2
        bar_rows_tok.append(f"""
        <div style="margin-bottom: 14px;">
            <div style="display: flex; justify-content: space-between; font-size: 13px; font-weight: 600; margin-bottom: 4px;">
                <span>{m}</span>
                <span>{tok:,} tokens</span>
            </div>
            <div style="background: #e2e8f0; border-radius: 6px; height: 24px; overflow: hidden;">
                <div style="background: #8b5cf6; width: {width_pct}%; height: 100%; border-radius: 6px;"></div>
            </div>
        </div>
        """)

    # Table rows
    table_rows = []
    for m in methods:
        d = results[m]
        table_rows.append(f"""
        <tr style="border-bottom: 1px solid #e2e8f0;">
            <td style="padding: 10px; font-weight: 600;">{d['method']}</td>
            <td style="padding: 10px; text-align: center;"><span style="background: #e0f2fe; color: #0369a1; padding: 2px 8px; border-radius: 4px; font-size: 12px;">{d['category']}</span></td>
            <td style="padding: 10px; text-align: right; font-weight: bold; color: #0f172a;">{d['accuracy']*100:.2f}%</td>
            <td style="padding: 10px; text-align: right;">{d['macro_f1']:.4f}</td>
            <td style="padding: 10px; text-align: right;">{d['prompt_length_tokens']}</td>
            <td style="padding: 10px; text-align: right;">{d['opt_llm_calls']}</td>
            <td style="padding: 10px; text-align: right; font-weight: 500;">{d['opt_total_tokens']:,}</td>
        </tr>
        """)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Programmatic Prompt Optimization Dashboard</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
            background-color: #f8fafc;
            color: #1e293b;
            margin: 0;
            padding: 30px;
        }}
        .container {{
            max-width: 1100px;
            margin: 0 auto;
        }}
        .header {{
            background: linear-gradient(135deg, #1e3a8a 0%, #3b82f6 100%);
            color: white;
            padding: 28px 32px;
            border-radius: 12px;
            margin-bottom: 25px;
            box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.1);
        }}
        .grid {{
            display: grid;
            grid-template-columns: 1fr 1fr;
            gap: 20px;
            margin-bottom: 25px;
        }}
        .card {{
            background: white;
            padding: 24px;
            border-radius: 12px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
            border: 1px solid #e2e8f0;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            font-size: 14px;
        }}
        th {{
            background: #f1f5f9;
            padding: 10px;
            text-align: left;
            font-weight: 600;
            color: #475569;
        }}
    </style>
</head>
<body>
    <div class="container">
        <div class="header">
            <h1 style="margin: 0 0 8px 0; font-size: 24px;">Programmatic Prompt Optimization for Intent Classification</h1>
            <p style="margin: 0; opacity: 0.9; font-size: 14px;">Comparative Benchmark: Manual Baselines vs DSPy, TextGrad, and AdalFlow on Banking77 (10-Intent)</p>
        </div>

        <div class="grid">
            <div class="card">
                <h3 style="margin-top: 0; font-size: 16px; color: #334155;">Test Accuracy Comparison (%)</h3>
                {"".join(bar_rows_acc)}
            </div>

            <div class="card">
                <h3 style="margin-top: 0; font-size: 16px; color: #334155;">Optimization Token Consumption</h3>
                {"".join(bar_rows_tok)}
            </div>
        </div>

        <div class="card">
            <h3 style="margin-top: 0; font-size: 16px; color: #334155; margin-bottom: 16px;">Comprehensive Metrics Table</h3>
            <table>
                <thead>
                    <tr>
                        <th>Method</th>
                        <th style="text-align: center;">Type</th>
                        <th style="text-align: right;">Accuracy</th>
                        <th style="text-align: right;">Macro F1</th>
                        <th style="text-align: right;">Prompt Tokens</th>
                        <th style="text-align: right;">Opt Calls</th>
                        <th style="text-align: right;">Opt Tokens</th>
                    </tr>
                </thead>
                <tbody>
                    {"".join(table_rows)}
                </tbody>
            </table>
        </div>
    </div>
</body>
</html>
"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html)
    return output_path
