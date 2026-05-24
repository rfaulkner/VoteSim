#!/usr/bin/env python3
"""Generate tables.tex with both standard and p635 models.

Computes:
  1. Mean Likert score ± SE (pooled across all voters and 12 issues)
  2. Mean ranking ± SE (pooled across all voters and 12 issues)

Outputs LaTeX code for a combined 9-model table.
"""

import json
import math
import os

import numpy as np

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "results", "golden",
)

# Columns in display order.
MODELS = [
    ("Llama-3.3", "divisive-12.issue.llama-3.3-70b-instruct.json"),
    ("Mistral-Med", "divisive-12.issue.mistral-medium-3.json"),
    ("Gemma-4", "divisive-12.issue.gemma-4-31b-it.json"),
    ("Hermes-3", "divisive-12.issue.hermes-3-llama-3.1-70b.json"),
    ("Llama-3.1", "divisive-12.issue.llama-3.1-8b-instruct.json"),
    ("Grok-4.1", "divisive-12.issue.grok-4.1-fast.json"),
    ("Mistral-Sm", "divisive-12.issue.mistral-small-3.1-24b-instruct.json"),
    ("Llama-3.3 (635)", "divisive-12.issue.p635.llama-3.3-70b-instruct.json"),
    ("Mistral-Med (635)", "divisive-12.issue.p635.mistral-medium-3.json"),
]

# Systems in display order.
SYSTEM_ORDER = [
    "fptp", "sntv", "alternative_vote", "trs",
    "dhondt", "stv", "sainte_lague",
    "baseline", "baseline_informed",
]
SYSTEM_DISPLAY = {
    "fptp": "FPTP",
    "sntv": "PBV",
    "alternative_vote": "IRV",
    "trs": "Two-Round",
    "dhondt": "D'Hondt",
    "stv": "STV",
    "sainte_lague": "Sainte-Lagu\\\"{e}",
    "baseline": "Base Model",
    "baseline_informed": "Base Med.",
}
SYSTEM_CATEGORY = {
    "fptp": "maj", "sntv": "maj", "alternative_vote": "maj", "trs": "maj",
    "dhondt": "prop", "stv": "prop", "sainte_lague": "prop",
    "baseline": "base", "baseline_informed": "base",
}


def load_all_data():
    all_data = {}
    for display_name, fname in MODELS:
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"WARNING: {path} does not exist!")
            continue
        with open(path) as f:
            all_data[display_name] = json.load(f)
    return all_data


def compute_likert_table(all_data):
    """Compute mean Likert score ± SE pooled across all voters and issues."""
    table = {}
    for display_name, _ in MODELS:
        if display_name not in all_data:
            continue
        model_data = all_data[display_name]
        table[display_name] = {}

        for sys_name in SYSTEM_ORDER:
            all_scores = []
            for issue_key, issue in model_data.items():
                scores = issue.get("scores", {})
                for uid, sys_scores in scores.items():
                    if sys_name in sys_scores:
                        all_scores.append(float(sys_scores[sys_name]))
            if all_scores:
                avg = np.mean(all_scores)
                se = np.std(all_scores, ddof=1) / math.sqrt(len(all_scores)) if len(all_scores) > 1 else 0.0
                table[display_name][sys_name] = (avg, se)
            else:
                table[display_name][sys_name] = (0.0, 0.0)
    return table


def compute_ranking_table(all_data):
    """Compute mean rank ± SE pooled across all voters and issues."""
    table = {}
    for display_name, _ in MODELS:
        if display_name not in all_data:
            continue
        model_data = all_data[display_name]
        table[display_name] = {}

        for sys_name in SYSTEM_ORDER:
            all_ranks = []
            for issue_key, issue in model_data.items():
                rankings = issue.get("rankings", {})
                for uid, r_list in rankings.items():
                    if sys_name in r_list:
                        all_ranks.append(r_list.index(sys_name) + 1)
            if all_ranks:
                avg = np.mean(all_ranks)
                se = np.std(all_ranks, ddof=1) / math.sqrt(len(all_ranks)) if len(all_ranks) > 1 else 0.0
                table[display_name][sys_name] = (avg, se)
            else:
                table[display_name][sys_name] = (0.0, 0.0)
    return table


def _find_best_per_model(table, higher_is_better=True):
    """Find best system overall (including baselines) per model."""
    best = {}
    active_models = [m[0] for m in MODELS if m[0] in table]
    for model_name in active_models:
        best_val = None
        best_sys = None
        for sys_name in SYSTEM_ORDER:
            val = table[model_name][sys_name][0]
            if best_val is None:
                best_val = val
                best_sys = sys_name
            elif higher_is_better and val > best_val:
                best_val = val
                best_sys = sys_name
            elif not higher_is_better and val < best_val:
                best_val = val
                best_sys = sys_name
        best[model_name] = best_sys
    return best


def generate_table_latex(table, title, label, is_likert=True):

    active_models = [m[0] for m in MODELS if m[0] in table]
    n_models = len(active_models)
    n_cols = n_models + 1

    best = _find_best_per_model(table, higher_is_better=is_likert)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{" + title + "}")
    lines.append(r"\label{" + label + "}")

    # Stretch table to full text width using tabular* with extracolsep
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * n_models + r"@{}}")
    lines.append(r"\toprule")

    # Header Row 1: Model Names
    row1 = [r"\textbf{System}"]
    for m in active_models:
        short_name = m.split(" (")[0]
        row1.append(r"\textbf{" + short_name + "}")
    lines.append(" & ".join(row1) + r" \\")

    # Header Row 2: Cohort Size (N=120 / N=635)
    row2 = [""]
    for m in active_models:
        size = "635" if "635" in m else "120"
        row2.append(r"\textit{N=" + size + "}")
    lines.append(" & ".join(row2) + r" \\")
    lines.append(r"\midrule")

    # Category rows
    prev_cat = None
    for sys_name in SYSTEM_ORDER:
        cat = SYSTEM_CATEGORY[sys_name]
        if cat != prev_cat:
            if prev_cat is not None:
                lines.append(r"\midrule")
            cat_label = {"maj": "Majoritarian", "prop": "Proportional", "base": "Baselines"}[cat]
            cat_color = {"maj": "red", "prop": "green", "base": "blue"}[cat]
            lines.append(
                r"\multicolumn{" + str(n_cols) + r"}{l}{\cellcolor{"
                + cat_color + r"!10}\textit{" + cat_label + r"}} \\"
            )
            prev_cat = cat

        display = SYSTEM_DISPLAY[sys_name]
        cells = [r"\quad " + display]
        for m in active_models:
            mean_val, se_val = table[m][sys_name]
            val_str = f"{mean_val:.2f}"
            se_str = f"{se_val:.2f}"
            is_best = (best[m] == sys_name)
            if is_best:
                cell = r"\textbf{" + val_str + r"}" + r" \tiny{$\pm$}" + se_str
            else:
                cell = val_str + r" \tiny{$\pm$}" + se_str
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")


    return "\n".join(lines)


if __name__ == "__main__":
    all_data = load_all_data()

    likert_table = compute_likert_table(all_data)
    ranking_table = compute_ranking_table(all_data)

    likert_latex = generate_table_latex(
        likert_table,
        "Mean Likert scores ($1$--$5$, higher is better) with standard errors across the \\textsc{Divisive-12} issue set. Bold indicates the best score per model.",
        "tab:likert_new",
        is_likert=True
    )
    ranking_latex = generate_table_latex(
        ranking_table,
        "Mean rankings ($1$ = best) with standard errors across the \\textsc{Divisive-12} issue set. Bold indicates the best (lowest) rank per model.",
        "tab:rankings_new",
        is_likert=False
    )

    out_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "..", "draft", "tables.tex"
    )
    with open(out_path, "w") as f:
        f.write(likert_latex + "\n\n" + ranking_latex + "\n")

    print(f"Successfully generated {out_path}")

