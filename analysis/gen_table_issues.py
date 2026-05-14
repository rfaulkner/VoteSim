#!/usr/bin/env python3
"""Generate table_issues.tex from golden_votesim results.

For each (voting_system, issue) cell, computes:
  - Mean Likert score across all voters for each model
  - Takes the MEAN across the 7 model means as the cell value
  - Reports std-err across the 7 model means

Outputs LaTeX to draft/table_issues.tex.
"""

import json
import math
import os
from collections import defaultdict

import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "results", "golden",
)
DRAFT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "draft",
)

MODEL_FILES = {
    "Llama-3.3-70B": "divisive-12.issue.llama-3.3-70b-instruct.json",
    "Llama-3.1-8B": "divisive-12.issue.llama-3.1-8b-instruct.json",
    "Mistral-Med-3": "divisive-12.issue.mistral-medium-3.json",
    "Mistral-Sm-24B": "divisive-12.issue.mistral-small-3.1-24b-instruct.json",
    "Gemma-4-31B": "divisive-12.issue.gemma-4-31b-it.json",
    "Hermes-3-70B": "divisive-12.issue.hermes-3-llama-3.1-70b.json",
    "Grok-4.1-Fast": "divisive-12.issue.grok-4.1-fast.json",
}

# Voting systems in display order.
SYSTEM_ORDER = [
    "sntv", "fptp", "alternative_vote", "trs",
    "dhondt", "stv", "sainte_lague",
    "baseline", "baseline_informed",
]
SYSTEM_DISPLAY = {
    "fptp": "FPTP",
    "sntv": "PBV",
    "alternative_vote": "Alt.\\,Vote",
    "trs": "Two-Round",
    "dhondt": "D'Hondt",
    "stv": "STV",
    "sainte_lague": "Sainte-L.",
    "baseline": "Baseline",
    "baseline_informed": "Oracle Med.",
}
SYSTEM_CATEGORY = {
    "fptp": "maj", "sntv": "maj", "alternative_vote": "maj", "trs": "maj",
    "dhondt": "prop", "stv": "prop", "sainte_lague": "prop",
    "baseline": "base", "baseline_informed": "base",
}

# Short issue labels (≤10 chars, horizontal headers).
# Map from full issue text → short label.
ISSUE_SHORT_MAP = {
    "universal basic income": "UBI",
    "abortion": "Abortion",
    "firearms": "Firearms",
    "immigration": "Immigr.",
    "affirmative action": "Aff.Act.",
    "death penalty": "Death Pen.",
    "religious": "Relig.Frd.",
    "law enforcement": "Policing",
    "healthcare": "Healthcare",
    "fossil fuel": "Climate",
    "national service": "Nat.Svc.",
    "transgender": "Trans Ath.",
}


def _issue_to_short(issue_text):
    """Map a full issue question to a short label."""
    lower = issue_text.lower()
    for key, label in ISSUE_SHORT_MAP.items():
        if key in lower:
            return label
    return issue_text[:10]


def load_all_data():
    all_data = {}
    for model_name, fname in MODEL_FILES.items():
        path = os.path.join(RESULTS_DIR, fname)
        with open(path) as f:
            all_data[model_name] = json.load(f)
    return all_data


def compute_table(all_data):
    """Compute the max-model mean Likert score per (system, issue).

    Returns:
        issue_labels: ordered list of short issue labels
        table: {system: [CellValue, ...]} where CellValue = (mean, se)
    """
    # Get issue keys from first model.
    first_data = next(iter(all_data.values()))
    issue_keys = list(first_data.keys())
    issue_labels = [_issue_to_short(k) for k in issue_keys]

    table = {}

    for sys_name in SYSTEM_ORDER:
        row = []
        for issue_key in issue_keys:
            # Collect mean Likert per model for this (system, issue).
            model_means = []
            for model_name, model_data in all_data.items():
                issue = model_data.get(issue_key, {})
                scores = issue.get("scores", {})
                # scores: {user_id: {system: score}}
                voter_scores = []
                for uid, sys_scores in scores.items():
                    if sys_name in sys_scores:
                        voter_scores.append(float(sys_scores[sys_name]))
                if voter_scores:
                    model_means.append(np.mean(voter_scores))

            if model_means:
                avg_mean = np.mean(model_means)
                se = np.std(model_means, ddof=1) / math.sqrt(len(model_means)) if len(model_means) > 1 else 0.0
                row.append((avg_mean, se))
            else:
                row.append((0.0, 0.0))
        table[sys_name] = row

    return issue_labels, table


def generate_latex(issue_labels, table):
    """Generate the LaTeX table string."""
    n_issues = len(issue_labels)
    n_cols = n_issues + 1  # system name + issues

    # Find best system per issue (excluding baselines).
    best_per_issue = []
    non_baseline = [s for s in SYSTEM_ORDER if SYSTEM_CATEGORY[s] != "base"]
    for i in range(n_issues):
        best_val = -1
        best_sys = None
        for sys_name in non_baseline:
            val = table[sys_name][i][0]
            if val > best_val:
                best_val = val
                best_sys = sys_name
        best_per_issue.append(best_sys)

    # Also find overall best (including baselines) for bold.
    overall_best = []
    for i in range(n_issues):
        best_val = -1
        best_sys = None
        for sys_name in SYSTEM_ORDER:
            val = table[sys_name][i][0]
            if val > best_val:
                best_val = val
                best_sys = sys_name
        overall_best.append(best_sys)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Mean Likert scores ($1$--$5$) by electoral system and "
        r"issue, averaged across all seven models. Bold indicates the "
        r"best system per issue.}"
    )
    lines.append(r"\label{tab:likert_by_issue}")

    col_spec = "@{}l" + "c" * n_issues + "@{}"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row — horizontal, abbreviated.
    header_cells = [r"\textbf{System}"]
    for label in issue_labels:
        header_cells.append(r"\textbf{" + label + "}")
    lines.append(" & ".join(header_cells) + r" \\")
    lines.append(r"\midrule")

    # Category separators and rows.
    prev_cat = None
    for sys_name in SYSTEM_ORDER:
        cat = SYSTEM_CATEGORY[sys_name]
        if cat != prev_cat:
            if prev_cat is not None:
                lines.append(r"\midrule")
            cat_label = {
                "maj": "Majoritarian",
                "prop": "Proportional",
                "base": "Baselines",
            }[cat]
            cat_color = {"maj": "red", "prop": "green", "base": "blue"}[cat]
            lines.append(
                r"\multicolumn{"
                + str(n_cols)
                + r"}{l}{\cellcolor{"
                + cat_color
                + r"!10}\textit{"
                + cat_label
                + r"}} \\"
            )
            prev_cat = cat

        display = SYSTEM_DISPLAY[sys_name]
        cells = [r"\quad " + display]
        for i in range(n_issues):
            mean_val, se_val = table[sys_name][i]
            is_best = (overall_best[i] == sys_name)
            val_str = f"{mean_val:.2f}"
            se_str = f"{se_val:.2f}"
            cell = val_str + r"\tiny{$\pm$}" + se_str
            if is_best:
                cell = r"\textbf{" + val_str + "}" + r"\tiny{$\pm$}" + se_str
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"


if __name__ == "__main__":
    all_data = load_all_data()
    issue_labels, table = compute_table(all_data)
    latex = generate_latex(issue_labels, table)

    out_path = os.path.join(DRAFT_DIR, "table_issues.tex")
    with open(out_path, "w") as f:
        f.write(latex)
    print(f"Wrote {out_path}")
    print(f"Issues: {issue_labels}")
    print(f"Systems: {list(SYSTEM_DISPLAY.values())}")
