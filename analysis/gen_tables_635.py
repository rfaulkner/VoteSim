#!/usr/bin/env python3
"""Generate tables_635.tex from p635 (large-voter) golden results.

Produces two LaTeX tables matching the format of tables.tex:
  1. Mean Likert scores (±SE) per system, per model
  2. Mean rankings (±SE) per system, per model

Averages are taken across 12 issues; SE is across the 12 issue means.
"""

import json
import math
import os

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
    "Llama-3.3-70B": "divisive-12.issue.p635.llama-3.3-70b-instruct.json",
    "Mistral-Med-3": "divisive-12.issue.p635.mistral-medium-3.json",
}

# Column display order.
MODEL_ORDER = ["Llama-3.3-70B", "Mistral-Med-3"]

# Voting systems in display order.
SYSTEM_ORDER = [
    "fptp", "sntv", "alternative_vote", "trs",
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
    "sainte_lague": "Sainte-Lagu\\\"{e}",
    "baseline": "Baseline",
    "baseline_informed": "Oracle Med.",
}
SYSTEM_CATEGORY = {
    "fptp": "maj", "sntv": "maj", "alternative_vote": "maj", "trs": "maj",
    "dhondt": "prop", "stv": "prop", "sainte_lague": "prop",
    "baseline": "base", "baseline_informed": "base",
}


def load_all_data():
    """Load JSON result files for all models."""
    all_data = {}
    for model_name, fname in MODEL_FILES.items():
        path = os.path.join(RESULTS_DIR, fname)
        with open(path) as f:
            all_data[model_name] = json.load(f)
    return all_data


def compute_likert_table(all_data):
    """Compute mean Likert score ± SE for each (system, model).

    For each (system, model) cell:
      - For each issue, compute mean Likert across all voters.
      - Report mean ± SE across the 12 issue means.

    Returns:
        table: {model: {system: (mean, se)}}
    """
    table = {}
    for model_name in MODEL_ORDER:
        model_data = all_data[model_name]
        issue_keys = list(model_data.keys())
        table[model_name] = {}

        for sys_name in SYSTEM_ORDER:
            issue_means = []
            for issue_key in issue_keys:
                issue = model_data[issue_key]
                scores = issue.get("scores", {})
                voter_scores = []
                for uid, sys_scores in scores.items():
                    if sys_name in sys_scores:
                        voter_scores.append(float(sys_scores[sys_name]))
                if voter_scores:
                    issue_means.append(np.mean(voter_scores))

            if issue_means:
                avg = np.mean(issue_means)
                se = (np.std(issue_means, ddof=1) / math.sqrt(len(issue_means))
                      if len(issue_means) > 1 else 0.0)
                table[model_name][sys_name] = (avg, se)
            else:
                table[model_name][sys_name] = (0.0, 0.0)

    return table


def compute_ranking_table(all_data):
    """Compute mean ranking ± SE for each (system, model).

    For each (system, model) cell:
      - For each issue, compute mean rank (1-based) across all voters.
      - Report mean ± SE across the 12 issue means.

    Returns:
        table: {model: {system: (mean, se)}}
    """
    table = {}
    for model_name in MODEL_ORDER:
        model_data = all_data[model_name]
        issue_keys = list(model_data.keys())
        table[model_name] = {}

        for sys_name in SYSTEM_ORDER:
            issue_means = []
            for issue_key in issue_keys:
                issue = model_data[issue_key]
                rankings = issue.get("rankings", {})
                voter_ranks = []
                for uid, ranking_list in rankings.items():
                    if sys_name in ranking_list:
                        rank = ranking_list.index(sys_name) + 1  # 1-based
                        voter_ranks.append(rank)
                if voter_ranks:
                    issue_means.append(np.mean(voter_ranks))

            if issue_means:
                avg = np.mean(issue_means)
                se = (np.std(issue_means, ddof=1) / math.sqrt(len(issue_means))
                      if len(issue_means) > 1 else 0.0)
                table[model_name][sys_name] = (avg, se)
            else:
                table[model_name][sys_name] = (0.0, 0.0)

    return table


def _find_best_per_model(table, higher_is_better=True):
    """Find the best system per model (excluding baselines)."""
    best = {}
    non_baseline = [s for s in SYSTEM_ORDER if SYSTEM_CATEGORY[s] != "base"]
    for model_name in MODEL_ORDER:
        best_val = None
        best_sys = None
        for sys_name in non_baseline:
            val = table[model_name][sys_name][0]
            if best_val is None or (higher_is_better and val > best_val) or (not higher_is_better and val < best_val):
                best_val = val
                best_sys = sys_name
        best[model_name] = best_sys
    return best


def generate_likert_latex(table):
    """Generate the Likert scores LaTeX table."""
    n_models = len(MODEL_ORDER)
    n_cols = n_models + 1  # system name + models

    best = _find_best_per_model(table, higher_is_better=True)

    lines = []
    lines.append(r"% ============ LIKERT SCORES TABLE (p635) ============")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Mean Likert scores ($1$--$5$, higher is better) with "
        r"standard errors across the \textsc{Divisive-12} issue set "
        r"($N=635$ voters). Bold indicates the best score per model.}"
    )
    lines.append(r"\label{tab:likert_p635}")

    col_spec = "@{}l" + "c" * n_models + "@{}"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row.
    header = [r"\textbf{System}"]
    for m in MODEL_ORDER:
        header.append(r"\textbf{" + m + "}")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Rows with category separators.
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
        for model_name in MODEL_ORDER:
            mean_val, se_val = table[model_name][sys_name]
            val_str = f"{mean_val:.2f}"
            se_str = f"{se_val:.2f}"
            is_best = (best[model_name] == sys_name)
            if is_best:
                cell = r"\textbf{" + val_str + r"}" + r" \tiny{$\pm$}" + se_str
            else:
                cell = val_str + r" \tiny{$\pm$}" + se_str
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


def generate_ranking_latex(table):
    """Generate the Rankings LaTeX table."""
    n_models = len(MODEL_ORDER)
    n_cols = n_models + 1

    best = _find_best_per_model(table, higher_is_better=False)

    lines = []
    lines.append(r"% ============ RANKINGS TABLE (p635) ============")
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Mean rankings ($1$ = best) with standard errors across "
        r"the \textsc{Divisive-12} issue set ($N=635$ voters). Bold "
        r"indicates the best (lowest) rank per model.}"
    )
    lines.append(r"\label{tab:rankings_p635}")

    col_spec = "@{}l" + "c" * n_models + "@{}"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header row.
    header = [r"\textbf{System}"]
    for m in MODEL_ORDER:
        header.append(r"\textbf{" + m + "}")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Rows with category separators.
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
        for model_name in MODEL_ORDER:
            mean_val, se_val = table[model_name][sys_name]
            val_str = f"{mean_val:.2f}"
            se_str = f"{se_val:.2f}"
            is_best = (best[model_name] == sys_name)
            if is_best:
                cell = r"\textbf{" + val_str + r"}" + r" \tiny{$\pm$}" + se_str
            else:
                cell = val_str + r" \tiny{$\pm$}" + se_str
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines)


if __name__ == "__main__":
    all_data = load_all_data()

    likert_table = compute_likert_table(all_data)
    ranking_table = compute_ranking_table(all_data)

    likert_latex = generate_likert_latex(likert_table)
    ranking_latex = generate_ranking_latex(ranking_table)

    out_path = os.path.join(DRAFT_DIR, "tables_635.tex")
    with open(out_path, "w") as f:
        f.write(likert_latex + "\n\n" + ranking_latex + "\n")

    print(f"Saved {out_path}")
    print(f"Models: {MODEL_ORDER}")
    print(f"Systems: {[SYSTEM_DISPLAY[s] for s in SYSTEM_ORDER]}")
