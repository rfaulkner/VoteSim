#!/usr/bin/env python3
"""Generate tables_harm.tex for Harm-12 dataset.

Computes:
  1. Mean overall harm score +/- SE averaged across all three judges
     (Claude-Opus, Gemini-Flash, GPT-4o) per source model.
  2. Mean harm dimension scores +/- SE averaged across all three judges
     per source model.

Outputs LaTeX code for these tables.
"""

import json
import math
import os
import numpy as np

DATA_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "golden_data",
)

DRAFT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "draft",
)

# Source model -> list of judge files
SOURCE_MODELS = {
    "Llama-3.3-70B": [
        "harm-12.judge.llama-3.3-70b-instruct.judged_by.claude-opus-4.7.json",
        "harm-12.judge.llama-3.3-70b-instruct.judged_by.gemini-3-flash-preview.json",
        "harm-12.judge.llama-3.3-70b-instruct.judged_by.gpt-4o-2024-11-20.json",
    ],
    "Mistral-Med-3": [
        "harm-12.judge.mistral-medium-3.judged_by.claude-opus-4.7.json",
        "harm-12.judge.mistral-medium-3.judged_by.gemini-3-flash-preview.json",
        "harm-12.judge.mistral-medium-3.judged_by.gpt-4o-2024-11-20.json",
    ],
}

SYSTEM_ORDER = [
    "sntv", "fptp", "trs", "alternative_vote",
    "stv", "dhondt", "sainte_lague",
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
    "baseline": "Mod.\\ Base.",
    "baseline_informed": "Med.\\ Base.",
}

SYSTEM_CATEGORY = {
    "fptp": "maj", "sntv": "maj", "alternative_vote": "maj", "trs": "maj",
    "dhondt": "prop", "stv": "prop", "sainte_lague": "prop",
    "baseline": "base", "baseline_informed": "base",
}

DELIBERATED = {"fptp", "sntv", "alternative_vote", "trs", "dhondt", "stv", "sainte_lague"}

DIMENSIONS = ["discrimination", "vulnerable_populations", "civil_liberties", "abuse_potential", "extremity", "economic_harm"]
DIMENSION_DISPLAY = {
    "discrimination": "Discrimination",
    "vulnerable_populations": "Vulnerable Pop.",
    "civil_liberties": "Civil Liberties",
    "abuse_potential": "Abuse Potential",
    "extremity": "Extremity",
    "economic_harm": "Economic Harm",
}

MODEL_DISPLAY = {
    "Llama-3.3-70B": "Llama-3.3-70B",
    "Mistral-Med-3": "Mistral-Med-3",
}

def load_data(fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"WARNING: {path} does not exist!")
        return None
    with open(path) as f:
        return json.load(f)


def compute_overall_averaged(source_model_files):
    """Average overall harm across all judges for a given source model.

    Collects one score per (judge, issue) pair and computes mean +/- SE.
    """
    result = {}
    for sys_name in SYSTEM_ORDER:
        scores = []
        for fname in source_model_files:
            data = load_data(fname)
            if data is None:
                continue
            for issue_key, issue in data.items():
                mean_harm = issue.get("aggregate", {}).get("mean_harm_score", {})
                if sys_name in mean_harm:
                    scores.append(float(mean_harm[sys_name]))
        if scores:
            avg = np.mean(scores)
            se = np.std(scores, ddof=1) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
            result[sys_name] = (avg, se)
        else:
            result[sys_name] = (0.0, 0.0)
    return result


def compute_dimensions_averaged(source_model_files):
    """Average dimension scores across all judges for a given source model."""
    result = {}
    for sys_name in SYSTEM_ORDER:
        result[sys_name] = {}
        for dim in DIMENSIONS:
            scores = []
            for fname in source_model_files:
                data = load_data(fname)
                if data is None:
                    continue
                for issue_key, issue in data.items():
                    dim_scores = issue.get("aggregate", {}).get("mean_dimension_scores", {}).get(sys_name, {})
                    if dim in dim_scores:
                        scores.append(float(dim_scores[dim]))
            if scores:
                avg = np.mean(scores)
                se = np.std(scores, ddof=1) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
                result[sys_name][dim] = (avg, se)
            else:
                result[sys_name][dim] = (0.0, 0.0)
    return result


def _format_cell(val_str, se_str, is_best_overall, is_best_deliberated):
    """Return a formatted cell with bold/underline as appropriate."""
    if is_best_overall and is_best_deliberated:
        return r"\underline{\textbf{" + val_str + r"}}" + r" \tiny{$\pm$}" + se_str
    elif is_best_overall:
        return r"\textbf{" + val_str + r"}" + r" \tiny{$\pm$}" + se_str
    elif is_best_deliberated:
        return r"\underline{" + val_str + r"}" + r" \tiny{$\pm$}" + se_str
    else:
        return val_str + r" \tiny{$\pm$}" + se_str


def generate_overall_table_latex(table):
    """Generate overall harm table with one column per source model, averaged over judges."""
    model_names = list(table.keys())
    n_cols = len(model_names) + 1

    # Find best overall and best deliberated per column
    best_overall = {}
    best_delib = {}
    for model in model_names:
        best_val = None
        best_sys = None
        best_dval = None
        best_dsys = None
        for sys in SYSTEM_ORDER:
            val = table[model][sys][0]
            if best_val is None or val < best_val:
                best_val = val
                best_sys = sys
            if sys in DELIBERATED and (best_dval is None or val < best_dval):
                best_dval = val
                best_dsys = sys
        best_overall[model] = best_sys
        best_delib[model] = best_dsys

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Mean overall harm scores ($1$--$5$, lower is better) with standard errors "
        r"across the \textsc{Harm-12} issue set, averaged over Claude-Opus, Gemini-Flash, and "
        r"GPT-4o judges. \textbf{Bold} indicates the best (lowest) score per column; "
        r"\underline{underline} indicates the best among deliberated systems.}"
    )
    lines.append(r"\label{tab:overall_harm}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(model_names) + r"@{}}")
    lines.append(r"\toprule")

    row1 = [r"\textbf{System}"]
    for model in model_names:
        row1.append(r"\textbf{" + MODEL_DISPLAY[model] + r"}")
    lines.append(" & ".join(row1) + r" \\")
    lines.append(r"\midrule")

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
        for model in model_names:
            mean_val, se_val = table[model][sys_name]
            val_str = f"{mean_val:.2f}"
            se_str = f"{se_val:.2f}"
            is_best = (best_overall[model] == sys_name)
            is_best_d = (best_delib[model] == sys_name)
            cells.append(_format_cell(val_str, se_str, is_best, is_best_d))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


def generate_dimensions_table_latex(table, model_title, label):
    n_cols = len(DIMENSIONS) + 1

    # Find best overall and best deliberated per dimension
    best_overall = {}
    best_delib = {}
    for dim in DIMENSIONS:
        best_val = None
        best_sys = None
        best_dval = None
        best_dsys = None
        for sys in SYSTEM_ORDER:
            val = table[sys][dim][0]
            if best_val is None or val < best_val:
                best_val = val
                best_sys = sys
            if sys in DELIBERATED and (best_dval is None or val < best_dval):
                best_dval = val
                best_dsys = sys
        best_overall[dim] = best_sys
        best_delib[dim] = best_dsys

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Mean harm dimension scores ($1$--$5$, lower is better) with standard errors "
        r"across the \textsc{Harm-12} issue set for " + model_title + r", averaged over "
        r"Claude-Opus, Gemini-Flash, and GPT-4o judges. "
        r"\textbf{Bold} indicates the best (lowest) score per dimension; "
        r"\underline{underline} indicates the best among deliberated systems.}"
    )
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(DIMENSIONS) + r"@{}}")
    lines.append(r"\toprule")

    row = [r"\textbf{System}"]
    for dim in DIMENSIONS:
        row.append(r"\textbf{" + DIMENSION_DISPLAY[dim] + "}")
    lines.append(" & ".join(row) + r" \\")
    lines.append(r"\midrule")

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
        for dim in DIMENSIONS:
            mean_val, se_val = table[sys_name][dim]
            val_str = f"{mean_val:.2f}"
            se_str = f"{se_val:.2f}"
            is_best = (best_overall[dim] == sys_name)
            is_best_d = (best_delib[dim] == sys_name)
            cells.append(_format_cell(val_str, se_str, is_best, is_best_d))
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular*}")
    lines.append(r"\end{table*}")
    return "\n".join(lines)


if __name__ == "__main__":
    # Overall table: one column per source model, averaged over all judges
    overall_table = {}
    for model_name, files in SOURCE_MODELS.items():
        overall_table[model_name] = compute_overall_averaged(files)
    overall_latex = generate_overall_table_latex(overall_table)

    # Dimension tables: one per source model, averaged over all judges
    dim_latex_blocks = []
    for model_name, files in SOURCE_MODELS.items():
        dim_table = compute_dimensions_averaged(files)
        label = "tab:dimensions_" + model_name.lower().replace("-", "_").replace(".", "_")
        latex = generate_dimensions_table_latex(dim_table, model_name, label)
        dim_latex_blocks.append(latex)

    out_path = os.path.join(DRAFT_DIR, "tables_harm.tex")
    with open(out_path, "w") as f:
        f.write(overall_latex + "\n\n" + "\n\n".join(dim_latex_blocks) + "\n")

    print(f"Successfully generated {out_path}")
