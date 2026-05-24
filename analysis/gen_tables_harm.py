#!/usr/bin/env python3
"""Generate tables_harm.tex for Harm-12 dataset.

Computes:
  1. Mean overall harm score ± SE (across 12 issues) comparing Llama-3.3-70B and Mistral-Med-3
     across 3 different judge models (Claude-Opus, Gemini-Flash, GPT-4o).
  2. Mean harm dimension scores ± SE (across 12 issues) for Llama-3.3-70B and Mistral-Med-3
     using Claude-Opus as the golden judge.

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

# Define configurations for the overall comparison table
CONFIGS = [
    ("Llama (Opus)", "harm-12.judge.llama-3.3-70b-instruct.judged_by.claude-opus-4.7.json"),
    ("Llama (Flash)", "harm-12.judge.llama-3.3-70b-instruct.judged_by.gemini-3-flash-preview.json"),
    ("Llama (GPT-4o)", "harm-12.judge.llama-3.3-70b-instruct.judged_by.gpt-4o-2024-11-20.json"),
    ("Mistral (Opus)", "harm-12.judge.mistral-medium-3.judged_by.claude-opus-4.7.json"),
    ("Mistral (Flash)", "harm-12.judge.mistral-medium-3.judged_by.gemini-3-flash-preview.json"),
    ("Mistral (GPT-4o)", "harm-12.judge.mistral-medium-3.judged_by.gpt-4o-2024-11-20.json"),
]

DIMENSION_CONFIGS = {
    "Llama-3.3-70B": "harm-12.judge.llama-3.3-70b-instruct.judged_by.claude-opus-4.7.json",
    "Mistral-Med-3": "harm-12.judge.mistral-medium-3.judged_by.claude-opus-4.7.json",
}

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

DIMENSIONS = ["discrimination", "vulnerable_populations", "civil_liberties", "abuse_potential", "extremity", "economic_harm"]
DIMENSION_DISPLAY = {
    "discrimination": "Discrimination",
    "vulnerable_populations": "Vulnerable Pop.",
    "civil_liberties": "Civil Liberties",
    "abuse_potential": "Abuse Potential",
    "extremity": "Extremity",
    "economic_harm": "Economic Harm",
}

def load_data(fname):
    path = os.path.join(DATA_DIR, fname)
    if not os.path.exists(path):
        print(f"WARNING: {path} does not exist!")
        return None
    with open(path) as f:
        return json.load(f)

def compute_overall_harm_table():
    table = {}
    for display_name, fname in CONFIGS:
        data = load_data(fname)
        if data is None:
            continue
        table[display_name] = {}
        for sys_name in SYSTEM_ORDER:
            scores = []
            for issue_key, issue in data.items():
                mean_harm = issue.get("aggregate", {}).get("mean_harm_score", {})
                if sys_name in mean_harm:
                    scores.append(float(mean_harm[sys_name]))
            if scores:
                avg = np.mean(scores)
                se = np.std(scores, ddof=1) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
                table[display_name][sys_name] = (avg, se)
            else:
                table[display_name][sys_name] = (0.0, 0.0)
    return table

def compute_dimensions_table(fname):
    data = load_data(fname)
    if data is None:
        return None
    table = {}
    for sys_name in SYSTEM_ORDER:
        table[sys_name] = {}
        for dim in DIMENSIONS:
            scores = []
            for issue_key, issue in data.items():
                dim_scores = issue.get("aggregate", {}).get("mean_dimension_scores", {}).get(sys_name, {})
                if dim in dim_scores:
                    scores.append(float(dim_scores[dim]))
            if scores:
                avg = np.mean(scores)
                se = np.std(scores, ddof=1) / math.sqrt(len(scores)) if len(scores) > 1 else 0.0
                table[sys_name][dim] = (avg, se)
            else:
                table[sys_name][dim] = (0.0, 0.0)
    return table

def generate_overall_table_latex(table):
    active_configs = [c[0] for c in CONFIGS if c[0] in table]
    n_cols = len(active_configs) + 1
    
    # Find best (lowest harm) per column
    best = {}
    for cfg in active_configs:
        best_val = None
        best_sys = None
        for sys in SYSTEM_ORDER:
            val = table[cfg][sys][0]
            if best_val is None or val < best_val:
                best_val = val
                best_sys = sys
        best[cfg] = best_sys

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Mean overall harm scores ($1$--$5$, lower is better) with standard errors across the \textsc{Harm-12} issue set. Evaluated across Llama-3.3-70B and Mistral-Med-3 source models under Claude-Opus, Gemini-Flash, and GPT-4o judges. Bold indicates the best (lowest) score per column.}")
    lines.append(r"\label{tab:overall_harm}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(active_configs) + r"@{}}")
    lines.append(r"\toprule")
    
    # Header Row 1: Source Model / Judge Names
    row1 = [r"\textbf{System}"]
    for cfg in active_configs:
        row1.append(r"\textbf{" + cfg + "}")
    lines.append(" & ".join(row1) + r" \\")
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
        for cfg in active_configs:
            mean_val, se_val = table[cfg][sys_name]
            val_str = f"{mean_val:.2f}"
            se_str = f"{se_val:.2f}"
            is_best = (best[cfg] == sys_name)
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

def generate_dimensions_table_latex(table, model_title, label):
    n_cols = len(DIMENSIONS) + 1
    
    # Find best (lowest harm) per dimension column
    best = {}
    for dim in DIMENSIONS:
        best_val = None
        best_sys = None
        for sys in SYSTEM_ORDER:
            val = table[sys][dim][0]
            if best_val is None or val < best_val:
                best_val = val
                best_sys = sys
        best[dim] = best_sys

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\scriptsize")
    lines.append(r"\caption{Mean harm dimension scores ($1$--$5$, lower is better) with standard errors across the \textsc{Harm-12} issue set for **" + model_title + "** under the Claude-Opus judge. Bold indicates the best (lowest) score per dimension column.}")
    lines.append(r"\label{" + label + "}")
    lines.append(r"\begin{tabular*}{\textwidth}{@{\extracolsep{\fill}}l" + "c" * len(DIMENSIONS) + r"@{}}")
    lines.append(r"\toprule")
    
    # Header row
    row = [r"\textbf{System}"]
    for dim in DIMENSIONS:
        row.append(r"\textbf{" + DIMENSION_DISPLAY[dim] + "}")
    lines.append(" & ".join(row) + r" \\")
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
        for dim in DIMENSIONS:
            mean_val, se_val = table[sys_name][dim]
            val_str = f"{mean_val:.2f}"
            se_str = f"{se_val:.2f}"
            is_best = (best[dim] == sys_name)
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
    overall_table = compute_overall_harm_table()
    overall_latex = generate_overall_table_latex(overall_table)
    
    dim_latex_blocks = []
    for mtitle, fname in DIMENSION_CONFIGS.items():
        dim_table = compute_dimensions_table(fname)
        if dim_table is not None:
            label = "tab:dimensions_" + mtitle.lower().replace("-", "_").replace(".", "_")
            latex = generate_dimensions_table_latex(dim_table, mtitle, label)
            dim_latex_blocks.append(latex)
            
    out_path = os.path.join(DRAFT_DIR, "tables_harm.tex")
    with open(out_path, "w") as f:
        f.write(overall_latex + "\n\n" + "\n\n".join(dim_latex_blocks) + "\n")
        
    print(f"Successfully generated {out_path}")
