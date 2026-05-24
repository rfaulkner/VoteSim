#!/usr/bin/env python3
"""Generate harm analysis outputs for draft/harm_analysis.tex.

Produces:
  1. draft/harm_h2h_heatmap.pdf  — head-to-head pairwise win-rate heatmap
  2. draft/table_harm_issues.tex — mean harm scores by system and issue
"""

import glob
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "golden_data")
DRAFT_DIR = os.path.join(SCRIPT_DIR, "..", "draft")

# ── System config ─────────────────────────────────────────────────────
SYSTEMS = [
    "sntv", "fptp", "trs", "alternative_vote",
    "stv", "dhondt", "sainte_lague",
    "baseline", "baseline_informed",
]
DISPLAY = {
    "fptp": "FPTP", "sntv": "PBV", "alternative_vote": "IRV",
    "trs": "TRS", "dhondt": "D'Hondt", "stv": "STV",
    "sainte_lague": "S-L", "baseline": "Base Model",
    "baseline_informed": "Base Med.",
}
SYSTEM_CATEGORY = {
    "fptp": "maj", "sntv": "maj", "alternative_vote": "maj", "trs": "maj",
    "dhondt": "prop", "stv": "prop", "sainte_lague": "prop",
    "baseline": "base", "baseline_informed": "base",
}

FRONTIER_JUDGES = {"claude-opus-4.7", "gemini-3-flash-preview", "gpt-4o-2024-11-20"}

# Short labels for issues
ISSUE_SHORT = {
    "vaccination": "Vaccine",
    "facial recognition": "Facial R.",
    "social media": "Platform",
    "water resources": "Water",
    "genetic editing": "Gene Ed.",
    "social credit": "Soc.Credit",
    "autonomous ai": "Auton.AI",
    "hate speech": "Hate Sp.",
    "border security": "Border",
    "surveillance": "Surveill.",
    "pharmaceutical": "Drug Pr.",
    "fossil fuel": "Fossil F.",
}

N = len(SYSTEMS)


def _issue_short(text):
    lower = text.lower()
    for key, label in ISSUE_SHORT.items():
        if key in lower:
            return label
    return text[:10]


def load_frontier_files():
    """Return list of (source, judge, filepath) for 6 frontier judge files."""
    files = []
    for fp in sorted(glob.glob(os.path.join(DATA_DIR, "harm-12.judge.*.judged_by.*.json"))):
        fn = os.path.basename(fp)
        parts = fn.replace("harm-12.judge.", "").replace(".json", "").split(".judged_by.")
        source, judge = parts
        if judge in FRONTIER_JUDGES:
            files.append((source, judge, fp))
    return files


# =====================================================================
# 1. HEAD-TO-HEAD HEATMAP
# =====================================================================

def compute_h2h(file_list):
    wins = np.zeros((N, N), dtype=int)
    counts = np.zeros((N, N), dtype=int)
    for _, _, fpath in file_list:
        with open(fpath) as f:
            jdata = json.load(f)
        for issue_d in jdata.values():
            for val in issue_d.get("pairwise", {}).values():
                w, l = val.get("winner"), val.get("loser")
                if w in SYSTEMS and l in SYSTEMS:
                    wi, li = SYSTEMS.index(w), SYSTEMS.index(l)
                    wins[wi, li] += 1
                    counts[wi, li] += 1
                    counts[li, wi] += 1
    rates = np.full((N, N), 0.5)
    for i in range(N):
        for j in range(N):
            if i != j and counts[i, j] > 0:
                rates[i, j] = wins[i, j] / counts[i, j]
    return rates, counts


def make_h2h_heatmap(rates):
    """Generate a clean H2H win-rate heatmap and save to PDF + PNG."""
    labels = [DISPLAY[s] for s in SYSTEMS]

    fig, ax = plt.subplots(figsize=(7.2, 6.2))

    # ── Main heatmap ──
    masked = rates.copy()
    np.fill_diagonal(masked, np.nan)

    cmap = plt.cm.RdYlGn
    im = ax.imshow(masked, cmap=cmap, vmin=0.15, vmax=0.85, aspect="equal")

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels(labels, fontsize=8.5, rotation=45, ha="right")
    ax.set_yticklabels(labels, fontsize=8.5)

    # Category separator lines
    cat_boundaries = []
    prev_cat = None
    for i, s in enumerate(SYSTEMS):
        cat = SYSTEM_CATEGORY[s]
        if cat != prev_cat and prev_cat is not None:
            cat_boundaries.append(i - 0.5)
        prev_cat = cat

    for b in cat_boundaries:
        ax.axhline(b, color="black", linewidth=1.2)
        ax.axvline(b, color="black", linewidth=1.2)

    # Cell annotations
    for i in range(N):
        for j in range(N):
            if i == j:
                ax.text(j, i, "—", ha="center", va="center",
                             fontsize=7.5, color="gray")
            else:
                pct = rates[i, j] * 100
                color = "white" if pct > 70 or pct < 30 else "black"
                ax.text(j, i, f"{pct:.0f}", ha="center", va="center",
                             fontsize=7.5,
                             fontweight="bold" if pct > 60 else "normal",
                             color=color)

    ax.set_xlabel("Column System", fontsize=9, labelpad=6)
    ax.set_ylabel("Row System", fontsize=9, labelpad=6)

    # Colorbar on the right
    cbar = fig.colorbar(im, ax=ax, orientation="vertical",
                        shrink=0.8, pad=0.05, aspect=25)
    cbar.set_label("Pairwise Win Rate (row judged less harmful than column)",
                   fontsize=8.5, labelpad=10)
    cbar.ax.tick_params(labelsize=7)

    fig.suptitle(
        "Head-to-Head Harm Win Rates\n"
        "(pooled across 2 source models × 3 judges × 12 issues)",
        fontsize=10.5, y=0.96,
    )

    # Save PDF and PNG
    for ext, dpi in [("pdf", 300), ("png", 150)]:
        out = os.path.join(DRAFT_DIR, f"harm_h2h_heatmap.{ext}")
        fig.savefig(out, dpi=dpi, bbox_inches="tight")
        print(f"Wrote {out}")
    plt.close(fig)


# =====================================================================
# 2. HARM-PER-ISSUE TABLE (LaTeX)
# =====================================================================

def compute_harm_table(file_list):
    """Compute mean harm score per (system, issue), averaged across judges.

    Returns:
        issue_labels: list of short labels
        table: {system: [(mean, se), ...]}  one entry per issue
    """
    from collections import defaultdict

    # Collect scores: (source, judge, issue, system) -> score
    raw = defaultdict(list)  # (issue, system) -> [scores across source×judge]
    issue_keys_ordered = None

    for source, judge, fpath in file_list:
        with open(fpath) as f:
            jdata = json.load(f)
        if issue_keys_ordered is None:
            issue_keys_ordered = list(jdata.keys())
        for issue, issue_d in jdata.items():
            mhs = issue_d["aggregate"]["mean_harm_score"]
            for s in SYSTEMS:
                if s in mhs:
                    raw[(issue, s)].append(mhs[s])

    issue_labels = [_issue_short(k) for k in issue_keys_ordered]

    table = {}
    for sys_name in SYSTEMS:
        row = []
        for issue_key in issue_keys_ordered:
            vals = raw.get((issue_key, sys_name), [])
            if vals:
                row.append((np.mean(vals), np.std(vals, ddof=1) / np.sqrt(len(vals)) if len(vals) > 1 else 0.0))
            else:
                row.append((0.0, 0.0))
        table[sys_name] = row

    return issue_labels, issue_keys_ordered, table


def generate_harm_latex(issue_labels, table):
    """Generate LaTeX table of harm scores per issue, styled like table_issues.tex."""
    n_issues = len(issue_labels)
    n_cols = n_issues + 1

    # Find best system per issue (lowest harm, excluding baselines)
    non_baseline = [s for s in SYSTEMS if SYSTEM_CATEGORY[s] != "base"]
    best_neg = []
    for i in range(n_issues):
        best_val = 999
        best_sys = None
        for s in non_baseline:
            val = table[s][i][0]
            if 0 < val < best_val:
                best_val = val
                best_sys = s
        best_neg.append(best_sys)

    # Overall best (including baselines)
    overall_best = []
    for i in range(n_issues):
        best_val = 999
        best_sys = None
        for s in SYSTEMS:
            val = table[s][i][0]
            if 0 < val < best_val:
                best_val = val
                best_sys = s
        overall_best.append(best_sys)

    lines = []
    lines.append(r"\begin{table*}[t]")
    lines.append(r"\centering")
    lines.append(r"\setlength{\tabcolsep}{3pt}")
    lines.append(r"\scriptsize")
    lines.append(
        r"\caption{Mean harm scores ($1$--$5$, lower is safer) by electoral system and "
        r"issue, averaged across two source models and three frontier judges. "
        r"\textbf{Bold} indicates the least harmful system per issue.}"
    )
    lines.append(r"\label{tab:harm_by_issue}")

    col_spec = "@{}l" + "c" * n_issues + "@{}"
    lines.append(r"\begin{tabular}{" + col_spec + "}")
    lines.append(r"\toprule")

    # Header
    header = [r"\textbf{System}"]
    for label in issue_labels:
        header.append(r"\textbf{" + label + "}")
    lines.append(" & ".join(header) + r" \\")
    lines.append(r"\midrule")

    # Rows with category separators
    prev_cat = None
    for sys_name in SYSTEMS:
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

        display = DISPLAY[sys_name]
        cells = [r"\quad " + display]
        for i in range(n_issues):
            mean_val, se_val = table[sys_name][i]
            val_str = f"{mean_val:.2f}"
            se_str = f"{se_val:.2f}"
            is_best = (overall_best[i] == sys_name)
            if is_best:
                cell = r"\textbf{" + val_str + "}" + r"\tiny{$\pm$}" + se_str
            else:
                cell = val_str + r"\tiny{$\pm$}" + se_str
            cells.append(cell)
        lines.append(" & ".join(cells) + r" \\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(r"\end{table*}")

    return "\n".join(lines) + "\n"


# =====================================================================
# MAIN
# =====================================================================

if __name__ == "__main__":
    file_list = load_frontier_files()
    print(f"Using {len(file_list)} judge files")

    # 1. H2H heatmap
    rates, counts = compute_h2h(file_list)
    make_h2h_heatmap(rates)

    # 2. Harm-per-issue table
    issue_labels, issue_keys, table = compute_harm_table(file_list)
    latex = generate_harm_latex(issue_labels, table)
    table_path = os.path.join(DRAFT_DIR, "table_harm_issues.tex")
    with open(table_path, "w") as f:
        f.write(latex)
    print(f"Wrote {table_path}")
    print(f"Issues: {issue_labels}")
