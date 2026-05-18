#!/usr/bin/env python3
"""Combined preference-vs-harm scatter plot for both source models.

Two panels stacked vertically, one shared legend across the top,
shared axis labels, using Claude-Opus as the harm judge.
"""

import json
import os
import numpy as np
from collections import defaultdict
from scipy import stats
from matplotlib.patches import Ellipse
from matplotlib.lines import Line2D

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_data")
DRAFT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "draft")

# ── System grouping ──────────────────────────────────────────────────
GROUPS = {
    "Baseline": ["baseline", "baseline_informed"],
    "Majoritarian": ["fptp", "alternative_vote", "trs", "sntv"],
    "Proportional": ["dhondt", "sainte_lague", "stv"],
}
SYSTEM_TO_GROUP = {}
for g, systems in GROUPS.items():
    for s in systems:
        SYSTEM_TO_GROUP[s] = g

SYSTEM_MARKERS = {
    "baseline": "s", "baseline_informed": "D",
    "fptp": "o", "alternative_vote": "^", "trs": "v", "sntv": ">",
    "dhondt": "o", "sainte_lague": "^", "stv": "v",
}
SYSTEM_LABELS = {
    "alternative_vote": "AV", "baseline": "Base",
    "baseline_informed": "Oracle", "dhondt": "D'Hondt",
    "fptp": "FPTP", "sainte_lague": "S-L", "sntv": "SNTV",
    "stv": "STV", "trs": "TRS",
}

PLOT_COLORS = {
    "Baseline": "#7f7f7f",
    "Majoritarian": "#d62728",
    "Proportional": "#1f77b4",
}
PLOT_FILL = {
    "Baseline": "#cccccc",
    "Majoritarian": "#f4a5a7",
    "Proportional": "#aec7e8",
}

MODELS = [
    {
        "key": "llama",
        "label": "Llama-3.3-70B",
        "judge_file": os.path.join(
            DATA_DIR,
            "harm-12.judge.llama-3.3-70b-instruct.judged_by.claude-opus-4.7.json",
        ),
        "issue_file": os.path.join(
            DATA_DIR, "harm-12.issue.llama-3.3-70b-instruct.json"
        ),
    },
    {
        "key": "mistral",
        "label": "Mistral-Medium-3",
        "judge_file": os.path.join(
            DATA_DIR,
            "harm-12.judge.mistral-medium-3.judged_by.claude-opus-4.7.json",
        ),
        "issue_file": os.path.join(
            DATA_DIR, "harm-12.issue.mistral-medium-3.json"
        ),
    },
]


def confidence_ellipse(x, y, ax, n_std=1.5, **kwargs):
    """Draw a covariance-based confidence ellipse."""
    if len(x) < 3:
        return
    cov = np.cov(x, y)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = eigvals.argsort()[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    angle = np.degrees(np.arctan2(*eigvecs[:, 0][::-1]))
    width, height = 2 * n_std * np.sqrt(eigvals)
    ellipse = Ellipse(
        xy=(np.mean(x), np.mean(y)),
        width=width, height=height, angle=angle, **kwargs,
    )
    ax.add_patch(ellipse)


def build_records(issue_data, judge_data):
    """Build preference/harm records using Claude-Opus judge."""
    records = []
    for issue in issue_data:
        scores = issue_data[issue]["scores"]
        systems = list(list(scores.values())[0].keys())
        pref_per_sys = defaultdict(list)
        for voter_scores in scores.values():
            for s, sc in voter_scores.items():
                pref_per_sys[s].append(sc)
        mean_pref = {s: np.mean(v) for s, v in pref_per_sys.items()}

        harm_per_sys = {}
        if issue in judge_data:
            for s, h in judge_data[issue]["aggregate"]["mean_harm_score"].items():
                harm_per_sys[s] = h

        for s in systems:
            if s in harm_per_sys:
                records.append(dict(
                    system=s, group=SYSTEM_TO_GROUP[s],
                    preference=mean_pref[s], harm=harm_per_sys[s],
                ))
    return records


# ── Style ────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.linewidth": 0.8,
    "axes.edgecolor": "#333333",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
    "text.color": "#222222",
    "figure.facecolor": "white",
    "axes.facecolor": "#fafafa",
})

# ── Figure: 2 rows, 1 column ────────────────────────────────────────
fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(8, 10),
    sharex=True, sharey=True,
    gridspec_kw={"hspace": 0.08},
)

# Compute shared axis limits across both models
all_prefs, all_harms = [], []
all_records = {}
for mcfg in MODELS:
    with open(mcfg["judge_file"]) as f:
        judge_data = json.load(f)
    with open(mcfg["issue_file"]) as f:
        issue_data = json.load(f)
    records = build_records(issue_data, judge_data)
    all_records[mcfg["key"]] = records
    all_prefs.extend(r["preference"] for r in records)
    all_harms.extend(r["harm"] for r in records)

x_pad = 0.15
y_pad = 0.15
x_min, x_max = min(all_prefs) - x_pad, max(all_prefs) + x_pad
y_min, y_max = min(all_harms) - y_pad, max(all_harms) + y_pad

label_offsets = {
    "Baseline":     (-65, 50),
    "Majoritarian": (-55, -50),
    "Proportional": (55, 50),
}

for idx, mcfg in enumerate(MODELS):
    ax = axes[idx]
    records = all_records[mcfg["key"]]
    prefs = np.array([r["preference"] for r in records])
    harms = np.array([r["harm"] for r in records])
    rp, pp = stats.pearsonr(prefs, harms)

    # Confidence ellipses
    for g in ["Baseline", "Majoritarian", "Proportional"]:
        gx = np.array([r["preference"] for r in records if r["group"] == g])
        gy = np.array([r["harm"] for r in records if r["group"] == g])
        confidence_ellipse(
            gx, gy, ax, n_std=1.8,
            facecolor=PLOT_FILL[g], edgecolor=PLOT_COLORS[g],
            alpha=0.22, linewidth=1.5, linestyle="-", zorder=1,
        )

    # Data points
    for r in records:
        ax.scatter(
            r["preference"], r["harm"],
            c=PLOT_COLORS[r["group"]],
            marker=SYSTEM_MARKERS[r["system"]],
            s=55, alpha=0.7,
            edgecolors="white", linewidth=0.5, zorder=3,
        )

    # Group centroids + labels
    for g in ["Baseline", "Majoritarian", "Proportional"]:
        gx = np.mean([r["preference"] for r in records if r["group"] == g])
        gy = np.mean([r["harm"] for r in records if r["group"] == g])
        ax.scatter(
            gx, gy, c=PLOT_COLORS[g], marker="o", s=160,
            edgecolors="black", linewidth=1.4, zorder=5,
        )
        dx, dy = label_offsets[g]
        ax.annotate(
            g, (gx, gy), textcoords="offset points", xytext=(dx, dy),
            fontsize=11, fontweight="bold", color=PLOT_COLORS[g],
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=PLOT_COLORS[g], alpha=0.92, linewidth=0.8),
            arrowprops=dict(arrowstyle="-|>", color=PLOT_COLORS[g],
                            lw=1.0, connectionstyle="arc3,rad=0.15"),
            zorder=6,
        )

    # OLS trend line
    sl, ic, _, _, _ = stats.linregress(prefs, harms)
    xl = np.linspace(x_min, x_max, 100)
    ax.plot(xl, sl * xl + ic, color="#333333", linestyle="--",
            alpha=0.5, linewidth=1.2, zorder=2)

    # Correlation annotation (top-right)
    ax.annotate(
        f"r = {rp:.2f},  p = {pp:.1e}",
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=12, fontstyle="italic",
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#cccccc", alpha=0.9),
    )

    # Source model label (top-left area, below where legend will be)
    ax.annotate(
        mcfg["label"],
        xy=(0.03, 0.95), xycoords="axes fraction",
        ha="left", va="top", fontsize=14, fontweight="bold",
        color="#222222",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#aaaaaa", alpha=0.9, linewidth=0.6),
    )

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.grid(True, alpha=0.15, linewidth=0.5, color="#888888")
    ax.tick_params(labelsize=13)

    # Remove x-tick labels on top panel
    if idx == 0:
        ax.tick_params(axis="x", labelbottom=False)

# ── Shared x-axis label (bottom only) ───────────────────────────────
axes[-1].set_xlabel("Mean Preference Score (Likert)", fontsize=15, labelpad=10)

# ── Shared y-axis label (centered between both) ─────────────────────
fig.text(
    0.02, 0.5, "Mean Harm Score (Claude-Opus judge)",
    va="center", ha="center", rotation="vertical",
    fontsize=15, color="#222222",
)

# ── Shared horizontal legend across the top ──────────────────────────
sys_handles = []
for g in ["Baseline", "Majoritarian", "Proportional"]:
    for s in GROUPS[g]:
        sys_handles.append(
            Line2D(
                [0], [0], marker=SYSTEM_MARKERS[s], color="w",
                markerfacecolor=PLOT_COLORS[g], markersize=9,
                markeredgecolor="#555555", markeredgewidth=0.5,
                label=SYSTEM_LABELS[s],
            )
        )
sys_handles.append(
    Line2D([0], [0], linestyle="--", color="#333333", alpha=0.5,
           linewidth=1.5, label="OLS fit")
)

fig.legend(
    handles=sys_handles, loc="upper center",
    ncol=len(sys_handles), fontsize=11,
    frameon=True, framealpha=0.95, edgecolor="#cccccc",
    handletextpad=0.4, columnspacing=1.0,
    bbox_to_anchor=(0.53, 1.0),
)

plt.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.07)
out_path = os.path.join(DRAFT_DIR, "preference_vs_harm_grouped.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Plot saved to: {out_path}")
plt.close()
