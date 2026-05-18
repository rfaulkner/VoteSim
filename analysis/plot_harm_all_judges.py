#!/usr/bin/env python3
"""Combined preference-vs-harm scatter — all judges pooled.

Per-system ellipses with prominent centroids, category-level centroids retained.
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
    "dhondt": "p", "sainte_lague": "h", "stv": "*",
}
SYSTEM_LABELS = {
    "alternative_vote": "IRV", "baseline": "Base",
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
        "judge_files": [
            os.path.join(DATA_DIR, "harm-12.judge.llama-3.3-70b-instruct.judged_by.claude-opus-4.7.json"),
            os.path.join(DATA_DIR, "harm-12.judge.llama-3.3-70b-instruct.judged_by.gemini-3-flash-preview.json"),
            os.path.join(DATA_DIR, "harm-12.judge.llama-3.3-70b-instruct.judged_by.gpt-4o-2024-11-20.json"),
            os.path.join(DATA_DIR, "harm-12.judge.llama-3.3-70b-instruct.judged_by.claude-3.5-haiku.json"),
            os.path.join(DATA_DIR, "harm-12.judge.llama-3.3-70b-instruct.judged_by.llama-3.1-70b-instruct.json"),
        ],
        "issue_file": os.path.join(DATA_DIR, "harm-12.issue.llama-3.3-70b-instruct.json"),
    },
    {
        "key": "mistral",
        "label": "Mistral-Medium-3",
        "judge_files": [
            os.path.join(DATA_DIR, "harm-12.judge.mistral-medium-3.judged_by.claude-opus-4.7.json"),
            os.path.join(DATA_DIR, "harm-12.judge.mistral-medium-3.judged_by.gemini-3-flash-preview.json"),
            os.path.join(DATA_DIR, "harm-12.judge.mistral-medium-3.judged_by.gpt-4o-2024-11-20.json"),
            os.path.join(DATA_DIR, "harm-12.judge.mistral-medium-3.judged_by.claude-3.5-haiku.json"),
            os.path.join(DATA_DIR, "harm-12.judge.mistral-medium-3.judged_by.llama-3.1-70b-instruct.json"),
        ],
        "issue_file": os.path.join(DATA_DIR, "harm-12.issue.mistral-medium-3.json"),
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


def build_records_all_judges(issue_data, judge_files):
    """One record per (issue, system, judge)."""
    mean_prefs = {}
    for issue in issue_data:
        scores = issue_data[issue]["scores"]
        systems = list(list(scores.values())[0].keys())
        pref_per_sys = defaultdict(list)
        for voter_scores in scores.values():
            for s, sc in voter_scores.items():
                pref_per_sys[s].append(sc)
        for s in systems:
            mean_prefs[(issue, s)] = np.mean(pref_per_sys[s])

    records = []
    for jf in judge_files:
        if not os.path.exists(jf):
            continue
        with open(jf) as f:
            jdata = json.load(f)
        for issue in issue_data:
            if issue not in jdata:
                continue
            harm_scores = jdata[issue]["aggregate"]["mean_harm_score"]
            for s, h in harm_scores.items():
                if (issue, s) in mean_prefs:
                    records.append(dict(
                        system=s, group=SYSTEM_TO_GROUP[s],
                        preference=mean_prefs[(issue, s)], harm=h,
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

fig, axes = plt.subplots(
    nrows=2, ncols=1, figsize=(8, 10),
    sharex=True, sharey=True,
    gridspec_kw={"hspace": 0.08},
)

# Load all data
all_prefs, all_harms = [], []
all_records = {}
for mcfg in MODELS:
    with open(mcfg["issue_file"]) as f:
        issue_data = json.load(f)
    records = build_records_all_judges(issue_data, mcfg["judge_files"])
    all_records[mcfg["key"]] = records
    all_prefs.extend(r["preference"] for r in records)
    all_harms.extend(r["harm"] for r in records)
    n_judges = sum(1 for jf in mcfg["judge_files"] if os.path.exists(jf))
    print(f"{mcfg['label']}: {len(records)} pts "
          f"({len(records)//n_judges}/judge × {n_judges} judges)")

x_pad, y_pad = 0.15, 0.15
x_min, x_max = min(all_prefs) - x_pad, max(all_prefs) + x_pad
y_min, y_max = min(all_harms) - y_pad, max(all_harms) + y_pad

cat_label_offsets = {
    "Baseline":     (-65, 50),
    "Majoritarian": (-55, -50),
    "Proportional": (55, 50),
}

ALL_SYSTEMS = []
for g in ["Baseline", "Majoritarian", "Proportional"]:
    ALL_SYSTEMS.extend(GROUPS[g])

for idx, mcfg in enumerate(MODELS):
    ax = axes[idx]
    records = all_records[mcfg["key"]]
    prefs = np.array([r["preference"] for r in records])
    harms = np.array([r["harm"] for r in records])
    rp, pp = stats.pearsonr(prefs, harms)

    # ── Per-system ellipses ──────────────────────────────────────────
    for s in ALL_SYSTEMS:
        g = SYSTEM_TO_GROUP[s]
        sx = np.array([r["preference"] for r in records if r["system"] == s])
        sy = np.array([r["harm"] for r in records if r["system"] == s])
        if len(sx) >= 3:
            confidence_ellipse(
                sx, sy, ax, n_std=1.5,
                facecolor=PLOT_FILL[g], edgecolor=PLOT_COLORS[g],
                alpha=0.12, linewidth=0.8, linestyle="-", zorder=1,
            )

    # ── Data points (transparent) ────────────────────────────────────
    for r in records:
        ax.scatter(
            r["preference"], r["harm"],
            c=PLOT_COLORS[r["group"]],
            marker=SYSTEM_MARKERS[r["system"]],
            s=22, alpha=0.30,
            edgecolors="none", zorder=3,
        )

    # ── Per-system centroids (prominent) ─────────────────────────────
    for s in ALL_SYSTEMS:
        g = SYSTEM_TO_GROUP[s]
        sx = [r["preference"] for r in records if r["system"] == s]
        sy = [r["harm"] for r in records if r["system"] == s]
        if sx:
            ax.scatter(
                np.mean(sx), np.mean(sy),
                c=PLOT_COLORS[g],
                marker=SYSTEM_MARKERS[s],
                s=120, alpha=1.0,
                edgecolors="black", linewidth=1.0, zorder=5,
            )

    # ── Category centroids with labels ───────────────────────────────
    for g in ["Baseline", "Majoritarian", "Proportional"]:
        gx = np.mean([r["preference"] for r in records if r["group"] == g])
        gy = np.mean([r["harm"] for r in records if r["group"] == g])
        ax.scatter(
            gx, gy, c=PLOT_COLORS[g], marker="o", s=220,
            edgecolors="black", linewidth=1.8, zorder=6,
        )
        dx, dy = cat_label_offsets[g]
        ax.annotate(
            g, (gx, gy), textcoords="offset points", xytext=(dx, dy),
            fontsize=11, fontweight="bold", color=PLOT_COLORS[g],
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=PLOT_COLORS[g], alpha=0.92, linewidth=0.8),
            arrowprops=dict(arrowstyle="-|>", color=PLOT_COLORS[g],
                            lw=1.0, connectionstyle="arc3,rad=0.15"),
            zorder=7,
        )

    # ── OLS trend line ───────────────────────────────────────────────
    sl, ic, _, _, _ = stats.linregress(prefs, harms)
    xl = np.linspace(x_min, x_max, 100)
    ax.plot(xl, sl * xl + ic, color="#333333", linestyle="--",
            alpha=0.5, linewidth=1.2, zorder=2)

    # ── Correlation annotation ───────────────────────────────────────
    ax.annotate(
        f"r = {rp:.2f},  p = {pp:.1e}",
        xy=(0.97, 0.95), xycoords="axes fraction",
        ha="right", va="top", fontsize=12, fontstyle="italic",
        fontweight="bold", color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#cccccc", alpha=0.9),
    )

    # ── Source model label ───────────────────────────────────────────
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
    ax.tick_params(labelsize=13, width=1.2)

    # Bold tick labels
    for label in ax.get_xticklabels() + ax.get_yticklabels():
        label.set_fontweight("bold")

    if idx == 0:
        ax.tick_params(axis="x", labelbottom=False)

# ── Shared x-axis label (bold) ──────────────────────────────────────
axes[-1].set_xlabel(
    "Mean Preference Score (Likert)",
    fontsize=15, fontweight="bold", labelpad=10,
)

# ── Shared y-axis label (bold, centered) ────────────────────────────
fig.text(
    0.02, 0.5, "Mean Harm Score (all judges pooled)",
    va="center", ha="center", rotation="vertical",
    fontsize=15, fontweight="bold", color="#222222",
)

# ── Shared horizontal legend ────────────────────────────────────────
sys_handles = []
for g in ["Baseline", "Majoritarian", "Proportional"]:
    for s in GROUPS[g]:
        sys_handles.append(
            Line2D([0], [0], marker=SYSTEM_MARKERS[s], color="w",
                   markerfacecolor=PLOT_COLORS[g], markersize=9,
                   markeredgecolor="black", markeredgewidth=0.8,
                   label=SYSTEM_LABELS[s])
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
    prop={"weight": "bold"},
)

plt.subplots_adjust(left=0.11, right=0.97, top=0.94, bottom=0.07)
out_path = os.path.join(DRAFT_DIR, "preference_vs_harm_all_judges.png")
plt.savefig(out_path, dpi=200, bbox_inches="tight")
print(f"Plot saved to: {out_path}")
plt.close()
