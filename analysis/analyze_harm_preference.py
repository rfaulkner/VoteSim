#!/usr/bin/env python3
"""Analyze Llama-70B judge results: Cohen's kappa and preference vs harm.

Groups voting systems into:
  - Baselines:    baseline, baseline_informed
  - Majoritarian: fptp, alternative_vote, trs, sntv
  - Proportional: dhondt, sainte_lague, stv
"""

import json
import os
import numpy as np
from collections import defaultdict
from scipy import stats
from scipy.spatial import ConvexHull

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_data")

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

GROUP_COLORS = {
    "Baseline": "#888888",
    "Majoritarian": "#e41a1c",
    "Proportional": "#377eb8",
}

GROUP_FILL = {
    "Baseline": "#cccccc",
    "Majoritarian": "#fbb4ae",
    "Proportional": "#b3cde3",
}

# Marker per system (so individual systems are still distinguishable)
SYSTEM_MARKERS = {
    "baseline": "s",
    "baseline_informed": "D",
    "fptp": "o",
    "alternative_vote": "^",
    "trs": "v",
    "sntv": ">",
    "dhondt": "o",
    "sainte_lague": "^",
    "stv": "v",
}

SYSTEM_LABELS = {
    "alternative_vote": "IRV",
    "baseline": "Base",
    "baseline_informed": "Oracle",
    "dhondt": "D'Hondt",
    "fptp": "FPTP",
    "sainte_lague": "S-L",
    "sntv": "SNTV",
    "stv": "STV",
    "trs": "TRS",
}

# ── Model configurations ─────────────────────────────────────────────
MODELS = {
    "mistral": {
        "slug": "mistral-medium-3",
        "title": "Mistral-Medium",
        "judge_files": {
            "claude-haiku": os.path.join(DATA_DIR, "harm-12.judge.mistral-medium-3.judged_by.claude-3.5-haiku.json"),
            "gemini-flash": os.path.join(DATA_DIR, "harm-12.judge.mistral-medium-3.judged_by.gemini-3-flash-preview.json"),
            "gpt-4o": os.path.join(DATA_DIR, "harm-12.judge.mistral-medium-3.judged_by.gpt-4o-2024-11-20.json"),
        },
        "issue_file": os.path.join(DATA_DIR, "harm-12.issue.mistral-medium-3.json"),
    },
    "llama": {
        "slug": "llama-3.3-70b-instruct",
        "title": "Llama-3.3-70B",
        "judge_files": {
            "claude-opus": os.path.join(DATA_DIR, "harm-12.judge.llama-3.3-70b-instruct.judged_by.claude-opus-4.7.json"),
            "gemini-flash": os.path.join(DATA_DIR, "harm-12.judge.llama-3.3-70b-instruct.judged_by.gemini-3-flash-preview.json"),
            "gpt-4o": os.path.join(DATA_DIR, "harm-12.judge.llama-3.3-70b-instruct.judged_by.gpt-4o-2024-11-20.json"),
        },
        "issue_file": os.path.join(DATA_DIR, "harm-12.issue.llama-3.3-70b-instruct.json"),
    },
}

# =====================================================================
# Helper functions
# =====================================================================
def get_pairwise_decisions(judge_data):
    decisions = {}
    for issue, issue_d in judge_data.items():
        for pair_key, pair_val in issue_d["pairwise"].items():
            decisions[(issue, pair_key)] = pair_val["winner"]
    return decisions


def cohens_kappa(da, db):
    common = sorted(set(da) & set(db))
    if not common:
        return float("nan"), 0
    la = [da[k] for k in common]
    lb = [db[k] for k in common]
    cats = sorted(set(la) | set(lb))
    ci = {c: i for i, c in enumerate(cats)}
    n, k = len(common), len(cats)
    cm = np.zeros((k, k), dtype=int)
    for a, b in zip(la, lb):
        cm[ci[a], ci[b]] += 1
    po = np.trace(cm) / n
    pe = np.sum((cm.sum(1) / n) * (cm.sum(0) / n))
    return (po - pe) / (1 - pe) if pe < 1 else 1.0, n


def build_records(issue_data, judges):
    """Build (issue, system) records with group labels."""
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
        gpt_data = judges["gpt-4o"]
        if issue in gpt_data:
            for s, h in gpt_data[issue]["aggregate"]["mean_harm_score"].items():
                harm_per_sys[s] = h

        short = issue[:50] + "…" if len(issue) > 50 else issue
        for s in systems:
            if s in harm_per_sys:
                records.append(dict(
                    issue=short, system=s,
                    group=SYSTEM_TO_GROUP[s],
                    preference=mean_pref[s],
                    harm=harm_per_sys[s],
                ))
    return records


# =====================================================================
# Plot styling
# =====================================================================
from matplotlib.patches import Ellipse
import matplotlib.transforms as transforms

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
        width=width, height=height, angle=angle, **kwargs
    )
    ax.add_patch(ellipse)


def plot_preference_vs_harm(records, model_title, out_path):
    """Generate and save the preference-vs-harm scatter plot."""
    prefs = np.array([r["preference"] for r in records])
    harms = np.array([r["harm"] for r in records])
    rp, pp = stats.pearsonr(prefs, harms)

    fig, ax = plt.subplots(figsize=(9, 6.5))

    # ── Confidence ellipses ──────────────────────────────────────────
    for g in ["Baseline", "Majoritarian", "Proportional"]:
        gx = np.array([r["preference"] for r in records if r["group"] == g])
        gy = np.array([r["harm"] for r in records if r["group"] == g])
        confidence_ellipse(
            gx, gy, ax, n_std=1.8,
            facecolor=PLOT_FILL[g], edgecolor=PLOT_COLORS[g],
            alpha=0.22, linewidth=1.5, linestyle="-", zorder=1,
        )

    # ── Individual data points ───────────────────────────────────────
    for r in records:
        ax.scatter(
            r["preference"], r["harm"],
            c=PLOT_COLORS[r["group"]],
            marker=SYSTEM_MARKERS[r["system"]],
            s=50, alpha=0.7,
            edgecolors="white", linewidth=0.5, zorder=3,
        )

    # ── Centroids + labels (placed OUTSIDE ellipses) ─────────────────
    # Large offsets push labels well outside the ellipses; an arrow
    # connects each label back to its centroid so the association is
    # unambiguous even without proximity.
    label_offsets = {
        "Baseline":     (-65, 50),
        "Majoritarian": (-55, -50),
        "Proportional": (55, 50),
    }

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
            fontsize=9, fontweight="bold", color=PLOT_COLORS[g],
            ha="center", va="center",
            bbox=dict(boxstyle="round,pad=0.25", fc="white",
                      ec=PLOT_COLORS[g], alpha=0.92, linewidth=0.8),
            arrowprops=dict(
                arrowstyle="-|>",
                color=PLOT_COLORS[g],
                lw=1.0,
                connectionstyle="arc3,rad=0.15",
            ),
            zorder=6,
        )

    # ── OLS trend line ───────────────────────────────────────────────
    sl, ic, _, _, _ = stats.linregress(prefs, harms)
    xl = np.linspace(prefs.min() - 0.05, prefs.max() + 0.05, 100)
    ax.plot(xl, sl * xl + ic, color="#333333", linestyle="--",
            alpha=0.5, linewidth=1.2, zorder=2)

    # ── Correlation annotation ───────────────────────────────────────
    ax.annotate(
        f"r = {rp:.2f},  p = {pp:.1e}",
        xy=(0.97, 0.97), xycoords="axes fraction",
        ha="right", va="top", fontsize=9.5, fontstyle="italic",
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.3", fc="white",
                  ec="#cccccc", alpha=0.9),
    )

    # ── Legend ────────────────────────────────────────────────────────
    sys_handles = []
    for g in ["Baseline", "Majoritarian", "Proportional"]:
        for s in GROUPS[g]:
            sys_handles.append(
                Line2D(
                    [0], [0], marker=SYSTEM_MARKERS[s], color="w",
                    markerfacecolor=PLOT_COLORS[g], markersize=7,
                    markeredgecolor="#555555", markeredgewidth=0.5,
                    label=SYSTEM_LABELS[s],
                )
            )
    sys_handles.append(
        Line2D([0], [0], linestyle="--", color="#333333", alpha=0.5,
               label="OLS fit")
    )

    leg = ax.legend(
        handles=sys_handles, loc="upper left", fontsize=8,
        frameon=True, framealpha=0.92, edgecolor="#cccccc",
        ncol=1, handletextpad=0.5, borderpad=0.6,
        labelspacing=0.45,
    )
    leg.get_frame().set_linewidth(0.6)

    ax.set_xlabel("Mean Preference Score (Likert)", fontsize=12, labelpad=8)
    ax.set_ylabel("Mean Harm Score (GPT-4o judge)", fontsize=12, labelpad=8)
    ax.set_title(
        f"Preference vs Harm by System Category \u2014 {model_title}",
        fontsize=13, fontweight="bold", pad=12,
    )
    ax.grid(True, alpha=0.15, linewidth=0.5, color="#888888")
    ax.tick_params(labelsize=10)

    plt.tight_layout()
    plt.savefig(out_path, dpi=180, bbox_inches="tight")
    print(f"\n  Plot saved to: {out_path}")
    plt.close()

    return rp, pp


# =====================================================================
# Main: iterate over each model
# =====================================================================
for model_key, mcfg in MODELS.items():
    print("\n" + "#" * 70)
    print(f"# MODEL: {mcfg['title']}")
    print("#" * 70)

    # ── Load data ────────────────────────────────────────────────────
    judges = {}
    for name, path in mcfg["judge_files"].items():
        with open(path) as f:
            judges[name] = json.load(f)

    with open(mcfg["issue_file"]) as f:
        issue_data = json.load(f)

    # ── Cohen's Kappa ────────────────────────────────────────────────
    print("=" * 70)
    print(f"Cohen's Kappa: Inter-Judge Agreement — {mcfg['title']}")
    print("=" * 70)

    judge_names = list(judges.keys())
    all_decisions = {n: get_pairwise_decisions(d) for n, d in judges.items()}

    for i in range(len(judge_names)):
        for j in range(i + 1, len(judge_names)):
            na, nb = judge_names[i], judge_names[j]
            k, n = cohens_kappa(all_decisions[na], all_decisions[nb])
            print(f"  {na:20s} vs {nb:20s}:  κ = {k:.4f}  (n={n})")

    # Fleiss' kappa
    common_keys = sorted(set.intersection(*[set(d.keys()) for d in all_decisions.values()]))
    all_cats = sorted(set(v for d in all_decisions.values() for v in d.values()))
    ci = {c: i for i, c in enumerate(all_cats)}
    ratings = np.zeros((len(common_keys), len(all_cats)), dtype=int)
    for idx, key in enumerate(common_keys):
        for nm in judge_names:
            ratings[idx, ci[all_decisions[nm][key]]] += 1
    nr = len(judge_names)
    Pi = (np.sum(ratings**2, axis=1) - nr) / (nr * (nr - 1))
    Pbar = np.mean(Pi)
    pj = np.sum(ratings, axis=0) / (len(common_keys) * nr)
    Pe = np.sum(pj**2)
    fk = (Pbar - Pe) / (1 - Pe) if Pe < 1 else 1.0
    print(f"\n  Fleiss' κ = {fk:.4f}  (n={len(common_keys)}, {nr} raters)")
    agree_all = np.sum(np.max(ratings, 1) == nr)
    agree_maj = np.sum(np.max(ratings, 1) >= 2)
    print(f"  Full agreement: {agree_all}/{len(common_keys)} = {agree_all/len(common_keys):.1%}")
    print(f"  Majority (≥2/3): {agree_maj}/{len(common_keys)} = {agree_maj/len(common_keys):.1%}")

    # ── Build records ────────────────────────────────────────────────
    records = build_records(issue_data, judges)

    prefs = np.array([r["preference"] for r in records])
    harms = np.array([r["harm"] for r in records])

    rp, pp = stats.pearsonr(prefs, harms)
    rs, ps = stats.spearmanr(prefs, harms)
    print(f"\n  Overall  Pearson  r = {rp:.4f}  p = {pp:.4e}")
    print(f"  Overall  Spearman ρ = {rs:.4f}  p = {ps:.4e}")

    # ── Per-group statistics ─────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Group-Level Statistics")
    print("=" * 70)
    for g in ["Baseline", "Majoritarian", "Proportional"]:
        gp = np.array([r["preference"] for r in records if r["group"] == g])
        gh = np.array([r["harm"] for r in records if r["group"] == g])
        rg, pg = stats.pearsonr(gp, gh) if len(gp) > 2 else (float("nan"), float("nan"))
        print(f"\n  {g} (n={len(gp)})")
        print(f"    Preference: {gp.mean():.3f} ± {gp.std():.3f}")
        print(f"    Harm:       {gh.mean():.3f} ± {gh.std():.3f}")
        print(f"    Pearson r = {rg:.4f}, p = {pg:.4e}")

    # ── Kruskal-Wallis across groups for harm ────────────────────────
    gh_baseline = [r["harm"] for r in records if r["group"] == "Baseline"]
    gh_major = [r["harm"] for r in records if r["group"] == "Majoritarian"]
    gh_prop = [r["harm"] for r in records if r["group"] == "Proportional"]
    kw_stat, kw_p = stats.kruskal(gh_baseline, gh_major, gh_prop)
    print(f"\n  Kruskal-Wallis (harm across groups): H = {kw_stat:.3f}, p = {kw_p:.4e}")

    # Pairwise Mann-Whitney
    for (na, a), (nb, b) in [
        (("Baseline", gh_baseline), ("Majoritarian", gh_major)),
        (("Baseline", gh_baseline), ("Proportional", gh_prop)),
        (("Majoritarian", gh_major), ("Proportional", gh_prop)),
    ]:
        u, p = stats.mannwhitneyu(a, b, alternative="two-sided")
        print(f"    {na:15s} vs {nb:15s}: U = {u:.0f}, p = {p:.4e}")

    # ── Plot ─────────────────────────────────────────────────────────
    out_path = os.path.join(DATA_DIR, f"preference_vs_harm_grouped_{model_key}.png")
    plot_preference_vs_harm(records, mcfg["title"], out_path)

    # ── Summary table by group ───────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary by Group")
    print("=" * 70)
    print(f"  {'Group':<18s} {'Preference':>12s} {'Harm':>12s}  {'n':>4s}")
    print(f"  {'-'*18} {'-'*12} {'-'*12}  {'-'*4}")
    for g in ["Baseline", "Majoritarian", "Proportional"]:
        gp = [r["preference"] for r in records if r["group"] == g]
        gh = [r["harm"] for r in records if r["group"] == g]
        print(f"  {g:<18s} {np.mean(gp):>12.3f} {np.mean(gh):>12.3f}  {len(gp):>4d}")
