#!/usr/bin/env python3
"""Generate seats and rank-choice figures from golden_votesim results.

Reads the 7-model divisive-12 issue-mode results and produces:
  - seats_by_issue.png
  - seats_by_model.png
  - seats_by_system.png
  - new_rank_choice_dist.png
"""

import json
import os
from collections import Counter, defaultdict

import matplotlib.pyplot as plt
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

# The 7 model files (matching experiments.tex).
MODEL_FILES = {
    "Llama-3.3-70B": "divisive-12.issue.llama-3.3-70b-instruct.json",
    "Llama-3.1-8B": "divisive-12.issue.llama-3.1-8b-instruct.json",
    "Mistral-Med-3": "divisive-12.issue.mistral-medium-3.json",
    "Mistral-Sm-24B": "divisive-12.issue.mistral-small-3.1-24b-instruct.json",
    "Gemma-4-31B": "divisive-12.issue.gemma-4-31b-it.json",
    "Hermes-3-70B": "divisive-12.issue.hermes-3-llama-3.1-70b.json",
    "Grok-4.1-Fast": "divisive-12.issue.grok-4.1-fast.json",
}

# Party display names and colors (6-party spectrum).
PARTY_ORDER = [
    "left", "social-democrat", "green", "liberal", "conservative", "right-populist",
]
PARTY_DISPLAY = {
    "left": "Left",
    "social-democrat": "Social Democrat",
    "green": "Green",
    "liberal": "Liberal",
    "conservative": "Conservative",
    "right-populist": "Right Populist",
}
PARTY_COLORS = {
    "left": "#B22222",
    "social-democrat": "#E25822",
    "green": "#2E8B57",
    "liberal": "#4169E1",
    "conservative": "#6A0DAD",
    "right-populist": "#8B6914",
}

# Voting systems in display order.
SYSTEM_ORDER = [
    "fptp", "sntv", "alternative_vote", "trs",
    "dhondt", "stv", "sainte_lague",
]
SYSTEM_DISPLAY = {
    "fptp": "FPTP",
    "sntv": "PBV",
    "alternative_vote": "Alt. Vote",
    "trs": "Two-Round",
    "dhondt": "D'Hondt",
    "stv": "STV",
    "sainte_lague": "Sainte-Laguë",
}

# Short issue labels (same order as divisive-12).
ISSUE_SHORT = [
    "UBI", "Abortion", "Firearms", "Immigration",
    "Aff. Action", "Death Penalty", "Religious Exemp.",
    "Law Enforce.", "Healthcare", "Climate",
    "Nat. Service", "Trans Athletes",
]

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_data():
    """Load all 7 model result files and return {model_name: data_dict}."""
    all_data = {}
    for model_name, fname in MODEL_FILES.items():
        path = os.path.join(RESULTS_DIR, fname)
        with open(path) as f:
            all_data[model_name] = json.load(f)
    return all_data


# ---------------------------------------------------------------------------
# Seats plotting helpers
# ---------------------------------------------------------------------------

def _stacked_bar(ax, labels, party_pcts, title, xlabel):
    """Draw a stacked bar chart of party seat-share percentages."""
    x = np.arange(len(labels))
    bar_width = 0.55

    bottom = np.zeros(len(labels))
    for party in PARTY_ORDER:
        vals = party_pcts.get(party, np.zeros(len(labels)))
        color = PARTY_COLORS.get(party, "#888888")
        label = PARTY_DISPLAY.get(party, party)
        ax.bar(
            x, vals, bar_width,
            bottom=bottom, color=color, label=label,
            edgecolor="white", linewidth=0.5,
        )
        # Add percentage labels inside bars (only if >= 4%).
        for k in range(len(labels)):
            val = vals[k]
            if val >= 4:
                ax.text(
                    x[k], bottom[k] + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    color="white", fontsize=8, fontweight="bold",
                )
        bottom += vals

    ax.set_xlabel(xlabel, fontsize=13)
    ax.set_ylabel("Average Seat Share (%)", fontsize=13)
    ax.set_title(title, fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10, rotation=30, ha="right")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _compute_seat_pcts(seat_allocations):
    """Given a list of {party: seats} dicts, compute avg party % across them."""
    party_pcts = defaultdict(list)
    for alloc in seat_allocations:
        total = sum(alloc.values())
        if total == 0:
            continue
        for party in PARTY_ORDER:
            party_pcts[party].append(alloc.get(party, 0) / total * 100)
    return {p: np.mean(v) for p, v in party_pcts.items()}


# ---------------------------------------------------------------------------
# Plot: seats_by_issue
# ---------------------------------------------------------------------------

def plot_seats_by_issue(all_data):
    """Stacked bar: average seat share by party per issue, averaged across
    all models & voting systems."""
    # Collect issue keys in order from the first model (they're all the same).
    first_model_data = next(iter(all_data.values()))
    issue_keys = list(first_model_data.keys())

    party_pcts = {p: [] for p in PARTY_ORDER}  # party -> list of 12 values

    for issue_key in issue_keys:
        # Collect all seat allocations for this issue across models & systems.
        allocations = []
        for model_name, model_data in all_data.items():
            issue = model_data.get(issue_key, {})
            gov = issue.get("government", {})
            for sys_name in SYSTEM_ORDER:
                sys_gov = gov.get(sys_name, {})
                seat_alloc = sys_gov.get("seat_allocation", {})
                if seat_alloc:
                    allocations.append(seat_alloc)
        pcts = _compute_seat_pcts(allocations)
        for party in PARTY_ORDER:
            party_pcts[party].append(pcts.get(party, 0))

    # Convert to numpy arrays.
    for p in PARTY_ORDER:
        party_pcts[p] = np.array(party_pcts[p])

    fig, ax = plt.subplots(figsize=(16, 7))
    _stacked_bar(
        ax, ISSUE_SHORT, party_pcts,
        "Average Seat Allocation by Party per Issue\n"
        "(Averaged Across All Models & Voting Systems)",
        "Issue",
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=6, fontsize=10, frameon=False,
    )
    plt.tight_layout()
    out = os.path.join(DRAFT_DIR, "seats_by_issue.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot: seats_by_model
# ---------------------------------------------------------------------------

def plot_seats_by_model(all_data):
    """Stacked bar: average seat share by party per model, averaged across
    all issues & voting systems."""
    model_names = list(MODEL_FILES.keys())
    party_pcts = {p: [] for p in PARTY_ORDER}

    for model_name in model_names:
        model_data = all_data[model_name]
        allocations = []
        for issue_key, issue in model_data.items():
            gov = issue.get("government", {})
            for sys_name in SYSTEM_ORDER:
                sys_gov = gov.get(sys_name, {})
                seat_alloc = sys_gov.get("seat_allocation", {})
                if seat_alloc:
                    allocations.append(seat_alloc)
        pcts = _compute_seat_pcts(allocations)
        for party in PARTY_ORDER:
            party_pcts[party].append(pcts.get(party, 0))

    for p in PARTY_ORDER:
        party_pcts[p] = np.array(party_pcts[p])

    fig, ax = plt.subplots(figsize=(14, 7))
    _stacked_bar(
        ax, model_names, party_pcts,
        "Average Seat Allocation by Party per Model\n"
        "(Averaged Across All Issues & Voting Systems)",
        "Model",
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=6, fontsize=10, frameon=False,
    )
    plt.tight_layout()
    out = os.path.join(DRAFT_DIR, "seats_by_model.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot: seats_by_system
# ---------------------------------------------------------------------------

def plot_seats_by_system(all_data):
    """Stacked bar: average seat share by party per voting system, averaged
    across all models & issues."""
    sys_labels = [SYSTEM_DISPLAY[s] for s in SYSTEM_ORDER]
    party_pcts = {p: [] for p in PARTY_ORDER}

    for sys_name in SYSTEM_ORDER:
        allocations = []
        for model_name, model_data in all_data.items():
            for issue_key, issue in model_data.items():
                gov = issue.get("government", {})
                sys_gov = gov.get(sys_name, {})
                seat_alloc = sys_gov.get("seat_allocation", {})
                if seat_alloc:
                    allocations.append(seat_alloc)
        pcts = _compute_seat_pcts(allocations)
        for party in PARTY_ORDER:
            party_pcts[party].append(pcts.get(party, 0))

    for p in PARTY_ORDER:
        party_pcts[p] = np.array(party_pcts[p])

    fig, ax = plt.subplots(figsize=(14, 7))
    _stacked_bar(
        ax, sys_labels, party_pcts,
        "Average Seat Allocation by Party per Voting System\n"
        "(Averaged Across All Models & Issues)",
        "Voting System",
    )
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.12),
        ncol=6, fontsize=10, frameon=False,
    )
    plt.tight_layout()
    out = os.path.join(DRAFT_DIR, "seats_by_system.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot: rank_choice_dist (new_rank_choice_dist)
# ---------------------------------------------------------------------------

def plot_rank_choice_dist(all_data):
    """Stacked bar: voter ranked-choice distribution by party, averaged across
    all models & issues."""
    rank_counts = {}  # {rank_position: Counter(party)}

    for model_name, model_data in all_data.items():
        for issue_key, issue in model_data.items():
            ballots = issue.get("ballots", {})
            for voter_id, ballot in ballots.items():
                ranking = ballot.get("ranking", [])
                for i, party in enumerate(ranking):
                    if i not in rank_counts:
                        rank_counts[i] = Counter()
                    rank_counts[i][party] += 1

    if not rank_counts:
        print("No ballot data found!")
        return

    num_ranks = max(rank_counts.keys()) + 1
    # Only plot up to 6 ranks (6 parties).
    num_ranks = min(num_ranks, 6)

    # Build percentage matrix: rows = parties, columns = rank positions.
    pct_matrix = np.zeros((len(PARTY_ORDER), num_ranks))
    for rank in range(num_ranks):
        total = sum(rank_counts.get(rank, Counter()).values())
        if total == 0:
            continue
        for j, party in enumerate(PARTY_ORDER):
            pct_matrix[j, rank] = (
                rank_counts.get(rank, Counter()).get(party, 0) / total * 100
            )

    rank_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th"][:num_ranks]
    x = np.arange(num_ranks)
    bar_width = 0.55

    fig, ax = plt.subplots(figsize=(12, 7))

    bottom = np.zeros(num_ranks)
    for j, party in enumerate(PARTY_ORDER):
        color = PARTY_COLORS.get(party, "#888888")
        label = PARTY_DISPLAY.get(party, party)
        bars = ax.bar(
            x, pct_matrix[j], bar_width,
            bottom=bottom, color=color, label=label,
            edgecolor="white", linewidth=0.5,
        )
        # Add percentage labels inside bars (only if >= 5%).
        for k in range(num_ranks):
            val = pct_matrix[j, k]
            if val >= 5:
                ax.text(
                    x[k], bottom[k] + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold",
                )
        bottom += pct_matrix[j]

    ax.set_xlabel("Rank Position", fontsize=13)
    ax.set_ylabel("Party Share (%)", fontsize=13)
    ax.set_title(
        "Voter Ranked Choice Distribution by Party\n"
        "(Averaged Across All Models & Issues — 6 Parties)",
        fontsize=14, fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels(rank_labels, fontsize=12)
    ax.set_ylim(0, 100)
    ax.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=6, fontsize=10, frameon=False,
    )
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()
    out = os.path.join(DRAFT_DIR, "new_rank_choice_dist.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()

    # Also save as rank_choice_dist.png for compatibility.
    out2 = os.path.join(DRAFT_DIR, "rank_choice_dist.png")
    fig2, ax2 = plt.subplots(figsize=(12, 7))
    bottom2 = np.zeros(num_ranks)
    for j, party in enumerate(PARTY_ORDER):
        color = PARTY_COLORS.get(party, "#888888")
        label = PARTY_DISPLAY.get(party, party)
        ax2.bar(
            x, pct_matrix[j], bar_width,
            bottom=bottom2, color=color, label=label,
            edgecolor="white", linewidth=0.5,
        )
        for k in range(num_ranks):
            val = pct_matrix[j, k]
            if val >= 5:
                ax2.text(
                    x[k], bottom2[k] + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    color="white", fontsize=9, fontweight="bold",
                )
        bottom2 += pct_matrix[j]
    ax2.set_xlabel("Rank Position", fontsize=13)
    ax2.set_ylabel("Party Share (%)", fontsize=13)
    ax2.set_title(
        "Voter Ranked Choice Distribution by Party\n"
        "(Averaged Across All Models & Issues — 6 Parties)",
        fontsize=14, fontweight="bold",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(rank_labels, fontsize=12)
    ax2.set_ylim(0, 100)
    ax2.legend(
        loc="upper center", bbox_to_anchor=(0.5, -0.08),
        ncol=6, fontsize=10, frameon=False,
    )
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out2, dpi=150, bbox_inches="tight")
    print(f"Saved {out2}")
    plt.close()


# ---------------------------------------------------------------------------
# Plot: combined 2x2 figure
# ---------------------------------------------------------------------------

def _compute_rank_choice_data(all_data):
    """Compute rank-choice percentage matrix from all_data."""
    rank_counts = {}
    for model_name, model_data in all_data.items():
        for issue_key, issue in model_data.items():
            ballots = issue.get("ballots", {})
            for voter_id, ballot in ballots.items():
                ranking = ballot.get("ranking", [])
                for i, party in enumerate(ranking):
                    if i not in rank_counts:
                        rank_counts[i] = Counter()
                    rank_counts[i][party] += 1

    num_ranks = min(max(rank_counts.keys()) + 1, 6)
    pct_matrix = np.zeros((len(PARTY_ORDER), num_ranks))
    for rank in range(num_ranks):
        total = sum(rank_counts.get(rank, Counter()).values())
        if total == 0:
            continue
        for j, party in enumerate(PARTY_ORDER):
            pct_matrix[j, rank] = (
                rank_counts.get(rank, Counter()).get(party, 0) / total * 100
            )
    return pct_matrix, num_ranks


def _compute_seats_by_issue_data(all_data):
    """Compute per-issue party seat share percentages."""
    first_model_data = next(iter(all_data.values()))
    issue_keys = list(first_model_data.keys())

    party_pcts = {p: [] for p in PARTY_ORDER}
    for issue_key in issue_keys:
        allocations = []
        for model_name, model_data in all_data.items():
            issue = model_data.get(issue_key, {})
            gov = issue.get("government", {})
            for sys_name in SYSTEM_ORDER:
                sys_gov = gov.get(sys_name, {})
                seat_alloc = sys_gov.get("seat_allocation", {})
                if seat_alloc:
                    allocations.append(seat_alloc)
        pcts = _compute_seat_pcts(allocations)
        for party in PARTY_ORDER:
            party_pcts[party].append(pcts.get(party, 0))

    for p in PARTY_ORDER:
        party_pcts[p] = np.array(party_pcts[p])
    return party_pcts


def _compute_seats_by_model_data(all_data):
    """Compute per-model party seat share percentages."""
    model_names = list(MODEL_FILES.keys())
    party_pcts = {p: [] for p in PARTY_ORDER}

    for model_name in model_names:
        model_data = all_data[model_name]
        allocations = []
        for issue_key, issue in model_data.items():
            gov = issue.get("government", {})
            for sys_name in SYSTEM_ORDER:
                sys_gov = gov.get(sys_name, {})
                seat_alloc = sys_gov.get("seat_allocation", {})
                if seat_alloc:
                    allocations.append(seat_alloc)
        pcts = _compute_seat_pcts(allocations)
        for party in PARTY_ORDER:
            party_pcts[party].append(pcts.get(party, 0))

    for p in PARTY_ORDER:
        party_pcts[p] = np.array(party_pcts[p])
    return model_names, party_pcts


def _compute_seats_by_system_data(all_data):
    """Compute per-system party seat share percentages."""
    sys_labels = [SYSTEM_DISPLAY[s] for s in SYSTEM_ORDER]
    party_pcts = {p: [] for p in PARTY_ORDER}

    for sys_name in SYSTEM_ORDER:
        allocations = []
        for model_name, model_data in all_data.items():
            for issue_key, issue in model_data.items():
                gov = issue.get("government", {})
                sys_gov = gov.get(sys_name, {})
                seat_alloc = sys_gov.get("seat_allocation", {})
                if seat_alloc:
                    allocations.append(seat_alloc)
        pcts = _compute_seat_pcts(allocations)
        for party in PARTY_ORDER:
            party_pcts[party].append(pcts.get(party, 0))

    for p in PARTY_ORDER:
        party_pcts[p] = np.array(party_pcts[p])
    return sys_labels, party_pcts


def _draw_stacked_subplot(ax, labels, party_pcts, subtitle, xlabel,
                          bar_width, pct_fontsize=7, min_pct=5,
                          x_rotation=30, x_fontsize=9):
    """Draw a stacked bar chart on a subplot axis (black outlines, no legend)."""
    x = np.arange(len(labels))
    bottom = np.zeros(len(labels))

    for party in PARTY_ORDER:
        vals = party_pcts.get(party, np.zeros(len(labels)))
        color = PARTY_COLORS.get(party, "#888888")
        ax.bar(
            x, vals, bar_width,
            bottom=bottom, color=color,
            edgecolor="black", linewidth=0.6,
        )
        # Percentage labels inside bars.
        for k in range(len(labels)):
            val = vals[k]
            if val >= min_pct:
                ax.text(
                    x[k], bottom[k] + val / 2,
                    f"{val:.0f}%",
                    ha="center", va="center",
                    color="white", fontsize=pct_fontsize, fontweight="bold",
                )
        bottom += vals

    ax.set_xlabel(xlabel, fontsize=10, fontweight="bold", labelpad=4)
    ax.set_ylabel("Share (%)", fontsize=10, fontweight="bold", labelpad=4)
    ax.set_title(subtitle, fontsize=11, fontweight="bold", pad=6)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=x_fontsize, rotation=x_rotation,
                       ha="right" if x_rotation else "center")
    ax.set_ylim(0, 100)
    ax.tick_params(axis="y", labelsize=9)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def plot_combined_2x2(all_data):
    """Create a combined 2x2 figure with a single shared legend at the top.

    Layout:
        Top-Left:     Rank Choice Distribution
        Top-Right:    Seats by Issue
        Bottom-Left:  Seats by Model
        Bottom-Right: Seats by System
    """
    # ── Compute all data ────────────────────────────────────────────
    pct_matrix, num_ranks = _compute_rank_choice_data(all_data)
    rank_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th"][:num_ranks]

    issue_pcts = _compute_seats_by_issue_data(all_data)
    model_names, model_pcts = _compute_seats_by_model_data(all_data)
    sys_labels, sys_pcts = _compute_seats_by_system_data(all_data)

    # ── Create figure with GridSpec for control ─────────────────────
    fig = plt.figure(figsize=(18, 11))
    gs = fig.add_gridspec(
        2, 2,
        width_ratios=[1, 2],       # top-right is wider (12 issues)
        hspace=0.35, wspace=0.22,
        left=0.06, right=0.97,
        top=0.88, bottom=0.08,
    )

    # Uniform bar_width: use 0.6 everywhere for consistent bar height
    bar_w = 0.6

    # ── Top-Left: Rank Choice ───────────────────────────────────────
    ax_rank = fig.add_subplot(gs[0, 0])
    rank_party_pcts = {
        party: pct_matrix[j] for j, party in enumerate(PARTY_ORDER)
    }
    _draw_stacked_subplot(
        ax_rank, rank_labels, rank_party_pcts,
        "(a) Ranked-Choice Distribution",
        "Rank Position",
        bar_w, pct_fontsize=7.5, min_pct=5,
        x_rotation=0, x_fontsize=10,
    )

    # ── Top-Right: Seats by Issue ───────────────────────────────────
    ax_issue = fig.add_subplot(gs[0, 1])
    _draw_stacked_subplot(
        ax_issue, ISSUE_SHORT, issue_pcts,
        "(b) Seat Allocation by Issue",
        "Issue",
        bar_w, pct_fontsize=6.5, min_pct=5,
        x_rotation=35, x_fontsize=9,
    )

    # ── Bottom-Left: Seats by Model ─────────────────────────────────
    ax_model = fig.add_subplot(gs[1, 0])
    _draw_stacked_subplot(
        ax_model, model_names, model_pcts,
        "(c) Seat Allocation by Model",
        "Model",
        bar_w, pct_fontsize=7, min_pct=4,
        x_rotation=35, x_fontsize=9,
    )

    # ── Bottom-Right: Seats by System ───────────────────────────────
    ax_sys = fig.add_subplot(gs[1, 1])
    _draw_stacked_subplot(
        ax_sys, sys_labels, sys_pcts,
        "(d) Seat Allocation by Voting System",
        "Voting System",
        bar_w, pct_fontsize=7.5, min_pct=4,
        x_rotation=30, x_fontsize=10,
    )

    # ── Shared legend at top of figure ──────────────────────────────
    from matplotlib.patches import Patch
    legend_handles = [
        Patch(facecolor=PARTY_COLORS[p], edgecolor="black", linewidth=0.6,
              label=PARTY_DISPLAY[p])
        for p in PARTY_ORDER
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=6, fontsize=11, frameon=False,
        handlelength=1.8, handleheight=1.2, handletextpad=0.5,
        columnspacing=1.5,
    )

    out = os.path.join(DRAFT_DIR, "seats_rank_combined.png")
    plt.savefig(out, dpi=180, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    all_data = load_all_data()
    print(f"Loaded {len(all_data)} models:")
    for name in all_data:
        n_issues = len(all_data[name])
        print(f"  {name}: {n_issues} issues")

    plot_seats_by_issue(all_data)
    plot_seats_by_model(all_data)
    plot_seats_by_system(all_data)
    plot_rank_choice_dist(all_data)
    plot_combined_2x2(all_data)
    print("\nAll figures generated!")
