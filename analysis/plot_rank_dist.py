#!/usr/bin/env python3
"""Generate a stacked-bar rank choice distribution plot from survey data."""

import json
import os
from collections import Counter

import matplotlib.pyplot as plt
import numpy as np

# --- Load all survey files ---------------------------------------------------
SURVEY_DIR = "/tmp/survey_data"
survey_files = sorted(f for f in os.listdir(SURVEY_DIR) if f.endswith(".json"))

# Accumulate rank counts across all models and issues.
rank_counts = {}  # {rank_position: Counter(party)}
for fname in survey_files:
    with open(os.path.join(SURVEY_DIR, fname)) as f:
        data = json.load(f)
    issues = data.get("issues", {})
    for issue_key, issue_data in issues.items():
        ballots = issue_data.get("ballots", {})
        for voter_id, ballot in ballots.items():
            ranking = ballot["ranking"]
            for i, party in enumerate(ranking):
                if i not in rank_counts:
                    rank_counts[i] = Counter()
                rank_counts[i][party] += 1

# --- Compute percentages -----------------------------------------------------
num_ranks = max(rank_counts.keys()) + 1
all_parties = sorted(
    set(p for c in rank_counts.values() for p in c.keys())
)

# Party display names and colors (6-party spectrum).
PARTY_DISPLAY = {
    "left": "Left",
    "green": "Green",
    "social-democrat": "Soc-Dem",
    "liberal": "Liberal",
    "conservative": "Conservative",
    "right-populist": "Populist",
}
PARTY_COLORS = {
    "left": "#B22222",       # dark red
    "green": "#2E8B57",      # green
    "social-democrat": "#E25822",  # burnt orange
    "liberal": "#4169E1",    # royal blue
    "conservative": "#1E3A5F",  # dark navy
    "right-populist": "#DAA520",  # goldenrod
}

# Build percentage matrix: rows = parties, columns = rank positions.
pct_matrix = np.zeros((len(all_parties), num_ranks))
for rank in range(num_ranks):
    total = sum(rank_counts[rank].values())
    for j, party in enumerate(all_parties):
        pct_matrix[j, rank] = rank_counts[rank].get(party, 0) / total * 100

# --- Plot --------------------------------------------------------------------
rank_labels = ["1st", "2nd", "3rd", "4th", "5th", "6th"][:num_ranks]
x = np.arange(num_ranks)
bar_width = 0.55

fig, ax = plt.subplots(figsize=(12, 7))

bottom = np.zeros(num_ranks)
bars_by_party = {}
for j, party in enumerate(all_parties):
    color = PARTY_COLORS.get(party, "#888888")
    label = PARTY_DISPLAY.get(party, party)
    bars = ax.bar(
        x, pct_matrix[j], bar_width,
        bottom=bottom, color=color, label=label,
        edgecolor="white", linewidth=0.5,
    )
    bars_by_party[party] = bars

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

out_path = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "draft", "new_rank_choice_dist.png",
)
out_path = os.path.normpath(out_path)
plt.savefig(out_path, dpi=150, bbox_inches="tight")
print(f"Saved to {out_path}")
plt.close()
