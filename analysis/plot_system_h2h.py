#!/usr/bin/env python3
"""Generate a system-vs-system head-to-head win rate heatmap.

For every pair of voting systems (A, B), computes the fraction of
voter-rankings (across all issues and all models) where system A was
ranked higher than system B.  The diagonal is left blank.

Systems are grouped by family (majoritarian, proportional, baselines)
with separator lines between groups.

Saves to draft/system_h2h.pdf and draft/system_h2h.png.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "results",
    "golden",
)
DRAFT_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..",
    "draft",
)

MODEL_FILES = [
    "divisive-12.issue.llama-3.3-70b-instruct.json",
    "divisive-12.issue.llama-3.1-8b-instruct.json",
    "divisive-12.issue.mistral-medium-3.json",
    "divisive-12.issue.mistral-small-3.1-24b-instruct.json",
    "divisive-12.issue.gemma-4-31b-it.json",
    "divisive-12.issue.hermes-3-llama-3.1-70b.json",
    "divisive-12.issue.grok-4.1-fast.json",
]

# System order: majoritarian | proportional | baselines
SYSTEM_ORDER = [
    "sntv",
    "fptp",
    "alternative_vote",
    "trs",
    "dhondt",
    "stv",
    "sainte_lague",
    "baseline",
    "baseline_informed",
]
SYSTEM_DISPLAY = {
    "sntv": "SNTV",
    "fptp": "SMDP",
    "alternative_vote": "IRV",
    "trs": "Two-Round",
    "dhondt": "D'Hondt",
    "stv": "STV",
    "sainte_lague": "Sainte-Laguë",
    "baseline": "Baseline",
    "baseline_informed": "Oracle\nMediator",
}

# Group boundaries (index of first system in each group).
GROUP_BOUNDARIES = [4, 7]  # after trs, after sainte_lague


def load_all_rankings():
  """Load all voter system rankings from all models and issues.

  Returns a list of ranking lists, where each ranking is an ordered
  list of system names (best first).
  """
  all_rankings = []
  for fname in MODEL_FILES:
    path = os.path.join(RESULTS_DIR, fname)
    with open(path) as f:
      data = json.load(f)
    for issue_key, issue_data in data.items():
      rankings = issue_data.get("rankings", {})
      for uid, ranking in rankings.items():
        all_rankings.append(ranking)
  return all_rankings


def compute_h2h(all_rankings, systems):
  """Compute pairwise win rates.

  Returns an NxN matrix where matrix[i][j] = fraction of rankings
  where systems[i] was ranked above systems[j].
  """
  n = len(systems)
  wins = np.zeros((n, n))
  counts = np.zeros((n, n))

  sys_idx = {s: i for i, s in enumerate(systems)}

  for ranking in all_rankings:
    # Build position map for this voter.
    pos = {}
    for rank, sys_name in enumerate(ranking):
      if sys_name in sys_idx:
        pos[sys_name] = rank

    # Pairwise comparison.
    for si in systems:
      for sj in systems:
        if si == sj:
          continue
        if si in pos and sj in pos:
          i, j = sys_idx[si], sys_idx[sj]
          counts[i][j] += 1
          if pos[si] < pos[sj]:  # lower rank = better
            wins[i][j] += 1

  # Convert to rates.
  with np.errstate(divide="ignore", invalid="ignore"):
    rates = np.where(counts > 0, wins / counts, 0.5)

  return rates


def main():
  print("Loading rankings...")
  all_rankings = load_all_rankings()
  print(f"Loaded {len(all_rankings)} voter rankings.")

  systems = SYSTEM_ORDER
  display_names = [SYSTEM_DISPLAY[s] for s in systems]
  n = len(systems)

  rates = compute_h2h(all_rankings, systems)

  # --- Plot ---------------------------------------------------------------
  fig, ax = plt.subplots(figsize=(9, 7.5))

  # Mask diagonal.
  masked = np.ma.masked_where(np.eye(n, dtype=bool), rates)

  im = ax.imshow(
      masked * 100,
      cmap="RdYlGn",
      vmin=20,
      vmax=80,
      aspect="equal",
  )

  # Annotate cells.
  for i in range(n):
    for j in range(n):
      if i == j:
        ax.text(j, i, "-", ha="center", va="center", fontsize=10, color="gray")
        continue
      val = rates[i, j] * 100
      color = "white" if val < 30 or val > 70 else "black"
      ax.text(
          j,
          i,
          f"{val:.0f}%",
          ha="center",
          va="center",
          fontsize=9,
          fontweight="bold",
          color=color,
      )

  # Axes.
  ax.set_xticks(range(n))
  ax.set_xticklabels(display_names, fontsize=8.5, ha="center")
  ax.set_yticks(range(n))
  ax.set_yticklabels(display_names, fontsize=8.5)
  ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

  ax.set_xlabel("Column System (opponent)", fontsize=10, labelpad=8)
  ax.set_ylabel("Row System (wins)", fontsize=10, labelpad=8)
  ax.xaxis.set_label_position("bottom")

  ax.set_title(
      "Head-to-Head Win Rate (%)\n"
      "Row system ranked higher than column system by voters",
      fontsize=12,
      fontweight="bold",
      pad=25,
  )

  # Group separator lines.
  for b in GROUP_BOUNDARIES:
    ax.axhline(y=b - 0.5, color="black", linewidth=2)
    ax.axvline(x=b - 0.5, color="black", linewidth=2)

  # Grid.
  ax.set_xticks(np.arange(-0.5, n, 1), minor=True)
  ax.set_yticks(np.arange(-0.5, n, 1), minor=True)
  ax.grid(which="minor", color="white", linewidth=1.5)
  ax.tick_params(which="minor", size=0)

  # Colorbar.
  cbar = fig.colorbar(im, ax=ax, shrink=0.75, pad=0.02)
  cbar.set_label("Win Rate (%)", fontsize=10)

  plt.tight_layout()

  # Save.
  pdf_path = os.path.join(DRAFT_DIR, "system_h2h.pdf")
  fig.savefig(pdf_path, format="pdf", dpi=300, bbox_inches="tight")
  print(f"Saved: {pdf_path}")

  png_path = os.path.join(DRAFT_DIR, "system_h2h.png")
  fig.savefig(png_path, dpi=200, bbox_inches="tight")
  print(f"Preview: {png_path}")
  plt.close(fig)


if __name__ == "__main__":
  main()
