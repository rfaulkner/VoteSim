#!/usr/bin/env python3
"""Generate a heatmap of issue-vs-platform voter preference rates.

Reads compare-mode JSON files and plots a heatmap where:
  - Y-axis: issues (short labels)
  - X-axis: models
  - Cell value: issue-mode win rate (fraction of voters who preferred
    the issue-mode bill over the platform-mode bill)

Saves to draft/issue_platform_h2h.pdf as a high-resolution vector PDF.
"""

import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
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

MODEL_FILES = {
    "Gemma-4\n31B": "issue-12.compare.gemma-4-31b-it.json",
    "Grok-4.1\nFast": "issue-12.compare.grok-4.1-fast.json",
    "Hermes-3\n70B": "issue-12.compare.hermes-3-llama-3.1-70b.json",
    "Llama-3.1\n8B": "issue-12.compare.llama-3.1-8b-instruct.json",
    "Llama-3.3\n70B": "issue-12.compare.llama-3.3-70b-instruct.json",
    "Mistral\nMed-3": "issue-12.compare.mistral-medium-3.json",
    "Mistral\nSm-24B": "issue-12.compare.mistral-small-3.1-24b-instruct.json",
}

# Short issue labels.
ISSUE_SHORT_MAP = {
    "universal basic income": "UBI",
    "abortion": "Abortion",
    "firearms": "Firearms",
    "immigration": "Immigration",
    "affirmative action": "Aff. Action",
    "death penalty": "Death Penalty",
    "religious": "Relig. Freedom",
    "law enforcement": "Policing",
    "healthcare": "Healthcare",
    "fossil fuel": "Climate Liab.",
    "national service": "Nat. Service",
    "transgender": "Trans Athletes",
}


def _issue_to_short(issue_text):
  lower = issue_text.lower()
  for key, label in ISSUE_SHORT_MAP.items():
    if key in lower:
      return label
  return issue_text[:20]


def main():
  # Load all data.
  model_names = list(MODEL_FILES.keys())
  all_data = {}
  for model_name, fname in MODEL_FILES.items():
    path = os.path.join(RESULTS_DIR, fname)
    with open(path) as f:
      all_data[model_name] = json.load(f)

  # Get issue keys from first model (ordered).
  first_data = next(iter(all_data.values()))
  issue_keys = list(first_data.keys())
  issue_labels = [_issue_to_short(k) for k in issue_keys]

  # Build the matrix: rows=issues, cols=models.
  n_issues = len(issue_keys)
  n_models = len(model_names)
  matrix = np.zeros((n_issues, n_models))

  for j, model_name in enumerate(model_names):
    data = all_data[model_name]
    for i, issue_key in enumerate(issue_keys):
      entry = data.get(issue_key, {})
      rate = entry.get("issue_voting", {}).get("preference_rate", 0.5)
      matrix[i, j] = rate

  # Sort issues by mean win rate (descending) for visual clarity.
  mean_rates = matrix.mean(axis=1)
  sort_idx = np.argsort(-mean_rates)
  matrix = matrix[sort_idx]
  issue_labels = [issue_labels[i] for i in sort_idx]

  # --- Plot ---------------------------------------------------------------
  fig, ax = plt.subplots(figsize=(7, 4.8))

  # Use a diverging colormap centered at 0.5.
  im = ax.imshow(
      matrix,
      cmap="RdYlGn",
      vmin=0.3,
      vmax=0.9,
      aspect="auto",
  )

  # Annotate cells with win rate percentages.
  for i in range(n_issues):
    for j in range(n_models):
      val = matrix[i, j]
      color = "white" if val < 0.4 or val > 0.8 else "black"
      ax.text(
          j,
          i,
          f"{val:.0%}",
          ha="center",
          va="center",
          fontsize=9,
          fontweight="bold",
          color=color,
      )

  # Axes.
  ax.set_xticks(range(n_models))
  ax.set_xticklabels(model_names, fontsize=10, ha="center")
  ax.set_yticks(range(n_issues))
  ax.set_yticklabels(issue_labels, fontsize=11)
  ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False, pad=2)

  # (title removed for publication)

  # Colorbar.
  cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.01)
  cbar.set_label("Issue-mode preference rate", fontsize=10, labelpad=4)

  # Draw grid lines.
  ax.set_xticks(np.arange(-0.5, n_models, 1), minor=True)
  ax.set_yticks(np.arange(-0.5, n_issues, 1), minor=True)
  ax.grid(which="minor", color="white", linewidth=1.5)
  ax.tick_params(which="minor", size=0)

  plt.tight_layout(pad=0.5)

  # Save as high-resolution vector PDF.
  out_path = os.path.join(DRAFT_DIR, "issue_platform_h2h.pdf")
  fig.savefig(out_path, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
  print(f"Saved: {out_path}")

  # Also save a PNG preview.
  png_path = os.path.join(DRAFT_DIR, "issue_platform_h2h.png")
  fig.savefig(png_path, dpi=200, bbox_inches="tight", pad_inches=0.02)
  print(f"Preview: {png_path}")
  plt.close(fig)


if __name__ == "__main__":
  main()
