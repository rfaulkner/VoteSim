#!/usr/bin/env python3
"""Generate violin plots showing the distribution of scores for each voting system.

Reads the 9-model divisive-12 issue-mode results and produces:
  - score_dist_by_model_cohorts.png (Plot 1: Llama/Mistral cohorts 120 vs 635)
  - score_dist_by_model_others.png (Plot 2: Other models N=120)
  - score_dist_by_system.png (Grouped box plot, comparing all 9 models per system)
"""

import json
import os
from collections import defaultdict
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

MODEL_FILES = {
    "Llama-3.3 (N=120)": "divisive-12.issue.llama-3.3-70b-instruct.json",
    "Mistral-Med (N=120)": "divisive-12.issue.mistral-medium-3.json",
    "Llama-3.3 (N=635)": "divisive-12.issue.p635.llama-3.3-70b-instruct.json",
    "Mistral-Med (N=635)": "divisive-12.issue.p635.mistral-medium-3.json",
    "Gemma-4 (N=120)": "divisive-12.issue.gemma-4-31b-it.json",
    "Hermes-3 (N=120)": "divisive-12.issue.hermes-3-llama-3.1-70b.json",
    "Llama-3.1 (N=120)": "divisive-12.issue.llama-3.1-8b-instruct.json",
    "Grok-4.1 (N=120)": "divisive-12.issue.grok-4.1-fast.json",
    "Mistral-Sm (N=120)": "divisive-12.issue.mistral-small-3.1-24b-instruct.json",
}

# Order matching the tables
SYSTEM_ORDER = [
    "sntv", "fptp", "trs", "alternative_vote",
    "stv", "dhondt", "sainte_lague",
    "baseline", "baseline_informed",
]

SYSTEM_DISPLAY = {
    "fptp": "FPTP",
    "sntv": "PBV",
    "alternative_vote": "IRV",
    "trs": "TRS",
    "dhondt": "D'Hondt",
    "stv": "STV",
    "sainte_lague": "S-L",
    "baseline": "Mod. Base.",
    "baseline_informed": "Med. Base.",
}

# Distinct colors for the 9 models (paired colors for cohorts)
MODEL_COLORS = {
    "Llama-3.3 (N=120)": "#1f77b4",      # Dark Blue
    "Llama-3.3 (N=635)": "#aec7e8",      # Light Blue
    "Mistral-Med (N=120)": "#ff7f0e",    # Dark Orange
    "Mistral-Med (N=635)": "#ffbb78",    # Light Orange
    "Gemma-4 (N=120)": "#2ca02c",        # Green
    "Hermes-3 (N=120)": "#d62728",       # Red
    "Llama-3.1 (N=120)": "#9467bd",      # Purple
    "Grok-4.1 (N=120)": "#8c564b",       # Brown
    "Mistral-Sm (N=120)": "#e377c2",     # Pink
}

# ---------------------------------------------------------------------------
# Data loading & extraction
# ---------------------------------------------------------------------------

def load_scores():
    """Load all scores and return a nested dict: {model: {system: [scores]}}."""
    all_scores = defaultdict(lambda: defaultdict(list))
    
    for model_name, fname in MODEL_FILES.items():
        path = os.path.join(RESULTS_DIR, fname)
        if not os.path.exists(path):
            print(f"Warning: File not found {path}")
            continue
            
        with open(path) as f:
            data = json.load(f)
            
        for issue_key, issue in data.items():
            scores_dict = issue.get("scores", {})
            # scores_dict: {voter_id: {system_name: score}}
            for voter_id, sys_scores in scores_dict.items():
                for sys_name in SYSTEM_ORDER:
                    if sys_name in sys_scores:
                        try:
                            score = float(sys_scores[sys_name])
                            all_scores[model_name][sys_name].append(score)
                        except (ValueError, TypeError):
                            pass
                            
    return all_scores

# ---------------------------------------------------------------------------
# Plot 1 & 2: Faceted Violin Plots (parameterized)
# ---------------------------------------------------------------------------

def plot_faceted_violins(all_scores, models_to_plot, out_filename, title_suffix):
    """Generate a grid of violin plots for a specific subset of models."""
    # Filter models that actually have data
    models = [m for m in models_to_plot if m in all_scores]
    if not models:
        print(f"No data to plot for {out_filename}!")
        return
        
    fig, axes = plt.subplots(len(models), 1, figsize=(12, 3.0 * len(models)), sharex=True)
    if len(models) == 1:
        axes = [axes]
        
    for idx, model_name in enumerate(models):
        ax = axes[idx]
        model_data = all_scores[model_name]
        
        # Prepare data for violinplot
        plot_data = []
        labels = []
        for sys_name in SYSTEM_ORDER:
            scores = model_data.get(sys_name, [])
            if scores:
                plot_data.append(scores)
                labels.append(SYSTEM_DISPLAY[sys_name])
            else:
                plot_data.append([])
                labels.append(SYSTEM_DISPLAY[sys_name])
                
        # Filter out empty datasets
        positions = []
        filtered_data = []
        for i, d in enumerate(plot_data):
            if len(d) > 0:
                positions.append(i + 1)
                filtered_data.append(d)
                
        if not filtered_data:
            ax.text(0.5, 0.5, f"No data for {model_name}", ha='center', va='center')
            continue
            
        # Draw violins
        parts = ax.violinplot(filtered_data, positions=positions, showmeans=True, showmedians=False, showextrema=True)
        
        # Style violins
        color = MODEL_COLORS.get(model_name, "#333333")
        for pc in parts['bodies']:
            pc.set_facecolor(color)
            pc.set_edgecolor('black')
            pc.set_alpha(0.6)
            
        # Style lines
        parts['cbars'].set_color('black')
        parts['cbars'].set_linewidth(0.5)
        parts['cmaxes'].set_color('black')
        parts['cmaxes'].set_linewidth(0.5)
        parts['cmins'].set_color('black')
        parts['cmins'].set_linewidth(0.5)
        parts['cmeans'].set_color('red')
        parts['cmeans'].set_linewidth(1.5)
        
        # Add boxplot-like elements inside violins (quartiles)
        for i, d in enumerate(filtered_data):
            pos = positions[i]
            quartile1, median, quartile3 = np.percentile(d, [25, 50, 75])
            ax.scatter(pos, median, color='white', edgecolor='black', s=30, zorder=3)
            ax.vlines(pos, quartile1, quartile3, color='black', linestyle='-', lw=3)
            
        # Clean up display name for title (remove N=120 for cleaner look if desired, or keep it)
        # Label each subplot with model name using ylabel instead of title
        ax.annotate(model_name, xy=(0, 0.5), xytext=(-58, 0),
                    xycoords='axes fraction', textcoords='offset points',
                    fontsize=13, fontweight='bold', va='center', ha='right',
                    rotation=90)
        ax.set_ylabel("")  # cleared; shared ylabel added below
        ax.set_ylim(0.8, 5.2)
        ax.set_yticks([1, 2, 3, 4, 5])
        ax.tick_params(axis='y', labelsize=13, pad=2)
        for lbl in ax.get_yticklabels():
            lbl.set_fontweight('bold')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        
        # Draw category background zones
        ax.axvspan(0.5, 4.5, color='red', alpha=0.04, zorder=0)
        ax.axvspan(4.5, 7.5, color='green', alpha=0.04, zorder=0)
        ax.axvspan(7.5, 9.5, color='blue', alpha=0.04, zorder=0)
        
    # Set x-ticks on the bottom subplot
    axes[-1].set_xticks(range(1, len(SYSTEM_ORDER) + 1))
    axes[-1].set_xticklabels([SYSTEM_DISPLAY[s] for s in SYSTEM_ORDER],
                             fontsize=14, fontweight='bold', rotation=15)
    # Bold x-ticks on all axes (minor visible via sharex)
    for ax in axes:
        ax.tick_params(axis='x', labelsize=14)
        for lbl in ax.get_xticklabels():
            lbl.set_fontweight('bold')

    # (suptitle removed for publication)
    plt.subplots_adjust(hspace=0.15)
    plt.tight_layout(rect=[0.05, 0, 1, 1.0], pad=0.4)

    # Shared ylabel (placed after tight_layout so coordinates are stable)
    fig.text(0.01, 0.5, "Likert Score (1\u20135)", va='center', rotation='vertical',
             fontsize=15, fontweight='bold')
    
    out = os.path.join(DRAFT_DIR, out_filename)
    plt.savefig(out, dpi=150, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out}")
    
    # Save PDF
    out_pdf = os.path.splitext(out)[0] + ".pdf"
    plt.savefig(out_pdf, format="pdf", dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved {out_pdf}")
    plt.close()

# ---------------------------------------------------------------------------
# Plot 3: Grouped Box Plot (comparing all 9 models per system)
# ---------------------------------------------------------------------------

def plot_grouped_boxes(all_scores):
    """Generate a grouped box plot comparing all 9 models for each system.
    
    Optimized for 9 models with thinner boxes and wider figure.
    """
    # Maintain the order in CONFIG/MODELS
    models_order = list(MODEL_FILES.keys())
    models = [m for m in models_order if m in all_scores]
    if not models:
        print("No data to plot for grouped boxes!")
        return
        
    fig, ax = plt.subplots(figsize=(18, 8))
    
    n_systems = len(SYSTEM_ORDER)
    n_models = len(models)
    
    # Thinner boxes for 9 models to prevent overlap
    box_width = 0.07
    
    # Calculate positions
    offsets = np.linspace(-box_width * (n_models - 1) / 2, box_width * (n_models - 1) / 2, n_models)
    
    legend_handles = []
    
    for sys_idx, sys_name in enumerate(SYSTEM_ORDER):
        sys_center = sys_idx + 1
        
        for model_idx, model_name in enumerate(models):
            scores = all_scores[model_name].get(sys_name, [])
            if not scores:
                continue
                
            pos = sys_center + offsets[model_idx]
            color = MODEL_COLORS.get(model_name, "#333333")
            
            # Draw boxplot
            bp = ax.boxplot(scores, positions=[pos], widths=box_width,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", linewidth=1.2),
                            boxprops=dict(facecolor=color, edgecolor="black", alpha=0.8))
            
            # Collect handles for legend (only once per model)
            if sys_idx == 0:
                legend_handles.append((bp["boxes"][0], model_name))
                
    # Style the plot
    ax.set_xlim(0.5, n_systems + 0.5)
    ax.set_xticks(range(1, n_systems + 1))
    ax.set_xticklabels([SYSTEM_DISPLAY[s] for s in SYSTEM_ORDER], fontsize=12, rotation=15)
    ax.set_ylabel("Likert Score (1-5)", fontsize=14, fontweight='bold')
    ax.set_ylim(0.8, 5.2)
    ax.set_yticks([1, 2, 3, 4, 5])
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    
    # Add category background zones
    ax.axvspan(0.5, 4.5, color='red', alpha=0.03, zorder=0)
    ax.axvspan(4.5, 7.5, color='green', alpha=0.03, zorder=0)
    ax.axvspan(7.5, 9.5, color='blue', alpha=0.03, zorder=0)
    
    # Add labels for zones at the top
    ax.text(2.5, 5.08, "Majoritarian", color="red", fontsize=12, fontweight="bold", ha="center")
    ax.text(6.0, 5.08, "Proportional", color="green", fontsize=12, fontweight="bold", ha="center")
    ax.text(8.5, 5.08, "Baselines", color="blue", fontsize=12, fontweight="bold", ha="center")
    
    # Legend
    ax.legend([h[0] for h in legend_handles], [h[1] for h in legend_handles],
              loc="lower left", fontsize=10, frameon=True, facecolor="white", edgecolor="gray", ncol=2)
    
    ax.set_title("Comparison of Score Distributions across All Models and Electoral Systems",
                 fontsize=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    out = os.path.join(DRAFT_DIR, "score_dist_by_system.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("Loading scores from golden results...")
    scores = load_scores()
    
    # Define model subsets for split plots
    COHORT_MODELS = [
        "Llama-3.3 (N=120)",
        "Mistral-Med (N=120)",
        "Llama-3.3 (N=635)",
        "Mistral-Med (N=635)"
    ]
    
    OTHER_MODELS = [
        "Gemma-4 (N=120)",
        "Hermes-3 (N=120)",
        "Llama-3.1 (N=120)",
        "Grok-4.1 (N=120)",
        "Mistral-Sm (N=120)"
    ]
    
    print("Generating faceted violin plots for cohorts...")
    plot_faceted_violins(scores, COHORT_MODELS, "score_dist_by_model_cohorts.png", "Comparison Cohorts (N=120 vs N=635)")
    
    print("Generating faceted violin plots for other models...")
    plot_faceted_violins(scores, OTHER_MODELS, "score_dist_by_model_others.png", "Other Models (N=120)")
    
    print("Generating grouped box plots...")
    plot_grouped_boxes(scores)
    
    print("Done!")
