#!/usr/bin/env python3
"""Computes OLS fit and correlations when using the median harm score across judges."""

import json
import os
import numpy as np
from collections import defaultdict
from scipy import stats

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_data")

GROUPS = {
    "Baseline": ["baseline", "baseline_informed"],
    "Majoritarian": ["fptp", "alternative_vote", "trs", "sntv"],
    "Proportional": ["dhondt", "sainte_lague", "stv"],
}
SYSTEM_TO_GROUP = {}
for g, systems in GROUPS.items():
    for s in systems:
        SYSTEM_TO_GROUP[s] = g

SYSTEM_ORDER = [
    "fptp", "sntv", "alternative_vote", "trs",
    "dhondt", "stv", "sainte_lague",
    "baseline", "baseline_informed",
]

# The three key judges specified by the user
THREE_JUDGES = ["claude-opus", "gemini-flash", "gpt-4o"]

MODELS = [
    {
        "key": "llama",
        "label": "Llama-3.3-70B",
        "judge_files": {
            "claude-opus": "harm-12.judge.llama-3.3-70b-instruct.judged_by.claude-opus-4.7.json",
            "gemini-flash": "harm-12.judge.llama-3.3-70b-instruct.judged_by.gemini-3-flash-preview.json",
            "gpt-4o": "harm-12.judge.llama-3.3-70b-instruct.judged_by.gpt-4o-2024-11-20.json",
            "claude-haiku": "harm-12.judge.llama-3.3-70b-instruct.judged_by.claude-3.5-haiku.json",
            "llama-70b": "harm-12.judge.llama-3.3-70b-instruct.judged_by.llama-3.1-70b-instruct.json",
        },
        "issue_file": "harm-12.issue.llama-3.3-70b-instruct.json",
    },
    {
        "key": "mistral",
        "label": "Mistral-Medium-3",
        "judge_files": {
            "claude-opus": "harm-12.judge.mistral-medium-3.judged_by.claude-opus-4.7.json",
            "gemini-flash": "harm-12.judge.mistral-medium-3.judged_by.gemini-3-flash-preview.json",
            "gpt-4o": "harm-12.judge.mistral-medium-3.judged_by.gpt-4o-2024-11-20.json",
            "claude-haiku": "harm-12.judge.mistral-medium-3.judged_by.claude-3.5-haiku.json",
            "llama-70b": "harm-12.judge.mistral-medium-3.judged_by.llama-3.1-70b-instruct.json",
        },
        "issue_file": "harm-12.issue.mistral-medium-3.json",
    },
]

def analyze_median_harm(judge_list, judge_list_name):
    print("\n" + "=" * 80)
    print(f"MEDIAN ANALYSIS USING JUDGES: {judge_list_name} ({len(judge_list)} judges)")
    print("=" * 80)
    
    for mcfg in MODELS:
        print(f"\n--- Source Model: {mcfg['label']} ---")
        
        # Load issue data
        with open(os.path.join(DATA_DIR, mcfg["issue_file"])) as f:
            issue_data = json.load(f)
            
        # Compute mean preferences
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
                
        # Load judge data for the selected subset
        judge_dicts = {}
        for jname in judge_list:
            fname = mcfg["judge_files"][jname]
            with open(os.path.join(DATA_DIR, fname)) as f:
                judge_dicts[jname] = json.load(f)
                
        # Build records by taking the MEDIAN harm score across judges for each (issue, system)
        records = []
        for issue in issue_data:
            for s in SYSTEM_ORDER:
                harms = []
                for jname in judge_list:
                    jdata = judge_dicts[jname]
                    if issue in jdata and s in jdata[issue]["aggregate"]["mean_harm_score"]:
                        harms.append(jdata[issue]["aggregate"]["mean_harm_score"][s])
                if harms and (issue, s) in mean_prefs:
                    records.append({
                        "system": s,
                        "group": SYSTEM_TO_GROUP[s],
                        "preference": mean_prefs[(issue, s)],
                        "harm": np.median(harms)
                    })
                    
        # 1. Overall OLS and Correlation
        prefs = np.array([r["preference"] for r in records])
        harms = np.array([r["harm"] for r in records])
        rp, pp = stats.pearsonr(prefs, harms)
        rs, ps = stats.spearmanr(prefs, harms)
        slope, intercept, r_value, p_value, std_err = stats.linregress(prefs, harms)
        
        print(f"  Overall Statistics (n={len(records)}):")
        print(f"    Pearson  r = {rp:.4f} (p = {pp:.2e})")
        print(f"    Spearman r = {rs:.4f} (p = {ps:.2e})")
        print(f"    OLS Fit: harm = {slope:.4f} * pref + {intercept:.4f}")
        
        # 2. Category-level Statistics
        print("\n    Category Correlations and Fits:")
        for g in ["Baseline", "Majoritarian", "Proportional"]:
            g_prefs = np.array([r["preference"] for r in records if r["group"] == g])
            g_harms = np.array([r["harm"] for r in records if r["group"] == g])
            if len(g_prefs) > 2:
                rg, pg = stats.pearsonr(g_prefs, g_harms)
                g_slope, g_intercept, _, _, _ = stats.linregress(g_prefs, g_harms)
                print(f"      {g:<15s} (n={len(g_prefs):>2d}): r = {rg:>7.4f} (p = {pg:.1e}) | OLS: harm = {g_slope:.3f} * pref + {g_intercept:.3f}")
            else:
                print(f"      {g:<15s}: N/A")

if __name__ == "__main__":
    # Perform analysis for the 3 main judges requested by the user
    analyze_median_harm(THREE_JUDGES, "Claude-Opus, Gemini-Flash, GPT-4o")
    
    # Perform analysis for all 5 judges for comparison
    all_judges = ["claude-opus", "gemini-flash", "gpt-4o", "claude-haiku", "llama-70b"]
    analyze_median_harm(all_judges, "All 5 Judges Pooled")
