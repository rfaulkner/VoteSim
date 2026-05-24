#!/usr/bin/env python3
"""Compute head-to-head pairwise win rates for harm analysis.

A 'win' for system A over system B means system A was judged to have LESS harm.
We pool decisions across all 12 issues and the three frontier judges
(Claude-Opus, Gemini-Flash, GPT-4o).
"""

import json
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_data")

SYSTEM_ORDER = [
    "fptp", "sntv", "alternative_vote", "trs",
    "dhondt", "stv", "sainte_lague",
    "baseline", "baseline_informed",
]

SYSTEM_DISPLAY = {
    "fptp": "FPTP",
    "sntv": "PBV",
    "alternative_vote": "IRV",
    "trs": "Two-Round",
    "dhondt": "D'Hondt",
    "stv": "STV",
    "sainte_lague": "Sainte-Laguë",
    "baseline": "Baseline",
    "baseline_informed": "Oracle Med.",
}

THREE_JUDGES = ["claude-opus", "gemini-flash", "gpt-4o"]

MODELS = {
    "Llama-3.3-70B": {
        "claude-opus": "harm-12.judge.llama-3.3-70b-instruct.judged_by.claude-opus-4.7.json",
        "gemini-flash": "harm-12.judge.llama-3.3-70b-instruct.judged_by.gemini-3-flash-preview.json",
        "gpt-4o": "harm-12.judge.llama-3.3-70b-instruct.judged_by.gpt-4o-2024-11-20.json",
    },
    "Mistral-Medium-3": {
        "claude-opus": "harm-12.judge.mistral-medium-3.judged_by.claude-opus-4.7.json",
        "gemini-flash": "harm-12.judge.mistral-medium-3.judged_by.gemini-3-flash-preview.json",
        "gpt-4o": "harm-12.judge.mistral-medium-3.judged_by.gpt-4o-2024-11-20.json",
    },
}

def compute_h2h_matrix(mname, files):
    # Load all judge JSONs
    judges_data = {}
    for jname, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        if os.path.exists(path):
            with open(path) as f:
                judges_data[jname] = json.load(f)
                
    n = len(SYSTEM_ORDER)
    wins = np.zeros((n, n))
    counts = np.zeros((n, n))
    
    # Iterate over judges, issues, and pairwise comparisons
    for jname, jdata in judges_data.items():
        for issue, issue_d in jdata.items():
            pairwise = issue_d.get("pairwise", {})
            for key, val in pairwise.items():
                winner = val.get("winner")
                loser = val.get("loser")
                if winner and loser and winner in SYSTEM_ORDER and loser in SYSTEM_ORDER:
                    w_idx = SYSTEM_ORDER.index(winner)
                    l_idx = SYSTEM_ORDER.index(loser)
                    wins[w_idx, l_idx] += 1
                    counts[w_idx, l_idx] += 1
                    counts[l_idx, w_idx] += 1  # loser played winner
                    
    # Compute rates
    rates = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i == j:
                rates[i, j] = 0.5
            elif counts[i, j] > 0:
                rates[i, j] = wins[i, j] / counts[i, j]
            else:
                rates[i, j] = 0.5
    return rates, counts

def print_report(mname, rates, counts):
    print(f"\n{'='*80}")
    print(f"HARM pairwise win rates for source model: {mname}")
    print(f"Pooled across 12 issues and 3 judges: Claude-Opus, Gemini-Flash, GPT-4o")
    print(f"{'='*80}")
    
    n = len(SYSTEM_ORDER)
    print(f"\n  {'System':<15} " + " ".join([f"{SYSTEM_DISPLAY[s]:>6}" for s in SYSTEM_ORDER]))
    print("  " + "-" * (15 + 7 * n))
    
    for i, s1 in enumerate(SYSTEM_ORDER):
        row_strs = []
        for j, s2 in enumerate(SYSTEM_ORDER):
            if i == j:
                row_strs.append("   -  ")
            else:
                pct = rates[i, j] * 100
                row_strs.append(f"{pct:>5.1f}%")
        print(f"  {SYSTEM_DISPLAY[s1]:<15} " + " ".join(row_strs))
        
    # Calculate Condorcet winner: system that has win rate > 50% against all other systems
    condorcet_winners = []
    for i, s1 in enumerate(SYSTEM_ORDER):
        is_condorcet = True
        for j, s2 in enumerate(SYSTEM_ORDER):
            if i != j and rates[i, j] <= 0.5:
                is_condorcet = False
                break
        if is_condorcet:
            condorcet_winners.append(s1)
            
    print("\n  Matchup Wins (out of 8 opponents):")
    matchup_scores = {}
    for i, s1 in enumerate(SYSTEM_ORDER):
        wins_over_opponents = sum(1 for j, s2 in enumerate(SYSTEM_ORDER) if i != j and rates[i, j] > 0.5)
        ties_with_opponents = sum(1 for j, s2 in enumerate(SYSTEM_ORDER) if i != j and rates[i, j] == 0.5)
        matchup_scores[s1] = wins_over_opponents
        print(f"    {SYSTEM_DISPLAY[s1]:<15} won {wins_over_opponents}/8 matchups (ties: {ties_with_opponents})")
        
    print("\n  Condorcet Winner(s) (won 8/8 matchups):")
    if condorcet_winners:
        for cw in condorcet_winners:
            print(f"    *** {SYSTEM_DISPLAY[cw]} ***")
    else:
        print("    None (no single system beat all others head-to-head)")
        # Let's find the Borda-like count (sum of win rates)
        borda_scores = []
        for i, s1 in enumerate(SYSTEM_ORDER):
            borda = sum(rates[i, j] for j in range(n) if i != j)
            borda_scores.append((s1, borda))
        borda_scores.sort(key=lambda x: x[1], reverse=True)
        print("\n    Borda-like ranking (sum of win rates across all 8 opponents):")
        for rank, (sys, score) in enumerate(borda_scores):
            print(f"      {rank+1}. {SYSTEM_DISPLAY[sys]:<15} sum of win rates = {score:.3f}")

if __name__ == "__main__":
    for mname, files in MODELS.items():
        rates, counts = compute_h2h_matrix(mname, files)
        print_report(mname, rates, counts)
