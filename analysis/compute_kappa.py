#!/usr/bin/env python3
"""Compute Cohen's kappa across 3 judges for Llama-70B and Mistral-Medium."""

import json
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_data")

JUDGES = ["gemini-flash", "gpt-4o", "llama-70b"]

SOURCE_MODELS = {
    "Llama-3.3-70B": {
        "gemini-flash": "harm-12.judge.llama-3.3-70b-instruct.judged_by.gemini-3-flash-preview.json",
        "gpt-4o": "harm-12.judge.llama-3.3-70b-instruct.judged_by.gpt-4o-2024-11-20.json",
        "llama-70b": "harm-12.judge.llama-3.3-70b-instruct.judged_by.llama-3.1-70b-instruct.json",
    },
    "Mistral-Medium-3": {
        "gemini-flash": "harm-12.judge.mistral-medium-3.judged_by.gemini-3-flash-preview.json",
        "gpt-4o": "harm-12.judge.mistral-medium-3.judged_by.gpt-4o-2024-11-20.json",
        "llama-70b": "harm-12.judge.mistral-medium-3.judged_by.llama-3.1-70b-instruct.json",
    },
}

JUDGE_LABELS = {
    "gemini-flash": "Gemini-3-Flash",
    "gpt-4o": "GPT-4o",
    "llama-70b": "Llama-3.1-70B",
}


def get_pairwise_decisions(judge_data):
    """Return dict of (issue, pair_key) -> winner system name."""
    decisions = {}
    for issue, issue_d in judge_data.items():
        for pair_key, pair_val in issue_d["pairwise"].items():
            decisions[(issue, pair_key)] = pair_val["winner"]
    return decisions


def cohens_kappa(da, db):
    """Compute Cohen's kappa between two sets of categorical decisions."""
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


def fleiss_kappa(all_decisions, judge_names):
    """Compute Fleiss' kappa for multiple raters."""
    common_keys = sorted(
        set.intersection(*[set(d.keys()) for d in all_decisions.values()])
    )
    all_cats = sorted(set(v for d in all_decisions.values() for v in d.values()))
    ci = {c: i for i, c in enumerate(all_cats)}
    n_items = len(common_keys)
    n_raters = len(judge_names)

    ratings = np.zeros((n_items, len(all_cats)), dtype=int)
    for idx, key in enumerate(common_keys):
        for nm in judge_names:
            ratings[idx, ci[all_decisions[nm][key]]] += 1

    Pi = (np.sum(ratings**2, axis=1) - n_raters) / (n_raters * (n_raters - 1))
    Pbar = np.mean(Pi)
    pj = np.sum(ratings, axis=0) / (n_items * n_raters)
    Pe = np.sum(pj**2)
    fk = (Pbar - Pe) / (1 - Pe) if Pe < 1 else 1.0

    agree_all = int(np.sum(np.max(ratings, 1) == n_raters))
    agree_maj = int(np.sum(np.max(ratings, 1) >= 2))
    return fk, n_items, agree_all, agree_maj


# ── Main ─────────────────────────────────────────────────────────────
for source_model, files in SOURCE_MODELS.items():
    print("=" * 70)
    print(f"Source Model: {source_model}")
    print("=" * 70)

    # Load judge data
    judges = {}
    for jname, fname in files.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            judges[jname] = json.load(f)

    all_decisions = {
        jname: get_pairwise_decisions(jdata) for jname, jdata in judges.items()
    }

    # Pairwise Cohen's kappa
    print("\n  Pairwise Cohen's κ:")
    for i in range(len(JUDGES)):
        for j in range(i + 1, len(JUDGES)):
            na, nb = JUDGES[i], JUDGES[j]
            k, n = cohens_kappa(all_decisions[na], all_decisions[nb])
            print(
                f"    {JUDGE_LABELS[na]:18s} vs {JUDGE_LABELS[nb]:18s}:  "
                f"κ = {k:.4f}  (n={n})"
            )

    # Fleiss' kappa
    fk, n_items, agree_all, agree_maj = fleiss_kappa(all_decisions, JUDGES)
    print(f"\n  Fleiss' κ (all 3 judges) = {fk:.4f}  (n={n_items})")
    print(f"  Full agreement:   {agree_all}/{n_items} = {agree_all/n_items:.1%}")
    print(f"  Majority (≥2/3):  {agree_maj}/{n_items} = {agree_maj/n_items:.1%}")
    print()
