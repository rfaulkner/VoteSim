#!/usr/bin/env python3
"""Detect intransitive cycles in pairwise judge comparisons.

For each (source model, judge, issue), build a directed graph where
A -> B means "A beats B". Then find all 3-cycles (A>B, B>C, C>A).
"""

import json
import os
from itertools import combinations

DATA_DIR = os.path.dirname(os.path.abspath(__file__))

ALL_FILES = {
    "Llama-3.3-70B": {
        "Gemini-Flash": "harm-12.judge.llama-3.3-70b-instruct.judged_by.gemini-3-flash-preview.json",
        "GPT-4o": "harm-12.judge.llama-3.3-70b-instruct.judged_by.gpt-4o-2024-11-20.json",
        "Llama-3.1-70B": "harm-12.judge.llama-3.3-70b-instruct.judged_by.llama-3.1-70b-instruct.json",
    },
    "Mistral-Medium-3": {
        "Gemini-Flash": "harm-12.judge.mistral-medium-3.judged_by.gemini-3-flash-preview.json",
        "GPT-4o": "harm-12.judge.mistral-medium-3.judged_by.gpt-4o-2024-11-20.json",
        "Llama-3.1-70B": "harm-12.judge.mistral-medium-3.judged_by.llama-3.1-70b-instruct.json",
    },
}


def find_cycles(judge_data):
    """Find all 3-cycles per issue. Returns list of (issue, cycle_triple)."""
    cycles = []
    for issue, issue_d in judge_data.items():
        # Build directed graph: winner -> loser means winner beats loser
        beats = {}  # (a, b) -> True means a beats b
        systems = set()
        for pair_key, pair_val in issue_d["pairwise"].items():
            w = pair_val["winner"]
            l = pair_val["loser"]
            beats[(w, l)] = True
            systems.add(w)
            systems.add(l)

        # Check all triples for cycles
        systems = sorted(systems)
        for a, b, c in combinations(systems, 3):
            # Check all 3 rotational orderings for a 3-cycle
            # A>B, B>C, C>A
            if beats.get((a, b)) and beats.get((b, c)) and beats.get((c, a)):
                cycles.append((issue, (a, b, c)))
            # A>C, C>B, B>A
            if beats.get((a, c)) and beats.get((c, b)) and beats.get((b, a)):
                cycles.append((issue, (a, c, b)))

    return cycles


total_cycles = 0
total_checks = 0

for source_model, judge_files in ALL_FILES.items():
    print("=" * 70)
    print(f"Source Model: {source_model}")
    print("=" * 70)

    for judge_name, fname in judge_files.items():
        path = os.path.join(DATA_DIR, fname)
        with open(path) as f:
            data = json.load(f)

        cycles = find_cycles(data)
        n_issues = len(data)

        # Count total triples checked
        # For 9 systems, C(9,3) = 84 triples per issue
        first_issue = list(data.values())[0]
        systems = set()
        for pv in first_issue["pairwise"].values():
            systems.add(pv["winner"])
            systems.add(pv["loser"])
        n_systems = len(systems)
        n_triples = len(list(combinations(range(n_systems), 3)))
        checks = n_issues * n_triples
        total_checks += checks
        total_cycles += len(cycles)

        print(f"\n  Judge: {judge_name}")
        print(f"    Systems: {n_systems}, Issues: {n_issues}")
        print(f"    Triples checked: {checks}")
        print(f"    Cycles found: {len(cycles)}")

        if cycles:
            # Group by issue
            by_issue = {}
            for issue, triple in cycles:
                short = issue[:60] + "..." if len(issue) > 60 else issue
                by_issue.setdefault(short, []).append(triple)

            for issue, triples in sorted(by_issue.items()):
                print(f"      Issue: {issue}")
                for a, b, c in triples:
                    print(f"        {a} > {b} > {c} > {a}")
        else:
            print(f"    ✓ No cycles — fully transitive")

print()
print("=" * 70)
print(f"TOTAL: {total_cycles} cycles across {total_checks} triples checked")
print("=" * 70)
