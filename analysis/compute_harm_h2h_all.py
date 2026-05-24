#!/usr/bin/env python3
"""Compute raw head-to-head pairwise win rates from ALL harm-12 judge files.

Reports win rates by voting system, pooled across issues.
Breakdowns: per source model, per judge, and grand total.
"""

import glob
import json
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "golden_data")

# System ordering and display names
SYSTEMS = [
    "fptp", "sntv", "alternative_vote", "trs",
    "dhondt", "stv", "sainte_lague",
    "baseline", "baseline_informed",
]

DISPLAY = {
    "fptp": "FPTP",
    "sntv": "PBV/SNTV",
    "alternative_vote": "IRV",
    "trs": "Two-Round",
    "dhondt": "D'Hondt",
    "stv": "STV",
    "sainte_lague": "Sainte-Laguë",
    "baseline": "Baseline",
    "baseline_informed": "Oracle Med.",
}

N = len(SYSTEMS)


def compute_h2h(json_files):
    """Compute H2H win matrix from a list of judge JSON files.

    Returns:
        wins: NxN array where wins[i][j] = number of times system i beat system j
        counts: NxN array where counts[i][j] = number of comparisons between i and j
    """
    wins = np.zeros((N, N), dtype=int)
    counts = np.zeros((N, N), dtype=int)

    for fpath in json_files:
        with open(fpath) as f:
            jdata = json.load(f)

        for issue, issue_d in jdata.items():
            pw = issue_d.get("pairwise", {})
            for key, val in pw.items():
                winner = val.get("winner")
                loser = val.get("loser")
                if winner in SYSTEMS and loser in SYSTEMS:
                    wi = SYSTEMS.index(winner)
                    li = SYSTEMS.index(loser)
                    wins[wi, li] += 1
                    counts[wi, li] += 1
                    counts[li, wi] += 1

    return wins, counts


def print_h2h_table(title, wins, counts):
    """Print an H2H win-rate table."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")

    # Header
    hdr = f"  {'System':<14}"
    for s in SYSTEMS:
        hdr += f" {DISPLAY[s]:>10}"
    hdr += f" {'Win%':>7}  {'W':>4} {'L':>4} {'Total':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))

    for i, s1 in enumerate(SYSTEMS):
        row = f"  {DISPLAY[s1]:<14}"
        total_wins = 0
        total_played = 0
        for j, s2 in enumerate(SYSTEMS):
            if i == j:
                row += f" {'---':>10}"
            elif counts[i, j] > 0:
                rate = wins[i, j] / counts[i, j] * 100
                row += f" {rate:>9.1f}%"
                total_wins += wins[i, j]
                total_played += counts[i, j]
            else:
                row += f" {'n/a':>10}"

        overall_pct = total_wins / total_played * 100 if total_played > 0 else 0
        total_losses = total_played - total_wins
        row += f" {overall_pct:>6.1f}%  {total_wins:>4} {total_losses:>4} {total_played:>5}"
        print(row)

    # Summary: matchup wins (>50% vs each opponent)
    print()
    print("  Matchup wins (opponents beaten >50%):")
    for i, s1 in enumerate(SYSTEMS):
        beaten = []
        lost_to = []
        for j, s2 in enumerate(SYSTEMS):
            if i != j and counts[i, j] > 0:
                rate = wins[i, j] / counts[i, j]
                if rate > 0.5:
                    beaten.append(DISPLAY[s2])
                elif rate < 0.5:
                    lost_to.append(DISPLAY[s2])
        print(f"    {DISPLAY[s1]:<14} beat {len(beaten)}/8: {', '.join(beaten) if beaten else '(none)'}")


def main():
    # Only use 3 frontier judges
    FRONTIER_JUDGES = {"claude-opus-4.7", "gemini-3-flash-preview", "gpt-4o-2024-11-20"}

    # Discover all files
    pattern = os.path.join(DATA_DIR, "harm-12.judge.*.judged_by.*.json")
    all_files_raw = sorted(glob.glob(pattern))

    # Parse structure: source_model -> judge -> filepath
    file_map = {}  # {source: {judge: filepath}}
    for fpath in all_files_raw:
        fname = os.path.basename(fpath)
        # harm-12.judge.<source>.judged_by.<judge>.json
        parts = fname.replace("harm-12.judge.", "").replace(".json", "").split(".judged_by.")
        source = parts[0]
        judge = parts[1]
        if judge not in FRONTIER_JUDGES:
            continue
        file_map.setdefault(source, {})[judge] = fpath

    all_files = [fp for src in file_map.values() for fp in src.values()]

    print(f"Found {len(all_files)} judge files:")
    for source in sorted(file_map):
        judges = sorted(file_map[source].keys())
        print(f"  Source: {source} -> Judges: {', '.join(judges)}")

    # 1. Grand total: pool ALL files
    print_h2h_table(
        f"GRAND TOTAL — All {len(all_files)} files pooled (all source models × all judges × 12 issues)",
        *compute_h2h(all_files),
    )

    # 2. Per source model
    for source in sorted(file_map):
        files = list(file_map[source].values())
        print_h2h_table(
            f"Source: {source} — pooled across {len(files)} judges × 12 issues",
            *compute_h2h(files),
        )

    # 3. Per judge (pooled across source models)
    judge_files = {}
    for source, judges in file_map.items():
        for judge, fpath in judges.items():
            judge_files.setdefault(judge, []).append(fpath)

    for judge in sorted(judge_files):
        files = judge_files[judge]
        print_h2h_table(
            f"Judge: {judge} — pooled across {len(files)} source models × 12 issues",
            *compute_h2h(files),
        )

    # 4. Individual breakdowns (source × judge)
    print(f"\n\n{'#' * 100}")
    print(f"  INDIVIDUAL SOURCE × JUDGE BREAKDOWNS")
    print(f"{'#' * 100}")
    for source in sorted(file_map):
        for judge in sorted(file_map[source]):
            fpath = file_map[source][judge]
            print_h2h_table(
                f"Source: {source} | Judge: {judge} (12 issues)",
                *compute_h2h([fpath]),
            )


if __name__ == "__main__":
    main()
