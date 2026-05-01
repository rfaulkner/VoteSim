"""Voting system implementations for district-based seat allocation.

Provides six electoral systems:

*Majoritarian:*
1. **First Past The Post (FPTP)** — winner-takes-all in each district.
2. **Single-Member District Plurality (SMDP)** — like FPTP but fixed at
   one seat per district.
3. **Alternative Vote (AV / IRV)** — iterative elimination of the
   weakest candidate; the first candidate to reach an absolute majority
   wins the single district seat.

*Proportional:*
4. **D'Hondt** — highest-averages method with divisors 1, 2, 3, …
5. **Hare quota (largest remainders)** — seats allocated by full quotas
   then remaining seats to the parties with the largest fractional
   remainders.
6. **Sainte-Laguë** — highest-averages method with odd divisors
   1, 3, 5, …

Districts are assigned seats (1–5) based on their relative population within
the region.  The ``run_election`` dispatcher selects the system by name.
"""

from dataclasses import dataclass
import logging
from typing import Any
from typing import Dict
from typing import List

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class VoterBallot:
  """A single voter's ranked party preferences."""

  user_id: str
  district_name: str
  ranking: List[str]  # Ideology names, best-first.


@dataclass
class DistrictResult:
  """Seat-allocation outcome for a single district."""

  district_name: str
  seats_available: int
  vote_counts: Dict[str, int]
  seats: Dict[str, int]
  winner: str  # Party with the most seats (ties broken alphabetically).


@dataclass
class ElectionResult:
  """Aggregate election outcome across all districts."""

  voting_system: str
  district_results: List[DistrictResult]
  total_seats: Dict[str, int]
  governing_party: str  # Party with the most total seats.

  def summary(self) -> str:
    """Return a human-readable election summary."""
    lines = [
        "--- Election Results ---",
        f"Voting System: {self.voting_system}",
    ]

    for dr in sorted(self.district_results, key=lambda d: d.district_name):
      lines.append(
          f"\n  District: {dr.district_name} ({dr.seats_available}"
          f" seat{'s' if dr.seats_available != 1 else ''})"
      )
      lines.append(f"    Votes: {dr.vote_counts}")
      lines.append(f"    Seats: {dr.seats}")
      lines.append(f"    Winner: {dr.winner}")

    lines.append(f"\n  Total Seats: {self.total_seats}")
    lines.append(f"  Governing Party: {self.governing_party}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Seat allocation from population
# ---------------------------------------------------------------------------

MIN_SEATS = 1
MAX_SEATS = 5


def allocate_district_seats(
    districts: List[Dict[str, Any]],
    min_seats: int = MIN_SEATS,
    max_seats: int = MAX_SEATS,
) -> Dict[str, int]:
  """Assign seats to each district based on relative population.

  Uses a linear interpolation within the region's population range,
  clamped to [``min_seats``, ``max_seats``].

  Args:
    districts: List of district dicts, each with ``"name"`` and ``"population"``
      keys.
    min_seats: Minimum seats any district can have.
    max_seats: Maximum seats any district can have.

  Returns:
    ``{district_name: num_seats}``
  """
  if not districts:
    return {}

  populations = [d.get("population", 0) for d in districts]
  pop_min = min(populations)
  pop_max = max(populations)
  pop_range = pop_max - pop_min

  seat_map: Dict[str, int] = {}
  for d in districts:
    pop = d.get("population", 0)
    if pop_range > 0:
      # Linear interpolation from min_seats to max_seats.
      frac = (pop - pop_min) / pop_range
      raw = min_seats + frac * (max_seats - min_seats)
      seats = max(min_seats, min(max_seats, round(raw)))
    else:
      # All districts have equal population.
      seats = min_seats
    seat_map[d["name"]] = seats

  return seat_map


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _group_ballots_by_district(
    ballots: List[VoterBallot],
) -> Dict[str, List[VoterBallot]]:
  """Group ballots by district name."""
  grouped: Dict[str, List[VoterBallot]] = {}
  for b in ballots:
    grouped.setdefault(b.district_name, []).append(b)
  return grouped


def _count_first_choice_votes(
    ballots: List[VoterBallot],
) -> Dict[str, int]:
  """Count first-choice votes across a set of ballots."""
  counts: Dict[str, int] = {}
  for b in ballots:
    if b.ranking:
      party = b.ranking[0]
      counts[party] = counts.get(party, 0) + 1
  return counts


def _pick_winner(seats: Dict[str, int]) -> str:
  """Return the party with the most seats, ties broken alphabetically."""
  if not seats:
    return "none"
  max_seats = max(seats.values())
  winners = sorted(p for p, s in seats.items() if s == max_seats)
  return winners[0]


# ---------------------------------------------------------------------------
# FPTP — First Past The Post
# ---------------------------------------------------------------------------


def fptp(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
) -> ElectionResult:
  """First Past The Post: winner takes all seats in each district.

  Args:
    ballots: All voter ballots across all districts.
    district_seats: ``{district_name: num_seats}`` from
      ``allocate_district_seats``.

  Returns:
    An ``ElectionResult`` with per-district breakdowns.
  """
  grouped = _group_ballots_by_district(ballots)
  district_results: List[DistrictResult] = []
  total_seats: Dict[str, int] = {}

  for district_name, num_seats in sorted(district_seats.items()):
    district_ballots = grouped.get(district_name, [])
    vote_counts = _count_first_choice_votes(district_ballots)

    if vote_counts:
      winner = _pick_winner({p: c for p, c in vote_counts.items()})
      # Winner-takes-all: the district winner gets every seat.
      seats = {winner: num_seats}
    else:
      winner = "none"
      seats = {}

    district_results.append(
        DistrictResult(
            district_name=district_name,
            seats_available=num_seats,
            vote_counts=vote_counts,
            seats=seats,
            winner=winner,
        )
    )

    for party, s in seats.items():
      total_seats[party] = total_seats.get(party, 0) + s

  governing = _pick_winner(total_seats)
  logging.info("FPTP election complete.  Governing party: %s", governing)

  return ElectionResult(
      voting_system="fptp",
      district_results=district_results,
      total_seats=total_seats,
      governing_party=governing,
  )


# ---------------------------------------------------------------------------
# SMDP — Single-Member District Plurality
# ---------------------------------------------------------------------------


def smdp(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
) -> ElectionResult:
  """Single-Member District Plurality: one seat per district, plurality wins.

  Identical to FPTP except that every district is forced to exactly **one**
  seat regardless of the values in ``district_seats``.  The candidate
  (party) with the most first-choice votes wins that single seat.

  Args:
    ballots: All voter ballots across all districts.
    district_seats: ``{district_name: num_seats}`` — seat counts are
      overridden to 1 for every district.

  Returns:
    An ``ElectionResult`` with per-district breakdowns.
  """
  # Force every district to exactly one seat.
  single_seat_map = {d: 1 for d in district_seats}

  grouped = _group_ballots_by_district(ballots)
  district_results: List[DistrictResult] = []
  total_seats: Dict[str, int] = {}

  for district_name in sorted(single_seat_map):
    district_ballots = grouped.get(district_name, [])
    vote_counts = _count_first_choice_votes(district_ballots)

    if vote_counts:
      winner = _pick_winner({p: c for p, c in vote_counts.items()})
      seats = {winner: 1}
    else:
      winner = "none"
      seats = {}

    district_results.append(
        DistrictResult(
            district_name=district_name,
            seats_available=1,
            vote_counts=vote_counts,
            seats=seats,
            winner=winner,
        )
    )

    for party, s in seats.items():
      total_seats[party] = total_seats.get(party, 0) + s

  governing = _pick_winner(total_seats)
  logging.info("SMDP election complete.  Governing party: %s", governing)

  return ElectionResult(
      voting_system="smdp",
      district_results=district_results,
      total_seats=total_seats,
      governing_party=governing,
  )


# ---------------------------------------------------------------------------
# Alternative Vote (Instant-Runoff Voting)
# ---------------------------------------------------------------------------


def _irv_winner(
    ballots: List[VoterBallot],
    candidates: List[str] | None = None,
) -> tuple[str, Dict[str, int]]:
  """Run instant-runoff elimination and return (winner, final_round_counts).

  Each round:
    1. Count first-preference votes among remaining candidates.
    2. If a candidate has an absolute majority (> 50 %), they win.
    3. Otherwise, eliminate the candidate with the fewest votes (ties
       broken alphabetically — the *last* alphabetically is eliminated)
       and redistribute their ballots to the next-preferred remaining
       candidate.

  The function uses iterative elimination rather than recursion so that
  stack depth is bounded regardless of the number of candidates.

  Args:
    ballots: Ballots for a single district.
    candidates: Candidate set to consider.  ``None`` means derive from
      all rankings in ``ballots``.

  Returns:
    ``(winner_name, first_choice_counts_of_final_round)``
  """
  if candidates is None:
    candidates_set: set[str] = set()
    for b in ballots:
      candidates_set.update(b.ranking)
    remaining = sorted(candidates_set)
  else:
    remaining = sorted(candidates)

  # Deep-copy rankings so we can mutate without affecting the caller.
  active_rankings: List[List[str]] = [
      [c for c in b.ranking if c in remaining] for b in ballots
  ]

  while True:
    # Count first preferences.
    counts: Dict[str, int] = {c: 0 for c in remaining}
    for ranking in active_rankings:
      if ranking:
        counts[ranking[0]] = counts.get(ranking[0], 0) + 1

    total_votes = sum(counts.values())
    if total_votes == 0:
      return ("none", counts)

    # Check for absolute majority.
    for candidate in sorted(remaining):
      if counts[candidate] > total_votes / 2:
        return (candidate, counts)

    # If only one candidate remains, they win.
    if len(remaining) <= 1:
      return (remaining[0] if remaining else "none", counts)

    # Eliminate the candidate with the fewest votes.
    # Tie-break: eliminate the one that sorts *last* alphabetically.
    min_votes = min(counts[c] for c in remaining)
    eliminated = sorted(
        [c for c in remaining if counts[c] == min_votes]
    )[-1]

    remaining = [c for c in remaining if c != eliminated]

    # Redistribute: strip eliminated candidate from every ranking.
    for ranking in active_rankings:
      while eliminated in ranking:
        ranking.remove(eliminated)


def alternative_vote(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
) -> ElectionResult:
  """Alternative Vote (Instant-Runoff Voting): single seat per district.

  In each district the winner is determined by iterative elimination of
  the weakest candidate and redistribution of their ballots until one
  candidate achieves an absolute majority.  Each district is awarded
  exactly **one** seat (the AV method is inherently single-winner).

  Args:
    ballots: All voter ballots across all districts.
    district_seats: ``{district_name: num_seats}`` — seat counts are
      overridden to 1 (AV is a single-winner method).

  Returns:
    An ``ElectionResult`` with per-district breakdowns.
  """
  grouped = _group_ballots_by_district(ballots)
  district_results: List[DistrictResult] = []
  total_seats: Dict[str, int] = {}

  for district_name in sorted(district_seats):
    district_ballots = grouped.get(district_name, [])

    if district_ballots:
      winner, vote_counts = _irv_winner(district_ballots)
      seats = {winner: 1} if winner != "none" else {}
    else:
      winner = "none"
      vote_counts = {}
      seats = {}

    district_results.append(
        DistrictResult(
            district_name=district_name,
            seats_available=1,
            vote_counts=vote_counts,
            seats=seats,
            winner=winner,
        )
    )

    for party, s in seats.items():
      total_seats[party] = total_seats.get(party, 0) + s

  governing = _pick_winner(total_seats)
  logging.info(
      "Alternative Vote election complete.  Governing party: %s", governing
  )

  return ElectionResult(
      voting_system="alternative_vote",
      district_results=district_results,
      total_seats=total_seats,
      governing_party=governing,
  )


# ---------------------------------------------------------------------------
# D'Hondt proportional method
# ---------------------------------------------------------------------------


def dhondt(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
) -> ElectionResult:
  """D'Hondt proportional seat allocation per district.

  For each district, seats are allocated one at a time.  At each step the
  party with the highest *quotient* ``votes / (seats_already_won + 1)``
  receives the next seat.  Ties are broken alphabetically.

  Args:
    ballots: All voter ballots across all districts.
    district_seats: ``{district_name: num_seats}`` from
      ``allocate_district_seats``.

  Returns:
    An ``ElectionResult`` with per-district breakdowns.
  """
  grouped = _group_ballots_by_district(ballots)
  district_results: List[DistrictResult] = []
  total_seats: Dict[str, int] = {}

  for district_name, num_seats in sorted(district_seats.items()):
    district_ballots = grouped.get(district_name, [])
    vote_counts = _count_first_choice_votes(district_ballots)

    seats: Dict[str, int] = {}

    if vote_counts:
      for _ in range(num_seats):
        # Compute quotients: votes / (seats_already + 1)
        quotients = {
            party: votes / (seats.get(party, 0) + 1)
            for party, votes in vote_counts.items()
        }
        # Pick the party with the highest quotient (alphabetical tie-break).
        best_party = max(
            sorted(quotients.keys()),
            key=lambda p: quotients[p],
        )
        seats[best_party] = seats.get(best_party, 0) + 1

    winner = _pick_winner(seats) if seats else "none"

    district_results.append(
        DistrictResult(
            district_name=district_name,
            seats_available=num_seats,
            vote_counts=vote_counts,
            seats=seats,
            winner=winner,
        )
    )

    for party, s in seats.items():
      total_seats[party] = total_seats.get(party, 0) + s

  governing = _pick_winner(total_seats)
  logging.info("D'Hondt election complete.  Governing party: %s", governing)

  return ElectionResult(
      voting_system="dhondt",
      district_results=district_results,
      total_seats=total_seats,
      governing_party=governing,
  )


# ---------------------------------------------------------------------------
# Hare quota — Largest Remainders method
# ---------------------------------------------------------------------------


def _hare_allocate(
    vote_counts: Dict[str, int], num_seats: int,
) -> Dict[str, int]:
  """Allocate seats using the Hare quota with largest remainders.

  Algorithm:
    1. Compute the Hare quota: ``total_votes / num_seats``.
    2. Each party receives ``floor(votes / quota)`` automatic seats.
    3. Remaining seats are given one-at-a-time to the parties with the
       largest fractional remainders (ties broken alphabetically).

  Args:
    vote_counts: ``{party: first_choice_votes}``.
    num_seats: Number of seats to fill.

  Returns:
    ``{party: seats_won}``
  """
  if not vote_counts or num_seats <= 0:
    return {}

  total_votes = sum(vote_counts.values())
  if total_votes == 0:
    return {}

  quota = total_votes / num_seats

  # Step 1 — automatic seats from full quotas.
  seats: Dict[str, int] = {}
  remainders: Dict[str, float] = {}
  for party, votes in vote_counts.items():
    full = int(votes / quota)  # floor division
    seats[party] = full
    remainders[party] = (votes / quota) - full

  # Step 2 — distribute remaining seats by largest remainder.
  seats_allocated = sum(seats.values())
  seats_left = num_seats - seats_allocated

  # Sort by remainder desc, then alphabetically for tie-break.
  ranked = sorted(
      remainders.keys(),
      key=lambda p: (-remainders[p], p),
  )

  for i in range(min(seats_left, len(ranked))):
    seats[ranked[i]] = seats.get(ranked[i], 0) + 1

  # Remove parties with zero seats for cleanliness.
  return {p: s for p, s in seats.items() if s > 0}


def hare(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
) -> ElectionResult:
  """Hare quota proportional seat allocation (largest remainders) per district.

  For each district:
    1. Compute the Hare quota ``Q = total_votes / seats``.
    2. Award each party ``floor(votes / Q)`` seats automatically.
    3. Award remaining seats to parties with the largest fractional
       remainders, one at a time.

  Args:
    ballots: All voter ballots across all districts.
    district_seats: ``{district_name: num_seats}`` from
      ``allocate_district_seats``.

  Returns:
    An ``ElectionResult`` with per-district breakdowns.
  """
  grouped = _group_ballots_by_district(ballots)
  district_results: List[DistrictResult] = []
  total_seats: Dict[str, int] = {}

  for district_name, num_seats in sorted(district_seats.items()):
    district_ballots = grouped.get(district_name, [])
    vote_counts = _count_first_choice_votes(district_ballots)

    seats = _hare_allocate(vote_counts, num_seats)
    winner = _pick_winner(seats) if seats else "none"

    district_results.append(
        DistrictResult(
            district_name=district_name,
            seats_available=num_seats,
            vote_counts=vote_counts,
            seats=seats,
            winner=winner,
        )
    )

    for party, s in seats.items():
      total_seats[party] = total_seats.get(party, 0) + s

  governing = _pick_winner(total_seats)
  logging.info("Hare election complete.  Governing party: %s", governing)

  return ElectionResult(
      voting_system="hare",
      district_results=district_results,
      total_seats=total_seats,
      governing_party=governing,
  )


# ---------------------------------------------------------------------------
# Sainte-Laguë proportional method
# ---------------------------------------------------------------------------


def sainte_lague(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
) -> ElectionResult:
  """Sainte-Laguë (Webster) proportional seat allocation per district.

  Like D'Hondt, seats are allocated one at a time via highest quotients.
  The key difference is the divisor sequence: Sainte-Laguë uses
  **odd numbers** ``(1, 3, 5, 7, …)`` i.e. ``2*seats_already_won + 1``,
  whereas D'Hondt uses ``seats_already_won + 1``.  This produces a more
  proportional result that is less biased toward large parties.

  Ties are broken alphabetically.

  Args:
    ballots: All voter ballots across all districts.
    district_seats: ``{district_name: num_seats}`` from
      ``allocate_district_seats``.

  Returns:
    An ``ElectionResult`` with per-district breakdowns.
  """
  grouped = _group_ballots_by_district(ballots)
  district_results: List[DistrictResult] = []
  total_seats: Dict[str, int] = {}

  for district_name, num_seats in sorted(district_seats.items()):
    district_ballots = grouped.get(district_name, [])
    vote_counts = _count_first_choice_votes(district_ballots)

    seats: Dict[str, int] = {}

    if vote_counts:
      for _ in range(num_seats):
        # Sainte-Laguë divisor: 2 * seats_already + 1  →  1, 3, 5, …
        quotients = {
            party: votes / (2 * seats.get(party, 0) + 1)
            for party, votes in vote_counts.items()
        }
        best_party = max(
            sorted(quotients.keys()),
            key=lambda p: quotients[p],
        )
        seats[best_party] = seats.get(best_party, 0) + 1

    winner = _pick_winner(seats) if seats else "none"

    district_results.append(
        DistrictResult(
            district_name=district_name,
            seats_available=num_seats,
            vote_counts=vote_counts,
            seats=seats,
            winner=winner,
        )
    )

    for party, s in seats.items():
      total_seats[party] = total_seats.get(party, 0) + s

  governing = _pick_winner(total_seats)
  logging.info(
      "Sainte-Laguë election complete.  Governing party: %s", governing
  )

  return ElectionResult(
      voting_system="sainte_lague",
      district_results=district_results,
      total_seats=total_seats,
      governing_party=governing,
  )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SYSTEMS = {
    "fptp": fptp,
    "smdp": smdp,
    "alternative_vote": alternative_vote,
    "dhondt": dhondt,
    "hare": hare,
    "sainte_lague": sainte_lague,
}


def run_election(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
    system: str = "fptp",
) -> ElectionResult:
  """Run an election using the specified voting system.

  Args:
    ballots: All voter ballots.
    district_seats: ``{district_name: num_seats}``.
    system: Voting system name — one of ``"fptp"``, ``"smdp"``,
      ``"alternative_vote"``, ``"dhondt"``, ``"hare"``, or
      ``"sainte_lague"``.

  Returns:
    An ``ElectionResult``.

  Raises:
    ValueError: If ``system`` is not a recognised voting system.
  """
  fn = _SYSTEMS.get(system.lower())
  if fn is None:
    raise ValueError(
        f"Unknown voting system '{system}'.  "
        f"Available: {sorted(_SYSTEMS.keys())}"
    )
  logging.info(
      "Running election: system=%s, districts=%d, ballots=%d",
      system,
      len(district_seats),
      len(ballots),
  )
  return fn(ballots, district_seats)

