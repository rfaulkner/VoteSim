"""Voting system implementations for district-based seat allocation.

Provides two electoral systems:
1. **First Past The Post (FPTP)** — winner-takes-all in each district.
2. **D'Hondt proportional** — seats allocated proportionally per district.

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
# Dispatcher
# ---------------------------------------------------------------------------

_SYSTEMS = {
    "fptp": fptp,
    "dhondt": dhondt,
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
    system: Voting system name — ``"fptp"`` or ``"dhondt"``.

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

