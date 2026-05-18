"""Voting system implementations for district-based seat allocation.

Provides six electoral systems:

*Majoritarian:*
1. **Party Block Vote (SNTV)** — winner-takes-all in each district.
2. **First Past The Post (FPTP)** — like Party Block Vote but
   fixed at one seat per district.
3. **Instant-Runoff Voting (IRV)** — iterative elimination of the
   weakest candidate; the first candidate to reach an absolute majority
   wins the single district seat.
4. **Two-Round System (TRS)** — if no party wins an outright majority
   in round 1, a runoff between the top two determines the winner.

*Proportional:*
4. **D'Hondt** — highest-averages method with divisors 1, 2, 3, …
5. **Sainte-Laguë** — highest-averages method with odd divisors
   1, 3, 5, …
6. **Single Transferable Vote (STV)** — quota-based proportional
   method with surplus transfer and elimination rounds.

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

MIN_SEATS = 4
MAX_SEATS = 15


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
# SNTV — Party Block Vote (formerly FPTP)
# ---------------------------------------------------------------------------


def sntv(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
) -> ElectionResult:
  """Party Block Vote (SNTV): winner takes all seats in each district.

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
  logging.info("SNTV election complete.  Governing party: %s", governing)

  return ElectionResult(
      voting_system="sntv",
      district_results=district_results,
      total_seats=total_seats,
      governing_party=governing,
  )


# ---------------------------------------------------------------------------
# FPTP — First Past The Post
# ---------------------------------------------------------------------------


def fptp(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
) -> ElectionResult:
  """First Past The Post: one seat per district, plurality wins.

  Identical to Party Block Vote except that every district is forced to exactly **one**
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
  logging.info("FPTP election complete.  Governing party: %s", governing)

  return ElectionResult(
      voting_system="fptp",
      district_results=district_results,
      total_seats=total_seats,
      governing_party=governing,
  )


# ---------------------------------------------------------------------------
# Instant-Runoff Voting (IRV)
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
  """Instant-Runoff Voting (IRV): single seat per district.

  In each district the winner is determined by iterative elimination of
  the weakest candidate and redistribution of their ballots until one
  candidate achieves an absolute majority.  Each district is awarded
  exactly **one** seat (the IRV method is inherently single-winner).

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
      "Instant-Runoff Voting election complete.  Governing party: %s", governing
  )

  return ElectionResult(
      voting_system="alternative_vote",
      district_results=district_results,
      total_seats=total_seats,
      governing_party=governing,
  )


# ---------------------------------------------------------------------------
# TRS — Two-Round System
# ---------------------------------------------------------------------------


def trs(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
) -> ElectionResult:
  """Two-Round System: majority runoff in each district.

  **Round 1**: Count first-preference votes.  If any party receives
  more than 50 % of the vote, it wins all seats in the district.

  **Round 2 (runoff)**: If no party has an outright majority, only
  the top two parties advance.  Each ballot is reassigned to
  whichever of the two finalists appears highest in its ranking.
  The finalist with the most reassigned ballots wins all seats.

  This is a majoritarian system — the winner takes every seat in
  the district, regardless of the number of seats available.

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
    total_votes = sum(vote_counts.values()) if vote_counts else 0

    if not vote_counts or total_votes == 0:
      district_results.append(
          DistrictResult(
              district_name=district_name,
              seats_available=num_seats,
              vote_counts={},
              seats={},
              winner="none",
          )
      )
      continue

    # Round 1: check for outright majority.
    majority_threshold = total_votes / 2
    round1_winner = None
    for party in sorted(vote_counts, key=lambda p: (-vote_counts[p], p)):
      if vote_counts[party] > majority_threshold:
        round1_winner = party
        break

    if round1_winner:
      seats = {round1_winner: num_seats}
      winner = round1_winner
    else:
      # Round 2: runoff between top two parties.
      ranked = sorted(
          vote_counts.keys(), key=lambda p: (-vote_counts[p], p)
      )
      finalist_a, finalist_b = ranked[0], ranked[1]
      finalists = {finalist_a, finalist_b}

      # Reassign each ballot to whichever finalist ranks higher.
      runoff_counts: Dict[str, int] = {finalist_a: 0, finalist_b: 0}
      for b in district_ballots:
        for party in b.ranking:
          if party in finalists:
            runoff_counts[party] += 1
            break

      # Winner of the runoff (ties broken alphabetically).
      if runoff_counts[finalist_a] >= runoff_counts[finalist_b]:
        winner = finalist_a
      else:
        winner = finalist_b

      # Override vote_counts with runoff counts for reporting.
      vote_counts = runoff_counts
      seats = {winner: num_seats}

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
      "Two-Round System election complete.  Governing party: %s", governing
  )

  return ElectionResult(
      voting_system="trs",
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
# STV — Single Transferable Vote
# ---------------------------------------------------------------------------


def _stv_allocate(
    ballots: List[VoterBallot], num_seats: int,
) -> tuple[Dict[str, int], Dict[str, int]]:
  """Allocate seats using STV with the Droop quota.

  Algorithm:
    1. Compute the Droop quota: ``floor(total_votes / (seats + 1)) + 1``.
    2. Count first-preference votes for each remaining party.
    3. If any party meets or exceeds the quota, it wins a seat.
       Its surplus (``votes - quota``) is redistributed to next
       preferences at a fractional transfer value
       ``surplus / votes``.
    4. If no party meets the quota, eliminate the party with the
       fewest votes and redistribute its ballots at full value.
    5. Repeat until all seats are filled or no parties remain.

  Because this simulation treats parties (not individual candidates)
  as the unit of election, a party can win multiple seats.

  Args:
    ballots: Ballots for a single district.
    num_seats: Number of seats to fill.

  Returns:
    ``(seats_dict, vote_counts)`` where ``seats_dict`` is
    ``{party: seats_won}`` and ``vote_counts`` is the first-round
    first-choice counts (for reporting).
  """
  if not ballots or num_seats <= 0:
    return {}, {}

  # Each ballot carries a weight (starts at 1.0).
  # We track (weight, ranking_of_remaining_parties).
  active: List[tuple[float, List[str]]] = [
      (1.0, list(b.ranking)) for b in ballots
  ]

  total_votes = len(ballots)
  quota = total_votes // (num_seats + 1) + 1

  seats: Dict[str, int] = {}
  seats_filled = 0
  remaining_parties: set[str] = set()
  for b in ballots:
    remaining_parties.update(b.ranking)

  # Record first-round counts for reporting.
  first_round_counts: Dict[str, int] = {}
  for b in ballots:
    if b.ranking:
      first_round_counts[b.ranking[0]] = (
          first_round_counts.get(b.ranking[0], 0) + 1
      )

  max_rounds = len(remaining_parties) * num_seats + 10  # safety bound

  for _ in range(max_rounds):
    if seats_filled >= num_seats or not remaining_parties:
      break

    # Count weighted first-preference votes.
    counts: Dict[str, float] = {p: 0.0 for p in remaining_parties}
    for weight, ranking in active:
      if ranking:
        top = ranking[0]
        if top in remaining_parties:
          counts[top] += weight

    # Check for parties meeting the quota.
    elected_this_round = []
    for party in sorted(remaining_parties):
      if counts.get(party, 0) >= quota:
        elected_this_round.append(party)

    if elected_this_round:
      for party in elected_this_round:
        seats[party] = seats.get(party, 0) + 1
        seats_filled += 1
        if seats_filled >= num_seats:
          break

        # Transfer surplus.
        party_votes = counts[party]
        surplus = party_votes - quota
        if surplus > 0 and party_votes > 0:
          transfer_value = surplus / party_votes
          new_active = []
          for weight, ranking in active:
            if ranking and ranking[0] == party:
              # Remove elected party and scale weight.
              new_ranking = [p for p in ranking[1:] if p in remaining_parties]
              new_active.append((weight * transfer_value, new_ranking))
            else:
              new_active.append((weight, ranking))
          active = new_active
        else:
          # No surplus — just remove from ballots.
          active = [
              (w, [p for p in r if p != party])
              for w, r in active
          ]

        # Party can win another seat if it still has surplus,
        # but remove it from this round's re-election.
        remaining_parties.discard(party)

      if seats_filled >= num_seats:
        break
    else:
      # No party met quota — eliminate the weakest.
      if not counts:
        break
      min_votes = min(counts.values())
      # Tie-break: eliminate last alphabetically.
      weakest = sorted(
          [p for p in remaining_parties if counts.get(p, 0) == min_votes]
      )[-1]
      remaining_parties.discard(weakest)

      # Redistribute eliminated party's ballots at full weight.
      active = [
          (w, [p for p in r if p != weakest])
          for w, r in active
      ]

  # If seats remain unfilled, give them to highest remaining counts.
  if seats_filled < num_seats and remaining_parties:
    counts_final: Dict[str, float] = {p: 0.0 for p in remaining_parties}
    for weight, ranking in active:
      if ranking and ranking[0] in remaining_parties:
        counts_final[ranking[0]] += weight
    ranked_remaining = sorted(
        remaining_parties, key=lambda p: (-counts_final.get(p, 0), p)
    )
    for party in ranked_remaining:
      if seats_filled >= num_seats:
        break
      seats[party] = seats.get(party, 0) + 1
      seats_filled += 1

  return {p: s for p, s in seats.items() if s > 0}, first_round_counts


def stv(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
) -> ElectionResult:
  """Single Transferable Vote: quota-based proportional seat allocation.

  For each district:
    1. Compute the Droop quota ``Q = floor(V / (S + 1)) + 1``.
    2. Elect any party whose weighted first-preference votes meet
       or exceed the quota.  Transfer surplus votes (fractionally)
       to next-preferred parties.
    3. If no party meets the quota, eliminate the weakest party and
       redistribute its ballots at full weight.
    4. Repeat until all seats are filled.

  This method is more proportional than D'Hondt for small district
  magnitudes and rewards cross-party appeal through transfers.

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

    seats, vote_counts = _stv_allocate(district_ballots, num_seats)
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
  logging.info("STV election complete.  Governing party: %s", governing)

  return ElectionResult(
      voting_system="stv",
      district_results=district_results,
      total_seats=total_seats,
      governing_party=governing,
  )


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------

_SYSTEMS = {
    "sntv": sntv,
    "fptp": fptp,
    "alternative_vote": alternative_vote,
    "trs": trs,
    "dhondt": dhondt,
    "sainte_lague": sainte_lague,
    "stv": stv,
}


def run_election(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
    system: str = "sntv",
) -> ElectionResult:
  """Run an election using the specified voting system.

  Args:
    ballots: All voter ballots.
    district_seats: ``{district_name: num_seats}``.
    system: Voting system name — one of ``"sntv"``, ``"fptp"``,
      ``"alternative_vote"``, ``"trs"``, ``"dhondt"``,
      ``"sainte_lague"``, or ``"stv"``.

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

