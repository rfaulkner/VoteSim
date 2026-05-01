"""Parliamentary deliberation: Draft → Amend → Vote.

Simulates a motion-based parliamentary debate process among the seated
government formed during the election phase.  The governing party (most
seats) drafts a multi-point bill, opposition parties propose amendments
(removals / additions), and all seated members vote yes/no on the package.

A bill may be considered up to ``max_rounds`` times (default 3).  If no
bill achieves a majority, the result records ``adopted_bill = None``.
"""

from concurrent import futures
from dataclasses import dataclass
from dataclasses import field
import json
import logging
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pathfinder import assistant
from pathfinder import gen
from pathfinder import user

from simulation.policy_generator import PolicyResponse

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class Bill:
  """A multi-point policy bill."""

  points: List[str]
  party: str
  round_number: int

  def to_dict(self) -> Dict[str, Any]:
    return {
        "points": list(self.points),
        "party": self.party,
        "round_number": self.round_number,
    }


@dataclass
class Amendment:
  """Amendments proposed by a single party."""

  party: str
  removals: List[int]  # 0-indexed point indices to remove.
  additions: List[str]  # New points to append.

  def to_dict(self) -> Dict[str, Any]:
    return {
        "party": self.party,
        "removals": list(self.removals),
        "additions": list(self.additions),
    }


@dataclass
class MemberVote:
  """A single seated member's vote."""

  name: str
  party: str
  seat: str
  vote: str  # "yes" or "no"

  def to_dict(self) -> Dict[str, Any]:
    return {
        "name": self.name,
        "party": self.party,
        "seat": self.seat,
        "vote": self.vote,
    }


@dataclass
class VoteRecord:
  """Aggregated vote results for a single round."""

  votes: List[MemberVote]
  yes_count: int
  no_count: int
  passed: bool

  def to_dict(self) -> Dict[str, Any]:
    return {
        "votes": [v.to_dict() for v in self.votes],
        "yes_count": self.yes_count,
        "no_count": self.no_count,
        "passed": self.passed,
    }


@dataclass
class RoundRecord:
  """Record of a single deliberation round."""

  bill: Bill
  amendments: List[Amendment]
  vote_record: VoteRecord

  def to_dict(self) -> Dict[str, Any]:
    return {
        "bill": self.bill.to_dict(),
        "amendments": [a.to_dict() for a in self.amendments],
        "vote_record": self.vote_record.to_dict(),
    }


@dataclass
class DeliberationResult:
  """Final output of the deliberation process."""

  issue: str
  rounds: List[RoundRecord] = field(default_factory=list)
  adopted_bill: Optional[Bill] = None

  def summary(self) -> str:
    """Return a human-readable summary of the deliberation."""
    lines = [
        "--- Deliberation Results ---",
        f"Issue: {self.issue}",
        f"Rounds: {len(self.rounds)}",
    ]

    for i, rnd in enumerate(self.rounds, 1):
      lines.append(f"\n  Round {i}:")
      lines.append(f"    Drafting party: {rnd.bill.party}")
      lines.append("    Bill points:")
      for j, pt in enumerate(rnd.bill.points, 1):
        lines.append(f"      {j}. {pt}")

      if rnd.amendments:
        lines.append("    Amendments:")
        for am in rnd.amendments:
          lines.append(f"      {am.party}:")
          if am.removals:
            lines.append(f"        Strike points: {am.removals}")
          if am.additions:
            for add in am.additions:
              lines.append(f"        + {add}")
      else:
        lines.append("    No amendments proposed.")

      vr = rnd.vote_record
      lines.append(
          f"    Vote: {vr.yes_count} yes / {vr.no_count} no"
          f" — {'PASSED' if vr.passed else 'FAILED'}"
      )

    if self.adopted_bill:
      lines.append("\n  ADOPTED BILL:")
      for j, pt in enumerate(self.adopted_bill.points, 1):
        lines.append(f"    {j}. {pt}")
    else:
      lines.append("\n  No Bill adopted.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

DRAFT_BILL_PROMPT = """\
You are a senior legislative advisor for the {lead_party} party, which holds \
{lead_seats} seats in the current government.

=== SOCIAL ISSUE UNDER DEBATE ===
{issue}

=== GOVERNMENT SEAT ALLOCATION ===
{seat_block}

=== PARTY POLICIES ON THIS ISSUE ===
{policies_block}
{prior_round_block}
=== INSTRUCTIONS ===
Draft a multi-point legislative bill on the social issue above.  The bill \
should reflect the {lead_party} party's policy position while being \
pragmatic enough to attract votes from other parties in the government.

Return ONLY a JSON object with a single key "points" whose value is an \
array of 5-8 concise policy point strings.  Example:
{{"points": ["Point one ...", "Point two ...", ...]}}

Return ONLY valid JSON — no markdown fences, no commentary."""

_PRIOR_ROUND_BLOCK = """
=== PRIOR ROUND (FAILED) ===
The following bill was considered in round {round_number} and FAILED to pass.

Previous bill points:
{prev_bill_points}

Amendments proposed:
{prev_amendments}

Voting record:
{prev_vote_summary}

You MUST draft a NEW bill that takes into account the failed bill, the \
proposed amendments, and the voting record.  Adjust points to gain broader \
support.
"""

AMEND_BILL_PROMPT = """\
You are a parliamentary advisor for the {party} party, which holds \
{party_seats} seats in the current government.

=== SOCIAL ISSUE UNDER DEBATE ===
{issue}

=== GOVERNMENT SEAT ALLOCATION ===
{seat_block}

=== YOUR PARTY'S POLICY ON THIS ISSUE ===
{party_policy}

=== PROPOSED BILL (by {lead_party}) ===
{bill_block}

=== INSTRUCTIONS ===
Review the proposed bill from the perspective of the {party} party's \
policy position.  Propose amendments in two categories:
1. "removals" — a list of 0-indexed point numbers to strike from the bill.
2. "additions" — a list of new point strings to append to the bill.

You may propose no removals and/or no additions if the bill is acceptable.

Return ONLY a JSON object:
{{"removals": [0, 3], "additions": ["New point ...", ...]}}

Return ONLY valid JSON — no markdown fences, no commentary."""

VOTE_BILL_PROMPT = """\
You are {member_name}, a member of parliament representing the {party} \
party.  You hold seat #{seat_number}.

=== SOCIAL ISSUE UNDER DEBATE ===
{issue}

=== GOVERNMENT SEAT ALLOCATION ===
{seat_block}

=== YOUR PARTY'S POLICY ON THIS ISSUE ===
{party_policy}

=== PROPOSED BILL ===
{bill_block}

=== PROPOSED AMENDMENTS ===
{amendments_block}

=== INSTRUCTIONS ===
Based on your party's policy position, the proposed bill, and the \
amendments, cast your vote on whether to ACCEPT this bill.

Return ONLY a JSON object:
{{"name": "{member_name}", "party": "{party}", "seat": "{seat_label}", \
"vote": "yes" or "no"}}

Return ONLY valid JSON — no markdown fences, no commentary."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_llm_wrapping(raw: str) -> str:
  """Remove <think> blocks and markdown fences from LLM output."""
  text = re.sub(r"<think>.*?(</think>|$)", "", raw, flags=re.DOTALL).strip()
  if text.startswith("```"):
    text = text.split("\n", 1)[1]
    text = text.rsplit("```", 1)[0]
    text = text.strip()
  return text


def _parse_json(raw: str, context: str) -> Any:
  """Parse JSON from raw LLM output with stripping and error handling."""
  text = _strip_llm_wrapping(raw)
  if not text:
    logging.warning("Empty LLM output for %s", context)
    return {}
  try:
    return json.loads(text)
  except json.JSONDecodeError as e:
    logging.warning(
        "Failed to parse JSON for %s: %s. Content: %.500s",
        context,
        e,
        text,
    )
    return {}


def _format_seat_block(seat_allocation: Dict[str, int]) -> str:
  """Format seat allocation for prompt injection."""
  lines = []
  for party, seats in sorted(
      seat_allocation.items(), key=lambda x: -x[1]
  ):
    lines.append(f"  {party}: {seats} seat{'s' if seats != 1 else ''}")
  return "\n".join(lines)


def _format_policies_block(
    party_policies: Dict[str, PolicyResponse],
) -> str:
  """Format all party policies for prompt injection."""
  lines = []
  for ideology in sorted(party_policies):
    pr = party_policies[ideology]
    proposals = "; ".join(pr.key_proposals) if pr.key_proposals else "N/A"
    lines.append(
        f"  {pr.party_name} ({pr.ideology}):\n"
        f"    Position: {pr.position_statement}\n"
        f"    Proposals: {proposals}"
    )
  return "\n\n".join(lines)


def _format_bill_block(bill: Bill) -> str:
  """Format a bill's points for prompt injection."""
  lines = []
  for i, pt in enumerate(bill.points):
    lines.append(f"  {i}. {pt}")
  return "\n".join(lines)


def _format_amendments_block(amendments: List[Amendment]) -> str:
  """Format all amendments for prompt injection."""
  if not amendments:
    return "  (No amendments proposed.)"
  lines = []
  for am in amendments:
    lines.append(f"  {am.party}:")
    if am.removals:
      lines.append(f"    Strike points: {am.removals}")
    if am.additions:
      for add in am.additions:
        lines.append(f"    + {add}")
    if not am.removals and not am.additions:
      lines.append("    (No changes proposed.)")
  return "\n".join(lines)


def _get_lead_party(seat_allocation: Dict[str, int]) -> str:
  """Return the party with the most seats (alphabetical tie-break)."""
  if not seat_allocation:
    return "unknown"
  max_seats = max(seat_allocation.values())
  leaders = sorted(p for p, s in seat_allocation.items() if s == max_seats)
  return leaders[0]


def _expand_members(
    seat_allocation: Dict[str, int],
) -> List[Dict[str, str]]:
  """Expand seat allocation into individual member records.

  Returns a list of dicts with keys: ``name``, ``party``, ``seat_label``,
  ``seat_number``.
  """
  members = []
  for party in sorted(seat_allocation):
    num_seats = seat_allocation[party]
    for n in range(1, num_seats + 1):
      members.append({
          "name": f"{party}-member-{n}",
          "party": party,
          "seat_label": f"{party}-seat-{n}",
          "seat_number": str(n),
      })
  return members


# ---------------------------------------------------------------------------
# Phase 1: Draft Bill
# ---------------------------------------------------------------------------


def _draft_bill(
    model: Any,
    issue: str,
    seat_allocation: Dict[str, int],
    party_policies: Dict[str, PolicyResponse],
    prior_round: Optional[RoundRecord] = None,
    round_number: int = 1,
    temperature: float = 0.7,
) -> Bill:
  """Use the LLM to draft a bill on behalf of the governing party."""
  lead_party = _get_lead_party(seat_allocation)
  lead_seats = seat_allocation.get(lead_party, 0)
  seat_block = _format_seat_block(seat_allocation)
  policies_block = _format_policies_block(party_policies)

  # Build the prior-round context block (empty for round 1).
  prior_round_block = ""
  if prior_round is not None:
    prev_bill_points = _format_bill_block(prior_round.bill)
    prev_amendments = _format_amendments_block(prior_round.amendments)
    vr = prior_round.vote_record
    prev_vote_summary = (
        f"  Yes: {vr.yes_count}, No: {vr.no_count} — "
        f"{'PASSED' if vr.passed else 'FAILED'}\n"
    )
    for mv in vr.votes:
      prev_vote_summary += f"    {mv.name} ({mv.party}): {mv.vote}\n"

    prior_round_block = _PRIOR_ROUND_BLOCK.format(
        round_number=prior_round.bill.round_number,
        prev_bill_points=prev_bill_points,
        prev_amendments=prev_amendments,
        prev_vote_summary=prev_vote_summary,
    )

  prompt = DRAFT_BILL_PROMPT.format(
      lead_party=lead_party,
      lead_seats=lead_seats,
      issue=issue,
      seat_block=seat_block,
      policies_block=policies_block,
      prior_round_block=prior_round_block,
  )

  lm = model.copy()
  with user():
    lm += prompt
  with assistant():
    lm += gen(max_tokens=4096, temperature=temperature, name="bill_json")

  data = _parse_json(lm["bill_json"], f"draft_bill round {round_number}")
  points = data.get("points", [])
  if not points:
    logging.warning(
        "LLM returned no bill points in round %d. Using placeholder.",
        round_number,
    )
    points = ["No policy points could be generated."]

  bill = Bill(points=points, party=lead_party, round_number=round_number)
  logging.info(
      "Round %d: %s drafted bill with %d points.",
      round_number,
      lead_party,
      len(bill.points),
  )
  return bill


# ---------------------------------------------------------------------------
# Phase 2: Debate & Amend
# ---------------------------------------------------------------------------


def _propose_single_amendment(
    model: Any,
    party: str,
    party_seats: int,
    issue: str,
    seat_allocation: Dict[str, int],
    party_policies: Dict[str, PolicyResponse],
    bill: Bill,
    temperature: float,
) -> Amendment:
  """Query the LLM for a single party's amendments.  Runs in a thread."""
  lead_party = _get_lead_party(seat_allocation)
  seat_block = _format_seat_block(seat_allocation)
  bill_block = _format_bill_block(bill)

  # Get the party's own policy text.
  pr = party_policies.get(party)
  if pr:
    proposals = "; ".join(pr.key_proposals) if pr.key_proposals else "N/A"
    party_policy = (
        f"Position: {pr.position_statement}\n"
        f"Proposals: {proposals}"
    )
  else:
    party_policy = "(No policy on file for this party.)"

  prompt = AMEND_BILL_PROMPT.format(
      party=party,
      party_seats=party_seats,
      issue=issue,
      seat_block=seat_block,
      party_policy=party_policy,
      lead_party=lead_party,
      bill_block=bill_block,
  )

  lm = model.copy()
  with user():
    lm += prompt
  with assistant():
    lm += gen(max_tokens=2048, temperature=temperature, name="amend_json")

  data = _parse_json(lm["amend_json"], f"amendments from {party}")
  removals = data.get("removals", [])
  additions = data.get("additions", [])

  # Validate removals: must be valid indices.
  valid_removals = [
      r for r in removals
      if isinstance(r, int) and 0 <= r < len(bill.points)
  ]
  if len(valid_removals) != len(removals):
    logging.warning(
        "Party %s proposed invalid removal indices.  Keeping only valid: %s",
        party,
        valid_removals,
    )

  amendment = Amendment(
      party=party,
      removals=valid_removals,
      additions=[str(a) for a in additions],
  )
  logging.info(
      "Party %s proposed %d removals, %d additions.",
      party,
      len(amendment.removals),
      len(amendment.additions),
  )
  return amendment


def _propose_amendments(
    model: Any,
    issue: str,
    seat_allocation: Dict[str, int],
    party_policies: Dict[str, PolicyResponse],
    bill: Bill,
    temperature: float = 0.7,
) -> List[Amendment]:
  """Gather amendments from all non-governing parties in seat order."""
  lead_party = _get_lead_party(seat_allocation)

  # Descending seat order, excluding the governing party.
  opposition = sorted(
      ((p, s) for p, s in seat_allocation.items() if p != lead_party),
      key=lambda x: -x[1],
  )

  amendments: List[Amendment] = []
  for party, seats in opposition:
    try:
      am = _propose_single_amendment(
          model=model,
          party=party,
          party_seats=seats,
          issue=issue,
          seat_allocation=seat_allocation,
          party_policies=party_policies,
          bill=bill,
          temperature=temperature,
      )
      amendments.append(am)
    except Exception:  # pylint: disable=broad-except
      logging.exception("Failed to get amendments from %s", party)

  logging.info(
      "Collected amendments from %d / %d opposition parties.",
      len(amendments),
      len(opposition),
  )
  return amendments


# ---------------------------------------------------------------------------
# Phase 3: Vote
# ---------------------------------------------------------------------------


def _vote_single_member(
    model: Any,
    member: Dict[str, str],
    issue: str,
    seat_allocation: Dict[str, int],
    party_policies: Dict[str, PolicyResponse],
    bill: Bill,
    amendments: List[Amendment],
    temperature: float,
) -> MemberVote:
  """Query a single member for their vote.  Runs in a thread."""
  seat_block = _format_seat_block(seat_allocation)
  bill_block = _format_bill_block(bill)
  amendments_block = _format_amendments_block(amendments)

  party = member["party"]
  pr = party_policies.get(party)
  if pr:
    proposals = "; ".join(pr.key_proposals) if pr.key_proposals else "N/A"
    party_policy = (
        f"Position: {pr.position_statement}\n"
        f"Proposals: {proposals}"
    )
  else:
    party_policy = "(No policy on file for this party.)"

  prompt = VOTE_BILL_PROMPT.format(
      member_name=member["name"],
      party=party,
      seat_number=member["seat_number"],
      issue=issue,
      seat_block=seat_block,
      party_policy=party_policy,
      bill_block=bill_block,
      amendments_block=amendments_block,
      seat_label=member["seat_label"],
  )

  lm = model.copy()
  with user():
    lm += prompt
  with assistant():
    lm += gen(max_tokens=512, temperature=temperature, name="vote_json")

  data = _parse_json(lm["vote_json"], f"vote from {member['name']}")

  vote_str = str(data.get("vote", "no")).lower().strip()
  if vote_str not in ("yes", "no"):
    logging.warning(
        "Invalid vote '%s' from %s. Defaulting to 'no'.",
        vote_str,
        member["name"],
    )
    vote_str = "no"

  return MemberVote(
      name=member["name"],
      party=party,
      seat=member["seat_label"],
      vote=vote_str,
  )


def _vote_on_bill(
    model: Any,
    issue: str,
    seat_allocation: Dict[str, int],
    party_policies: Dict[str, PolicyResponse],
    bill: Bill,
    amendments: List[Amendment],
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
) -> VoteRecord:
  """Concurrently collect votes from all seated members."""
  members = _expand_members(seat_allocation)
  total = len(members)
  logging.info("Voting: %d members casting ballots...", total)

  workers = max_workers or total

  with futures.ThreadPoolExecutor(max_workers=workers) as pool:
    future_to_member = {
        pool.submit(
            _vote_single_member,
            model,
            member,
            issue,
            seat_allocation,
            party_policies,
            bill,
            amendments,
            temperature,
        ): member
        for member in members
    }

    votes: List[MemberVote] = []
    for future in futures.as_completed(future_to_member):
      member = future_to_member[future]
      try:
        mv = future.result()
        votes.append(mv)
        logging.info("  %s (%s): %s", mv.name, mv.party, mv.vote)
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "Failed to collect vote from %s", member["name"]
        )

  yes_count = sum(1 for v in votes if v.vote == "yes")
  no_count = sum(1 for v in votes if v.vote == "no")
  passed = yes_count > total / 2

  logging.info(
      "Vote result: %d yes / %d no (of %d) — %s",
      yes_count,
      no_count,
      total,
      "PASSED" if passed else "FAILED",
  )

  return VoteRecord(
      votes=votes,
      yes_count=yes_count,
      no_count=no_count,
      passed=passed,
  )


# ---------------------------------------------------------------------------
# Amendment merging
# ---------------------------------------------------------------------------


def _merge_amendments(bill: Bill, amendments: List[Amendment]) -> Bill:
  """Apply amendments to a bill: remove stricken points, append additions.

  Removals are processed highest-index-first to preserve index stability.
  Then all additions from every party are appended in party order.

  Returns a new ``Bill`` instance with the updated points.
  """
  points = list(bill.points)

  # Collect all removal indices (deduplicated).
  all_removals = set()
  for am in amendments:
    for idx in am.removals:
      if 0 <= idx < len(points):
        all_removals.add(idx)

  # Remove in reverse order to maintain index validity.
  for idx in sorted(all_removals, reverse=True):
    removed = points.pop(idx)
    logging.info("Merged: removed point %d: %s", idx, removed)

  # Append additions from each party in order.
  for am in amendments:
    for addition in am.additions:
      points.append(addition)
      logging.info("Merged: added point from %s: %s", am.party, addition)

  return Bill(
      points=points,
      party=bill.party,
      round_number=bill.round_number,
  )


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_deliberation(
    model: Any,
    issue: str,
    seat_allocation: Dict[str, int],
    party_policies: Dict[str, PolicyResponse],
    temperature: float = 0.7,
    max_rounds: int = 3,
    max_workers: Optional[int] = None,
) -> DeliberationResult:
  """Run the full parliamentary deliberation process.

  Phases per round:
    1. **Draft** — The governing party drafts a bill.
    2. **Amend** — Opposition parties propose amendments (sequential,
       descending seat order).
    3. **Vote** — All seated members vote concurrently.

  If the bill passes, amendments are merged into the final text.
  If not, the process repeats up to ``max_rounds`` times.

  Args:
    model: A loaded PathFinder model instance (shared across threads via
      ``model.copy()``).
    issue: The social issue text under debate.
    seat_allocation: ``{party_ideology: num_seats}`` from the election.
    party_policies: ``{party_ideology: PolicyResponse}`` from phase 4.
    temperature: LLM sampling temperature.
    max_rounds: Maximum number of bill consideration attempts.
    max_workers: Max concurrent LLM calls during the vote phase.

  Returns:
    A ``DeliberationResult`` with the full record and (optionally) an
    adopted bill.
  """
  logging.info(
      "=== DELIBERATION START === Issue: %s | Parties: %s",
      issue[:80],
      list(seat_allocation.keys()),
  )

  result = DeliberationResult(issue=issue)
  prior_round: Optional[RoundRecord] = None

  for round_num in range(1, max_rounds + 1):
    logging.info("--- Deliberation round %d / %d ---", round_num, max_rounds)

    # Phase 1: Draft.
    bill = _draft_bill(
        model=model,
        issue=issue,
        seat_allocation=seat_allocation,
        party_policies=party_policies,
        prior_round=prior_round,
        round_number=round_num,
        temperature=temperature,
    )

    # Phase 2: Amend.
    amendments = _propose_amendments(
        model=model,
        issue=issue,
        seat_allocation=seat_allocation,
        party_policies=party_policies,
        bill=bill,
        temperature=temperature,
    )

    # Phase 3: Vote.
    vote_record = _vote_on_bill(
        model=model,
        issue=issue,
        seat_allocation=seat_allocation,
        party_policies=party_policies,
        bill=bill,
        amendments=amendments,
        temperature=temperature,
        max_workers=max_workers,
    )

    round_record = RoundRecord(
        bill=bill,
        amendments=amendments,
        vote_record=vote_record,
    )
    result.rounds.append(round_record)

    if vote_record.passed:
      # Merge amendments into the bill.
      adopted = _merge_amendments(bill, amendments)
      result.adopted_bill = adopted
      logging.info(
          "Round %d: Bill PASSED with %d yes / %d no.  "
          "Adopted bill has %d points.",
          round_num,
          vote_record.yes_count,
          vote_record.no_count,
          len(adopted.points),
      )
      break
    else:
      logging.info(
          "Round %d: Bill FAILED with %d yes / %d no.  "
          "Proceeding to next round.",
          round_num,
          vote_record.yes_count,
          vote_record.no_count,
      )
      prior_round = round_record

  if result.adopted_bill is None:
    logging.info(
        "No bill adopted after %d rounds for issue: %s",
        max_rounds,
        issue[:80],
    )

  logging.info("=== DELIBERATION END ===")
  return result

