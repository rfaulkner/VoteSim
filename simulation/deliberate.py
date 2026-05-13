"""Parliamentary deliberation: Draft → Vote.

Simulates a motion-based parliamentary debate process among the seated
government formed during the election phase.  The governing coalition
(largest party + politically aligned partner) drafts a multi-point bill,
and all seated members vote yes/no on the package.

A bill may be considered up to ``max_rounds`` times (default 5).  If no
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
from typing import Tuple

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

COALITION_DRAFT_BILL_PROMPT = """\
You are a senior legislative advisor drafting a bill on behalf of a \
governing coalition.

=== SOCIAL ISSUE UNDER DEBATE ===
{issue}

=== FULL GOVERNMENT SEAT ALLOCATION ===
{seat_block}

=== GOVERNING COALITION ===
{coalition_block}

=== COALITION PARTIES' POLICIES ON THIS ISSUE ===
{coalition_policies_block}
{prior_round_block}
=== VOTING RULES ===
There are {total_seats} total seats in parliament.  A bill passes only if \
it receives MORE than half the votes, i.e. at least {majority_votes} yes \
votes.  Every seated member (coalition AND opposition) votes.

=== INSTRUCTIONS ===
Draft a multi-point legislative bill that proportionally reflects the \
platforms of ALL coalition parties.  The party with the most seats in the \
coalition should have the most influence on the bill's content, but the \
bill must incorporate key proposals from every coalition partner in \
proportion to their seat share.

Do NOT alienate any coalition member's voter base.  The bill should be a \
genuine synthesis, not a dominant-party bill with token concessions.

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
support from both coalition members and potential opposition allies.
"""

AMEND_BILL_PROMPT = """\
You are a parliamentary advisor for the {party} party, which holds \
{party_seats} seats in the current government.  Your party is in the \
OPPOSITION — not part of the governing coalition.

=== SOCIAL ISSUE UNDER DEBATE ===
{issue}

=== GOVERNMENT SEAT ALLOCATION ===
{seat_block}

=== GOVERNING COALITION ===
{coalition_block}

=== YOUR PARTY'S POLICY ON THIS ISSUE ===
{party_policy}

=== PROPOSED BILL (by the governing coalition) ===
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

COALITION_EVALUATE_AMENDMENT_PROMPT = """\
You represent the governing coalition ({coalition_parties_list}).

=== SOCIAL ISSUE UNDER DEBATE ===
{issue}

=== COALITION PARTIES' POLICIES ===
{coalition_policies_block}

=== CURRENT BILL ===
{bill_block}

=== PROPOSED AMENDMENT FROM {amending_party} (opposition) ===
{amendment_block}

=== INSTRUCTIONS ===
Evaluate whether adopting this amendment would strengthen the bill \
without contradicting any coalition partner's core platform.  \
Do not adopt amendments that fundamentally undermine the coalition \
agreement, but consider adopting ones that improve the bill or \
broaden its appeal.

Return ONLY a JSON object:
{{"adopt": true or false, "reason": "brief explanation"}}

Return ONLY valid JSON — no markdown fences, no commentary."""

VOTE_BILL_PROMPT = """\
You are {member_name}, a member of parliament representing the {party} \
party.  You hold seat #{seat_number}.

=== SOCIAL ISSUE UNDER DEBATE ===
{issue}

=== GOVERNMENT SEAT ALLOCATION ===
{seat_block}
{constituency_block}
=== YOUR PARTY'S POLICY ON THIS ISSUE ===
{party_policy}

=== PROPOSED BILL ===
{bill_block}

=== AMENDMENTS ADOPTED ===
{amendments_block}

=== INSTRUCTIONS ===
Cast your vote considering BOTH your party's policy position AND your \
constituents' interests.  If the bill conflicts with your constituents' \
needs, you may vote against your party line.

Return ONLY a JSON object:
{{"name": "{member_name}", "party": "{party}", "seat": "{seat_label}", \
"vote": "yes" or "no"}}

Return ONLY valid JSON — no markdown fences, no commentary."""


BASELINE_BILL_PROMPT = """\
You are an expert, nonpartisan policy analyst.

=== SOCIAL ISSUE ===
{issue}

=== INSTRUCTIONS ===
Draft a comprehensive, multi-point legislative bill that addresses the \
social issue above.  You are not representing any particular party or \
ideology; craft the best policy you can that serves the broadest public \
interest.

Return ONLY a JSON object with a single key "points" whose value is an \
array of 5-8 concise policy point strings.  Example:
{{"points": ["Point one ...", "Point two ...", ...]}}

Return ONLY valid JSON — no markdown fences, no commentary."""


def draft_baseline_bill(
    model: Any,
    issue: str,
    temperature: float = 0.7,
) -> Bill:
  """Prompt the model directly to draft a bill — no deliberation.

  Unlike ``_draft_bill``, this function does not assume a governing
  party, seat allocation, prior rounds, or party policies.  It simply
  asks the model to act as a nonpartisan policy analyst and produce
  legislation for the given issue with no partisan context.

  The resulting ``Bill`` is labelled with party ``"baseline"`` so it
  can be distinguished from deliberation-produced bills.

  Args:
    model: A loaded PathFinder model instance.
    issue: The social issue text to legislate on.
    temperature: LLM sampling temperature.

  Returns:
    A ``Bill`` with ``party="baseline"`` and ``round_number=0``.
  """
  prompt = BASELINE_BILL_PROMPT.format(issue=issue)

  lm = model.copy()
  with user():
    lm += prompt
  with assistant():
    lm += gen(max_tokens=4096, temperature=temperature, name="bill_json")

  data = _parse_json(lm["bill_json"], "baseline_bill")
  points = data.get("points", [])
  if not points:
    logging.warning(
        "LLM returned no points for baseline bill. Using placeholder."
    )
    points = ["No policy points could be generated."]

  bill = Bill(points=points, party="baseline", round_number=0)
  logging.info("Baseline bill drafted with %d points.", len(bill.points))
  return bill


BASELINE_INFORMED_BILL_PROMPT = """\
You are an expert, nonpartisan policy analyst.

=== SOCIAL ISSUE ===
{issue}

=== PARTY POLICIES ON THIS ISSUE ===
{policies_block}

=== INSTRUCTIONS ===
Draft a comprehensive, multi-point legislative bill that addresses the \
social issue above.  You are not representing any particular party.  \
Use the party policies above to craft the best policy you can that \
reflects the broadest public interest.

Return ONLY a JSON object with a single key "points" whose value is an \
array of 5-8 concise policy point strings.  Example:
{{"points": ["Point one ...", "Point two ...", ...]}}

Return ONLY valid JSON — no markdown fences, no commentary."""


def _format_ballots_block(ballots: List) -> str:
  """Format voter ballots for prompt injection."""
  lines = []
  for b in ballots:
    lines.append(
        f"  {b.user_id} ({b.district_name}): "
        f"{' > '.join(b.ranking)}"
    )
  return "\n".join(lines)


def draft_baseline_informed_bill(
    model: Any,
    issue: str,
    party_policies: Dict[str, PolicyResponse],
    ballots: List = None,
    temperature: float = 0.7,
) -> Bill:
  """Draft a bill using party policies — no election or deliberation.

  This baseline gives the model visibility into what parties
  propose, but skips seat allocation (phase 6) and parliamentary
  deliberation (phase 7).  The model must synthesise policy
  directly from the party platforms.

  The resulting ``Bill`` is labelled with party
  ``"baseline_informed"`` so it can be distinguished from other
  bills.

  Args:
    model: A loaded PathFinder model instance.
    issue: The social issue text to legislate on.
    party_policies: ``{ideology: PolicyResponse}`` from the party
      response phase.
    ballots: Unused, kept for call-site compatibility.
    temperature: LLM sampling temperature.

  Returns:
    A ``Bill`` with ``party="baseline_informed"`` and
    ``round_number=0``.
  """
  del ballots  # No longer used.
  policies_block = _format_policies_block(party_policies)

  prompt = BASELINE_INFORMED_BILL_PROMPT.format(
      issue=issue,
      policies_block=policies_block,
  )

  lm = model.copy()
  with user():
    lm += prompt
  with assistant():
    lm += gen(
        max_tokens=4096, temperature=temperature, name="bill_json"
    )

  data = _parse_json(lm["bill_json"], "baseline_informed_bill")
  points = data.get("points", [])
  if not points:
    logging.warning(
        "LLM returned no points for baseline_informed bill."
        " Using placeholder."
    )
    points = ["No policy points could be generated."]

  bill = Bill(
      points=points, party="baseline_informed", round_number=0
  )
  logging.info(
      "Baseline-informed bill drafted with %d points.",
      len(bill.points),
  )
  return bill


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


def _rejoin_prefix(prefix: str, gen_output: str) -> str:
  """Reconstruct JSON from gen output, handling models that echo the prefix."""
  stripped = gen_output.lstrip()
  if stripped.startswith(prefix):
    return stripped
  return prefix + gen_output


def _parse_json(raw: str, context: str) -> Any:
  """Parse JSON from raw LLM output with stripping and error handling."""
  text = _strip_llm_wrapping(raw)
  if not text:
    logging.warning("Empty LLM output for %s", context)
    return {}
  # Deduplicate opening braces/brackets from prefix echo.
  while text.startswith('{{') or text.startswith('{ {'):
    text = text.replace('{ {', '{', 1).replace('{{', '{', 1)
  while text.startswith('[[') or text.startswith('[ ['):
    text = text.replace('[ [', '[', 1).replace('[[', '[', 1)
  try:
    return json.loads(text)
  except json.JSONDecodeError as e:
    # Try regex extraction for first complete JSON object or array.
    obj_match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
    if obj_match:
      try:
        return json.loads(obj_match.group())
      except json.JSONDecodeError:
        pass
    # Try truncation repair.
    idx = max(text.find('{'), text.find('['))
    if idx >= 0:
      fragment = text[idx:]
      for suffix in ('"]}', '"}', '"]}}}', '}', ']}', ']'):
        try:
          result = json.loads(fragment + suffix)
          logging.info(
              "Repaired truncated JSON for %s (added '%s').",
              context,
              suffix,
          )
          return result
        except json.JSONDecodeError:
          pass
    logging.warning(
        "Failed to parse JSON for %s. Content: %.500s",
        context,
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


# Political spectrum for coalition alignment (left → right).
_POLITICAL_SPECTRUM = [
    "left",
    "green",
    "socialist",
    "liberal",
    "conservative",
    "populist",
]


def _spectrum_distance(party_a: str, party_b: str) -> int:
  """Return the distance between two parties on the political spectrum.

  Parties not in the spectrum are placed at the far end (distance 100)
  to discourage forming coalitions with unknown parties.
  """
  try:
    idx_a = _POLITICAL_SPECTRUM.index(party_a)
  except ValueError:
    idx_a = 100
  try:
    idx_b = _POLITICAL_SPECTRUM.index(party_b)
  except ValueError:
    idx_b = 100
  return abs(idx_a - idx_b)


def _form_coalition(
    seat_allocation: Dict[str, int],
) -> Tuple[Dict[str, int], Dict[str, int]]:
  """Form a governing coalition based on political alignment.

  The largest party seeks to join with the most ideologically aligned
  smaller partner (excluding the second-largest party, which almost
  always enters opposition).  Partners are selected by proximity on
  the political spectrum, breaking ties by seat count.

  The coalition accumulates partners until it holds a majority
  (>50%) of total seats.

  Returns:
    A tuple of ``(coalition, opposition)`` where each is
    ``{party: seats}``.
  """
  total_seats = sum(seat_allocation.values())
  majority_threshold = total_seats / 2

  # Sort parties by seats descending, alphabetical tie-break.
  ordered = sorted(
      seat_allocation.items(),
      key=lambda x: (-x[1], x[0]),
  )

  if not ordered:
    return {}, {}

  largest_party, largest_seats = ordered[0]
  coalition: Dict[str, int] = {largest_party: largest_seats}
  coalition_seats = largest_seats

  # The second-largest party goes to opposition.
  second_party = ordered[1][0] if len(ordered) > 1 else None

  # Candidate partners: everyone except the largest and second-largest.
  candidates = [
      (p, s) for p, s in ordered
      if p != largest_party and p != second_party
  ]

  # Sort candidates by political alignment to the largest party,
  # then by seats descending (prefer larger aligned partners).
  candidates.sort(
      key=lambda x: (_spectrum_distance(largest_party, x[0]), -x[1], x[0])
  )

  for party, seats in candidates:
    if coalition_seats > majority_threshold:
      break
    coalition[party] = seats
    coalition_seats += seats

  # If still no majority, reluctantly add the second-largest party.
  if coalition_seats <= majority_threshold and second_party:
    coalition[second_party] = seat_allocation[second_party]
    coalition_seats += seat_allocation[second_party]

  opposition = {
      p: s for p, s in seat_allocation.items() if p not in coalition
  }

  logging.info(
      "Coalition formed: %s (%d/%d seats). Opposition: %s",
      list(coalition.keys()),
      coalition_seats,
      total_seats,
      list(opposition.keys()),
  )
  return coalition, opposition


def _format_coalition_block(
    coalition: Dict[str, int],
    total_seats: int,
) -> str:
  """Format coalition composition for prompt injection."""
  lines = ["The governing coalition consists of:"]
  coalition_total = sum(coalition.values())
  for party, seats in sorted(coalition.items(), key=lambda x: -x[1]):
    pct = seats / coalition_total * 100 if coalition_total else 0
    lines.append(
        f"  {party}: {seats} seat{'s' if seats != 1 else ''}"
        f" ({pct:.0f}% of coalition)"
    )
  lines.append(
      f"Coalition total: {coalition_total}/{total_seats}"
      f" seats (majority)"
  )
  return "\n".join(lines)


def _summarize_constituency_sentiment(
    voter_responses: List,
    district_name: str,
) -> str:
  """Summarize voter sentiment in a district using simple counts."""
  district_responses = [
      vr for vr in voter_responses
      if getattr(vr, 'district', None)
      and (
          (isinstance(vr.district, dict)
           and vr.district.get('name') == district_name)
          or (isinstance(vr.district, str)
              and vr.district == district_name)
      )
  ]
  if not district_responses:
    return "(No voter data available for this district.)"

  n = len(district_responses)
  # Provide a brief summary with count and sample opinions.
  summary_lines = [
      f"{n} voter{'s' if n != 1 else ''} in your district"
      " shared their views on this issue:",
  ]
  # Show up to 5 representative responses (truncated).
  for vr in district_responses[:5]:
    summary_lines.append(f'  - "{vr.response[:120]}"')
  if n > 5:
    summary_lines.append(f"  ... and {n - 5} more.")
  return "\n".join(summary_lines)


def _expand_members(
    seat_allocation: Dict[str, int],
    districts: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, str]]:
  """Expand seat allocation into individual member records.

  When ``districts`` is provided, members are assigned to districts
  round-robin based on their party's seat order, giving each member
  a constituency.

  Returns a list of dicts with keys: ``name``, ``party``, ``seat_label``,
  ``seat_number``, ``district_name``.
  """
  members = []
  district_names = (
      [d.get('name', d) if isinstance(d, dict) else str(d)
       for d in districts]
      if districts else []
  )
  member_idx = 0
  for party in sorted(seat_allocation):
    num_seats = seat_allocation[party]
    for n in range(1, num_seats + 1):
      district_name = (
          district_names[member_idx % len(district_names)]
          if district_names else "at-large"
      )
      members.append({
          "name": f"{party}-member-{n}",
          "party": party,
          "seat_label": f"{party}-seat-{n}",
          "seat_number": str(n),
          "district_name": district_name,
      })
      member_idx += 1
  return members


# ---------------------------------------------------------------------------
# Phase 1: Draft Bill
# ---------------------------------------------------------------------------


def _draft_bill(
    model: Any,
    issue: str,
    seat_allocation: Dict[str, int],
    party_policies: Dict[str, PolicyResponse],
    coalition: Dict[str, int],
    prior_round: Optional[RoundRecord] = None,
    round_number: int = 1,
    temperature: float = 0.7,
) -> Bill:
  """Use the LLM to draft a bill on behalf of the governing coalition."""
  lead_party = _get_lead_party(coalition)
  seat_block = _format_seat_block(seat_allocation)
  total_seats = sum(seat_allocation.values())
  coalition_block = _format_coalition_block(coalition, total_seats)

  # Build coalition-specific policies block.
  coalition_policies = {
      p: party_policies[p]
      for p in coalition if p in party_policies
  }
  coalition_policies_block = _format_policies_block(coalition_policies)

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

  majority_votes = total_seats // 2 + 1
  prompt = COALITION_DRAFT_BILL_PROMPT.format(
      issue=issue,
      seat_block=seat_block,
      coalition_block=coalition_block,
      coalition_policies_block=coalition_policies_block,
      prior_round_block=prior_round_block,
      total_seats=total_seats,
      majority_votes=majority_votes,
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
      "Round %d: coalition (%s) drafted bill with %d points.",
      round_number,
      ", ".join(coalition.keys()),
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
    coalition: Dict[str, int],
    temperature: float,
) -> Amendment:
  """Query the LLM for a single opposition party's amendments."""
  seat_block = _format_seat_block(seat_allocation)
  total_seats = sum(seat_allocation.values())
  coalition_block = _format_coalition_block(coalition, total_seats)
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
      coalition_block=coalition_block,
      party_policy=party_policy,
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
    coalition: Dict[str, int],
    temperature: float = 0.7,
) -> List[Amendment]:
  """Gather amendments from opposition (non-coalition) parties."""
  # Descending seat order, only opposition parties.
  opposition = sorted(
      ((p, s) for p, s in seat_allocation.items() if p not in coalition),
      key=lambda x: -x[1],
  )

  if not opposition:
    logging.info("No opposition parties — skipping amendments.")
    return []

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
          coalition=coalition,
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


def _evaluate_single_amendment(
    model: Any,
    issue: str,
    coalition: Dict[str, int],
    party_policies: Dict[str, PolicyResponse],
    bill: Bill,
    amendment: Amendment,
    temperature: float,
) -> Tuple[Amendment, bool, str]:
  """Have the coalition evaluate a single opposition amendment."""
  coalition_policies = {
      p: party_policies[p] for p in coalition if p in party_policies
  }
  coalition_policies_block = _format_policies_block(coalition_policies)
  bill_block = _format_bill_block(bill)

  # Format the specific amendment.
  am_lines = []
  if amendment.removals:
    am_lines.append(f"Strike points: {amendment.removals}")
  if amendment.additions:
    for add in amendment.additions:
      am_lines.append(f"+ {add}")
  if not am_lines:
    am_lines.append("(No changes proposed.)")
  amendment_block = "\n".join(am_lines)

  prompt = COALITION_EVALUATE_AMENDMENT_PROMPT.format(
      coalition_parties_list=", ".join(coalition.keys()),
      issue=issue,
      coalition_policies_block=coalition_policies_block,
      bill_block=bill_block,
      amending_party=amendment.party,
      amendment_block=amendment_block,
  )

  lm = model.copy()
  with user():
    lm += prompt
  with assistant():
    lm += gen(
        max_tokens=256, temperature=temperature,
        name="eval_amendment_json",
    )

  data = _parse_json(
      lm["eval_amendment_json"],
      f"evaluate amendment from {amendment.party}",
  )
  adopt = bool(data.get("adopt", False))
  reason = str(data.get("reason", ""))

  logging.info(
      "Coalition %s amendment from %s: %s",
      "ADOPTED" if adopt else "REJECTED",
      amendment.party,
      reason[:100],
  )
  return amendment, adopt, reason


def _evaluate_amendments(
    model: Any,
    issue: str,
    coalition: Dict[str, int],
    party_policies: Dict[str, PolicyResponse],
    bill: Bill,
    amendments: List[Amendment],
    temperature: float = 0.7,
) -> List[Amendment]:
  """Have the coalition evaluate and selectively adopt amendments."""
  if not amendments:
    return []

  adopted: List[Amendment] = []
  for am in amendments:
    # Skip no-op amendments.
    if not am.removals and not am.additions:
      continue
    try:
      _, adopt, _ = _evaluate_single_amendment(
          model=model,
          issue=issue,
          coalition=coalition,
          party_policies=party_policies,
          bill=bill,
          amendment=am,
          temperature=temperature,
      )
      if adopt:
        adopted.append(am)
    except Exception:  # pylint: disable=broad-except
      logging.exception(
          "Failed to evaluate amendment from %s", am.party
      )

  logging.info(
      "Coalition adopted %d / %d opposition amendments.",
      len(adopted),
      len(amendments),
  )
  return adopted


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
    voter_responses: Optional[List] = None,
) -> MemberVote:
  """Query a single member for their vote with constituency context."""
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

  # Build constituency block.
  district_name = member.get("district_name", "at-large")
  if voter_responses and district_name != "at-large":
    sentiment = _summarize_constituency_sentiment(
        voter_responses, district_name,
    )
    constituency_block = (
        f"\n=== YOUR CONSTITUENCY ==="
        f"\nYou represent the district of '{district_name}'.\n"
        f"\n{sentiment}\n"
    )
  else:
    constituency_block = ""

  prompt = VOTE_BILL_PROMPT.format(
      member_name=member["name"],
      party=party,
      seat_number=member["seat_number"],
      issue=issue,
      seat_block=seat_block,
      constituency_block=constituency_block,
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
    voter_responses: Optional[List] = None,
    districts: Optional[List[Dict[str, Any]]] = None,
) -> VoteRecord:
  """Concurrently collect votes from all seated members."""
  members = _expand_members(seat_allocation, districts=districts)
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
            voter_responses,
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
    max_rounds: int = 5,
    max_workers: Optional[int] = None,
    voter_responses: Optional[List] = None,
    districts: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[DeliberationResult, Dict[str, Any]]:
  """Run the full coalition-based parliamentary deliberation process.

  Phases per round:
    1. **Coalition formation** — form a politically aligned coalition.
    2. **Draft** — The coalition co-drafts a bill proportionally.
    3. **Vote** — All seated members vote with constituency awareness.

  If a bill fails, the coalition re-drafts incorporating the prior
  round's voting record and re-submits to a vote.

  Args:
    model: A loaded PathFinder model instance.
    issue: The social issue text under debate.
    seat_allocation: ``{party_ideology: num_seats}`` from the election.
    party_policies: ``{party_ideology: PolicyResponse}`` from phase 4.
    temperature: LLM sampling temperature.
    max_rounds: Maximum number of bill consideration attempts.
    max_workers: Max concurrent LLM calls during the vote phase.
    voter_responses: Optional voter responses for constituency context.
    districts: Optional list of district dicts for member assignment.

  Returns:
    A tuple of ``(DeliberationResult, government_info)`` where
    ``government_info`` is a dict with coalition/seat metadata.
  """
  logging.info(
      "=== DELIBERATION START === Issue: %s | Parties: %s",
      issue[:80],
      list(seat_allocation.keys()),
  )

  # Phase 0: Form coalition.
  coalition, opposition = _form_coalition(seat_allocation)

  government_info = {
      "seat_allocation": dict(seat_allocation),
      "coalition": dict(coalition),
      "opposition": dict(opposition),
      "total_seats": sum(seat_allocation.values()),
  }

  result = DeliberationResult(issue=issue)
  prior_round: Optional[RoundRecord] = None

  for round_num in range(1, max_rounds + 1):
    logging.info("--- Deliberation round %d / %d ---", round_num, max_rounds)

    # Phase 1: Coalition drafts bill.
    bill = _draft_bill(
        model=model,
        issue=issue,
        seat_allocation=seat_allocation,
        party_policies=party_policies,
        coalition=coalition,
        prior_round=prior_round,
        round_number=round_num,
        temperature=temperature,
    )

    # Phase 2: All members vote (no amendment phase).
    vote_record = _vote_on_bill(
        model=model,
        issue=issue,
        seat_allocation=seat_allocation,
        party_policies=party_policies,
        bill=bill,
        amendments=[],
        temperature=temperature,
        max_workers=max_workers,
        voter_responses=voter_responses,
        districts=districts,
    )

    round_record = RoundRecord(
        bill=bill,
        amendments=[],
        vote_record=vote_record,
    )
    result.rounds.append(round_record)

    if vote_record.passed:
      result.adopted_bill = bill
      logging.info(
          "Round %d: Bill PASSED with %d yes / %d no.  "
          "Bill has %d points.",
          round_num,
          vote_record.yes_count,
          vote_record.no_count,
          len(bill.points),
      )
      break
    else:
      logging.info(
          "Round %d: Bill FAILED with %d yes / %d no.  "
          "Coalition will re-draft.",
          round_num,
          vote_record.yes_count,
          vote_record.no_count,
      )
      prior_round = round_record

  if result.adopted_bill is None and result.rounds:
    # Fall back to the bill with the most yes votes across all rounds.
    best_round = max(result.rounds, key=lambda r: r.vote_record.yes_count)
    result.adopted_bill = best_round.bill
    logging.info(
        "No majority reached after %d rounds. Adopting best bill "
        "(round %d, %d yes / %d no) for issue: %s",
        max_rounds,
        best_round.bill.round_number,
        best_round.vote_record.yes_count,
        best_round.vote_record.no_count,
        issue[:80],
    )

  logging.info("=== DELIBERATION END ===")
  return result, government_info
