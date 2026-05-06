"""Concurrent survey functions for voter and party response generation.

This module provides two pipeline stages:
1. `generate_voter_responses` — Sample k voters from PRISM and concurrently
   elicit each voter's opinion on a social issue via an LLM.  Each voter is
   optionally grounded in a district from a generated Region.
2. `generate_party_responses` — Given voter responses and a social issue,
   concurrently generate policy responses for every party platform found in
   the party dataset directory, conditioned on voter sentiment and regional
   context.
"""

import hashlib
import json
import logging
import os
import random
import re

from concurrent import futures
from dataclasses import asdict
from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pathfinder import assistant
from pathfinder import gen
from pathfinder import user

from simulation.district_generator import Region
from simulation.policy_generator import PartyPlatform
from simulation.policy_generator import PolicyResponse
from simulation.prism_sampler import PrismSampler
from simulation.voting import VoterBallot

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

VOTER_PROMPT = """\
You are a person with the following background:
{demographics_block}
{region_block}
Here is how you describe your own values and outlook:
  "{self_description}"

Here are some statements you have made in the past that reflect your views:
{examples}

Answer the following social-issue question as yourself. Be authentic and \
draw on your background and the community you live in.  Provide a thoughtful \
response in 2-4 sentences.  Do not mention that you are an AI or that you \
are being personalized.

Question: {question}"""


PARTY_RESPONSE_PROMPT = """\
You are a senior political policy advisor for {party_name}, \
a {ideology} party.

=== PARTY PLATFORM ===
{platform_summary}

=== SOCIAL ISSUE ===
{issue}

=== CONSTITUENCY CONTEXT ===
{constituency_block}

=== VOTER RESPONSES FROM YOUR CONSTITUENCY ===
The following are responses from potential voters in your constituency. \
Use them to gauge constituent sentiment and tailor your proposal accordingly.

{voter_block}

=== INSTRUCTIONS ===
Draft a policy response as a JSON object with these fields:
- "issue": string — the issue text exactly as given above
- "position_statement": string — 1-2 sentence official party position
- "key_proposals": array of 3-5 strings — concrete, actionable policy proposals
- "voter_alignment_score": float 0-1 — estimated alignment between this \
response and the voter sentiments provided (1 = perfect alignment)
- "reasoning": string — brief explanation of how the party platform and voter \
feedback shaped this response

Return ONLY valid JSON — no markdown fences, no commentary."""


VOTER_RANKING_PROMPT = """\
You are a person with the following background:
{demographics_block}
{region_block}
Here is how you describe your own values and outlook:
  "{self_description}"

Here are some statements you have made in the past that reflect your views:
{examples}

You previously shared your opinion on the following social issue:
  "{question}"

Your response was:
  "{voter_response}"

The following political parties have published policy proposals on this issue:

{party_policies_block}

Based on your values, background, and your original response, rank the \
parties from MOST aligned with your views to LEAST aligned.  Return your \
ranking as a JSON array of party ideology strings, best first.  Rank \
exactly {max_rank} parties.

Return ONLY the JSON array — no other text."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class VoterResponse:
  """A single voter's response to a social issue."""

  user_id: str
  demographics: Dict[str, Any]
  district: Optional[Dict[str, Any]]
  question: str
  response: str

  def to_dict(self) -> Dict[str, Any]:
    return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _assign_district(
    user_id: str,
    region: Region,
) -> Dict[str, Any]:
  """Deterministically assign a district to a user based on their ID."""
  user_hash = hashlib.md5(user_id.encode()).digest()
  idx = int.from_bytes(user_hash, "big") % len(region.districts)
  return region.districts[idx]


def _format_region_block(
    region: Optional[Region],
    district: Optional[Dict[str, Any]],
) -> str:
  """Build the region/district section for a voter prompt."""
  if not region or not district:
    return ""
  return (
      f"\nYou live in the region of '{region.region_name}'"
      f" (inspired by {region.description}).\n"
      f"Specifically, you live in the district of '{district['name']}'.\n"
      f"District Characteristics:\n"
      f"- Wealth: {district['wealth']} (0-1 scale)\n"
      f"- Urbanisation: {district['urbanisation']} (0-1 scale)\n"
      f"- Primary Industry: {district['industry']}\n"
      f"- Political Leaning: {district.get('political_leaning', 'unknown')}\n"
      f"- Description: {district['description']}\n"
  )


def _format_demographics_block(demo: Dict[str, Any]) -> str:
  """Build a demographics bullet list from the expanded PRISM data."""
  lines = []
  _add = lambda label, key: (  # pylint: disable=unnecessary-lambda-assignment
      lines.append(f"- {label}: {demo[key]}")
      if demo.get(key) else None
  )
  _add("Age", "age")
  _add("Gender", "gender")
  _add("Employment", "employment")
  _add("Education", "education")
  _add("Ethnicity", "ethnicity")
  _add("Religion", "religion")
  _add("Location", "location")
  _add("Marital status", "marital_status")
  return "\n".join(lines) if lines else "(demographics unknown)"


def _format_examples_block(examples: List[str]) -> str:
  """Format example statements as a bulleted list."""
  return "\n".join(f"- {ex}" for ex in examples) if examples else ""


def _format_constituency_block(
    voter_responses: List["VoterResponse"],
    region: Optional[Region],
) -> str:
  """Build a constituency demographics summary for the party prompt."""
  if not region:
    return "No regional data available."

  lines = [
      f"Region: {region.region_name} ({region.description})",
      f"Total districts: {len(region.districts)}",
  ]

  # Summarise the districts represented by the voters.
  district_names = set()
  for vr in voter_responses:
    if vr.district:
      district_names.add(vr.district["name"])

  if district_names:
    lines.append(
        f"Districts represented by respondents: {', '.join(sorted(district_names))}"
    )

  return "\n".join(lines)


# ---------------------------------------------------------------------------
# Voter response generation
# ---------------------------------------------------------------------------


def _build_voter_prompt(
    voter: Dict[str, Any],
    question: str,
    region: Optional[Region] = None,
    district: Optional[Dict[str, Any]] = None,
) -> str:
  """Build a personalized prompt for a single voter."""
  demo = voter["demographics"]
  demographics_block = _format_demographics_block(demo)
  examples_str = _format_examples_block(voter["examples"])
  region_block = _format_region_block(region, district)
  return VOTER_PROMPT.format(
      demographics_block=demographics_block,
      region_block=region_block,
      self_description=demo.get("self_description", ""),
      examples=examples_str,
      question=question,
  )


def _query_voter(
    voter: Dict[str, Any],
    question: str,
    model: Any,
    temperature: float,
    region: Optional[Region] = None,
    district: Optional[Dict[str, Any]] = None,
) -> VoterResponse:
  """Query the LLM as a single voter.  Intended to run in a thread.

  ``model`` should be a shared PathFinder model instance — this function
  calls ``model.copy()`` to obtain a thread-local conversation state while
  sharing the underlying GPU model.
  """
  prompt = _build_voter_prompt(voter, question, region, district)

  lm = model.copy()

  with user():
    lm += prompt
  with assistant():
    lm += gen(max_tokens=1024, temperature=temperature, name="voter_response")

  raw = lm["voter_response"]
  # Strip <think> blocks if present.
  text = re.sub(r"<think>.*?(</think>|$)", "", raw, flags=re.DOTALL).strip()

  # Include examples in demographics so downstream ranking phases can
  # re-use them for prompt personalization.
  demo_with_examples = {**voter["demographics"], "examples": voter["examples"]}

  return VoterResponse(
      user_id=voter["user_id"],
      demographics=demo_with_examples,
      district=district,
      question=question,
      response=text,
  )


def generate_voter_responses(
    k: int,
    question: str,
    prism_sampler: PrismSampler,
    model: Any,
    seed: int = 42,
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
    region: Optional[Region] = None,
    voters_override: Optional[List[Dict[str, Any]]] = None,
) -> List[VoterResponse]:
  """Sample *k* voters from PRISM and concurrently generate their responses.

  Args:
    k: Number of voters to sample.
    question: The social-issue question text (e.g. from
      ``PoliticalQuestionSampler.sample_question_text``).
    prism_sampler: An initialised ``PrismSampler`` instance.
    model: A loaded PathFinder model instance (shared across threads).
    seed: Random seed — fixes both voter sampling and LLM generation.
    temperature: Sampling temperature.
    max_workers: Maximum concurrent LLM calls.  Defaults to *k*.
    region: Optional ``Region`` — if provided, each voter is
      deterministically assigned a district based on their user_id.
    voters_override: If provided, use these pre-loaded personas
      instead of sampling from ``prism_sampler``.

  Returns:
    A list of ``VoterResponse`` objects, one per sampled voter.
  """
  if voters_override is not None:
    voters = voters_override[:k]
  else:
    voters = prism_sampler.sample(num_samples=k, seed=seed)
  logging.info(
      "Using %d voters (seed=%d).  Generating responses concurrently...",
      len(voters),
      seed,
  )

  # Pre-assign districts so they are available for prompt building.
  voter_districts: List[Optional[Dict[str, Any]]] = []
  for v in voters:
    if region:
      district = _assign_district(v["user_id"], region)
      voter_districts.append(district)
      logging.info(
          "Voter %s assigned to district '%s'.",
          v["user_id"],
          district["name"],
      )
    else:
      voter_districts.append(None)

  workers = max_workers or len(voters)

  with futures.ThreadPoolExecutor(max_workers=workers) as pool:
    future_to_voter = {
        pool.submit(
            _query_voter,
            voter,
            question,
            model,
            temperature,
            region,
            district,
        ): voter
        for voter, district in zip(voters, voter_districts)
    }

    results: List[VoterResponse] = []
    for future in futures.as_completed(future_to_voter):
      voter = future_to_voter[future]
      try:
        result = future.result()
        results.append(result)
        logging.info(
            "Voter %s responded (%d chars).",
            result.user_id,
            len(result.response),
        )
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "Failed to generate response for voter %s", voter["user_id"]
        )

  logging.info("Collected %d / %d voter responses.", len(results), len(voters))
  return results


# ---------------------------------------------------------------------------
# Party response generation
# ---------------------------------------------------------------------------

def _parse_party_response(
    raw: str,
    platform: PartyPlatform,
) -> PolicyResponse:
  """Parse a single-issue policy JSON from the LLM into a PolicyResponse."""
  text = re.sub(r"<think>.*?(</think>|$)", "", raw, flags=re.DOTALL).strip()

  if text.startswith("```"):
    text = text.split("\n", 1)[1]
    text = text.rsplit("```", 1)[0]
    text = text.strip()

  if not text:
    raise ValueError(
        f"LLM returned empty response for {platform.party_name}. "
        f"Raw: {raw!r:.200}"
    )

  try:
    data = json.loads(text)
  except json.JSONDecodeError as e:
    logging.error("Failed to parse party JSON.  Content: %.500s", text)
    raise ValueError(
        f"Invalid JSON from LLM for {platform.party_name}: {e}. "
        f"First 200 chars: {text[:200]!r}"
    ) from e

  # Handle both bare object and wrapped list.
  if isinstance(data, list):
    data = data[0] if data else {}

  return PolicyResponse(
      party_name=platform.party_name,
      ideology=platform.ideology,
      issue=data.get("issue", "unknown"),
      position_statement=data.get("position_statement", ""),
      key_proposals=data.get("key_proposals", []),
      voter_alignment_score=float(data.get("voter_alignment_score", 0.0)),
      reasoning=data.get("reasoning", ""),
  )


def _query_party(
    platform: PartyPlatform,
    issue: str,
    voter_responses: List[VoterResponse],
    model: Any,
    temperature: float,
    region: Optional[Region] = None,
) -> PolicyResponse:
  """Generate a policy response for a single party.  Runs in a thread.

  ``model`` should be a shared PathFinder model instance — this function
  calls ``model.copy()`` to obtain a thread-local conversation state.
  """
  voter_block = "\n".join(
      f'  Voter {i + 1}: "{vr.response}"'
      for i, vr in enumerate(voter_responses)
  )
  if not voter_responses:
    voter_block = "  (No voter responses provided.)"

  constituency_block = _format_constituency_block(voter_responses, region)

  prompt = PARTY_RESPONSE_PROMPT.format(
      party_name=platform.party_name,
      ideology=platform.ideology,
      platform_summary=platform.summary(),
      issue=issue,
      constituency_block=constituency_block,
      voter_block=voter_block,
  )

  lm = model.copy()

  with user():
    lm += prompt
  with assistant():
    lm += gen(max_tokens=4096, temperature=temperature, name="party_json")

  raw = lm["party_json"]
  return _parse_party_response(raw, platform)


def generate_party_responses(
    issue: str,
    voter_responses: List[VoterResponse],
    party_dir: str,
    model: Any,
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
    region: Optional[Region] = None,
    parties: Optional[List[str]] = None,
) -> Dict[str, PolicyResponse]:
  """Generate policy responses for all party platforms concurrently.

  Args:
    issue: The social-issue question text.
    voter_responses: Voter responses from ``generate_voter_responses``.
    party_dir: Path to directory containing party platform JSON files
      (e.g. ``dataset/party/``).
    model: A loaded PathFinder model instance (shared across threads).
    temperature: Sampling temperature.
    max_workers: Maximum concurrent LLM calls.  Defaults to the number
      of party platforms.
    region: Optional ``Region`` — passed to party prompts to provide
      constituency context.
    parties: Optional list of ideology names to include (e.g.
      ``["liberal", "conservative"]``).  If ``None``, all platforms in
      ``party_dir`` are used.

  Returns:
    A dict mapping ideology string to its ``PolicyResponse``.

  Raises:
    FileNotFoundError: If no valid party platform JSON files are found
      in ``party_dir``.
  """
  # Normalise the filter set for case-insensitive matching.
  party_filter = {p.lower() for p in parties} if parties else None

  # Load party platforms, optionally filtered.
  platforms: List[PartyPlatform] = []
  for filename in sorted(os.listdir(party_dir)):
    if not filename.endswith(".json"):
      continue
    ideology_key = filename.removesuffix(".json").lower()
    if party_filter and ideology_key not in party_filter:
      continue
    path = os.path.join(party_dir, filename)
    try:
      platforms.append(PartyPlatform.from_json(path))
    except (json.JSONDecodeError, KeyError) as e:
      logging.warning("Skipping %s: %s", path, e)

  if not platforms:
    raise FileNotFoundError(
        f"No valid party platform JSON files found in {party_dir}"
    )

  logging.info(
      "Generating party responses for %d platforms concurrently...",
      len(platforms),
  )

  workers = max_workers or len(platforms)

  with futures.ThreadPoolExecutor(max_workers=workers) as pool:
    future_to_platform = {
        pool.submit(
            _query_party,
            platform,
            issue,
            voter_responses,
            model,
            temperature,
            region,
        ): platform
        for platform in platforms
    }

    results: Dict[str, PolicyResponse] = {}
    for future in futures.as_completed(future_to_platform):
      platform = future_to_platform[future]
      try:
        result = future.result()
        results[platform.ideology] = result
        logging.info(
            "Party %s (%s) responded.",
            platform.party_name,
            platform.ideology,
        )
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "Failed to generate response for %s (%s)",
            platform.party_name,
            platform.ideology,
        )

  logging.info(
      "Collected %d / %d party responses.", len(results), len(platforms)
  )
  return results


# ---------------------------------------------------------------------------
# Voter ranking generation
# ---------------------------------------------------------------------------


def _format_party_policies_block(
    party_responses: Dict[str, PolicyResponse],
    voter_seed: Optional[int] = None,
) -> str:
  """Format party policy responses for the voter ranking prompt.

  When ``voter_seed`` is provided, the parties are presented in
  a randomised order (seeded per-voter) to avoid ordering bias.
  """
  ideologies = list(party_responses.keys())
  if voter_seed is not None:
    random.Random(voter_seed).shuffle(ideologies)
  else:
    ideologies.sort()
  lines = []
  for ideology in ideologies:
    pr = party_responses[ideology]
    proposals = (
        "; ".join(pr.key_proposals)
        if pr.key_proposals
        else "N/A"
    )
    lines.append(
        f"  {pr.party_name} ({pr.ideology}):\n"
        f"    Position: {pr.position_statement}\n"
        f"    Key Proposals: {proposals}"
    )
  return "\n\n".join(lines)


def _query_voter_ranking(
    voter_response: VoterResponse,
    party_responses: Dict[str, PolicyResponse],
    model: Any,
    temperature: float,
    max_rank: int = 3,
    region: Optional[Region] = None,
) -> VoterBallot:
  """Ask one voter to rank parties based on policy responses.

  ``model`` should be a shared PathFinder model instance — this function
  calls ``model.copy()`` to obtain a thread-local conversation state.
  """
  demo = voter_response.demographics
  demographics_block = _format_demographics_block(demo)
  examples_str = _format_examples_block(demo.get("examples", []))
  region_block = _format_region_block(region, voter_response.district)
  voter_seed = int.from_bytes(
      hashlib.md5(
          f"{voter_response.user_id}:party_rank:"
          f"{voter_response.question}".encode()
      ).digest(),
      "big",
  )
  party_policies_block = _format_party_policies_block(
      party_responses, voter_seed=voter_seed
  )

  # Clamp max_rank to the number of available parties.
  available_parties = list(sorted(party_responses.keys()))
  effective_rank = min(max_rank, len(available_parties))

  prompt = VOTER_RANKING_PROMPT.format(
      demographics_block=demographics_block,
      region_block=region_block,
      self_description=demo.get("self_description", ""),
      examples=examples_str,
      question=voter_response.question,
      voter_response=voter_response.response,
      party_policies_block=party_policies_block,
      max_rank=effective_rank,
  )

  lm = model.copy()

  with user():
    lm += prompt
  with assistant():
    lm += gen(max_tokens=512, temperature=temperature, name="ranking_json")

  raw = lm["ranking_json"]
  # Strip <think> blocks if present.
  text = re.sub(r"<think>.*?(</think>|$)", "", raw, flags=re.DOTALL).strip()

  # Strip markdown fences if present.
  if text.startswith("```"):
    text = text.split("\n", 1)[1]
    text = text.rsplit("```", 1)[0]
    text = text.strip()

  try:
    ranking = json.loads(text)
  except json.JSONDecodeError:
    logging.warning(
        "Failed to parse ranking JSON for voter %s. Raw: %.200s",
        voter_response.user_id,
        text,
    )
    # Fallback: use available parties in alphabetical order.
    ranking = available_parties[:effective_rank]

  # Validate and sanitise: keep only recognised ideologies.
  valid = [r for r in ranking if r in party_responses]
  if len(valid) < effective_rank:
    # Fill missing slots with any un-ranked parties.
    for p in available_parties:
      if p not in valid:
        valid.append(p)
      if len(valid) >= effective_rank:
        break
  ranking = valid[:effective_rank]

  district_name = (
      voter_response.district["name"]
      if voter_response.district
      else "unassigned"
  )

  return VoterBallot(
      user_id=voter_response.user_id,
      district_name=district_name,
      ranking=ranking,
  )


def generate_voter_rankings(
    voter_responses: List[VoterResponse],
    party_responses: Dict[str, PolicyResponse],
    model: Any,
    temperature: float = 0.7,
    max_rank: int = 3,
    max_workers: Optional[int] = None,
    region: Optional[Region] = None,
) -> List[VoterBallot]:
  """Concurrently generate voter rankings of party policies.

  Args:
    voter_responses: Voter responses from phase 3.
    party_responses: Party policy responses from phase 4.
    model: A loaded PathFinder model instance (shared across threads).
    temperature: Sampling temperature.
    max_rank: Maximum number of parties each voter ranks.
    max_workers: Concurrency limit.
    region: Optional region for prompt context.

  Returns:
    A list of ``VoterBallot`` objects, one per voter.
  """
  logging.info(
      "Generating voter rankings for %d voters (max_rank=%d)...",
      len(voter_responses),
      max_rank,
  )

  workers = max_workers or len(voter_responses)

  with futures.ThreadPoolExecutor(max_workers=workers) as pool:
    future_to_voter = {
        pool.submit(
            _query_voter_ranking,
            vr,
            party_responses,
            model,
            temperature,
            max_rank,
            region,
        ): vr
        for vr in voter_responses
    }

    ballots: List[VoterBallot] = []
    for future in futures.as_completed(future_to_voter):
      vr = future_to_voter[future]
      try:
        ballot = future.result()
        ballots.append(ballot)
        logging.info(
            "Voter %s ranked: %s",
            ballot.user_id,
            ballot.ranking,
        )
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "Failed to generate ranking for voter %s",
            vr.user_id,
        )

  logging.info(
      "Collected %d / %d voter rankings.",
      len(ballots),
      len(voter_responses),
  )
  return ballots


# ---------------------------------------------------------------------------
# Platform-only voting mode
# ---------------------------------------------------------------------------

PLATFORM_RANKING_PROMPT = """\
You are a person with the following background:
{demographics_block}
{region_block}
Here is how you describe your own values and outlook:
  "{self_description}"

Here are some statements you have made in the past that reflect \
your views:
{examples}

The following political parties are running for election.  Review \
their platforms and rank them based on how well each party's values \
and policy positions align with your own.

{platforms_block}

Rank the parties from MOST aligned with your views to LEAST \
aligned.  Return your ranking as a JSON array of party ideology \
strings, best first.  Rank exactly {max_rank} parties.

Return ONLY the JSON array — no other text."""


def _format_platforms_block(
    platforms: List[PartyPlatform],
    voter_seed: Optional[int] = None,
) -> str:
  """Format party platform summaries for the ranking prompt.

  When ``voter_seed`` is provided, the platforms are presented in
  a randomised order (seeded per-voter) to avoid ordering bias.
  """
  ordered = list(platforms)
  if voter_seed is not None:
    random.Random(voter_seed).shuffle(ordered)
  else:
    ordered.sort(key=lambda x: x.ideology)
  lines = []
  for p in ordered:
    lines.append(
        f"=== {p.party_name} ({p.ideology}) ===\n"
        f"{p.summary()}"
    )
  return "\n\n".join(lines)


def _query_platform_ranking(
    voter_data: Dict[str, Any],
    district: Optional[Dict[str, Any]],
    platforms: List[PartyPlatform],
    model: Any,
    temperature: float,
    max_rank: int,
    region: Optional[Region],
) -> VoterBallot:
  """Ask one voter to rank parties based on platforms only.

  Args:
    voter_data: Raw PRISM voter dict with ``demographics``,
      ``examples``, and ``user_id`` keys.
    district: Optional district dict for this voter.
    platforms: List of ``PartyPlatform`` objects.
    model: A loaded PathFinder model instance.
    temperature: LLM sampling temperature.
    max_rank: Number of parties to rank.
    region: Optional ``Region`` for prompt context.

  Returns:
    A ``VoterBallot`` with the voter's platform-based ranking.
  """
  demo = voter_data["demographics"]
  demographics_block = _format_demographics_block(demo)
  examples_str = _format_examples_block(voter_data["examples"])
  region_block = _format_region_block(region, district)
  voter_seed = int.from_bytes(
      hashlib.md5(
          f"{voter_data['user_id']}:platform_rank".encode()
      ).digest(),
      "big",
  )
  platforms_block = _format_platforms_block(
      platforms, voter_seed=voter_seed
  )

  available = sorted(p.ideology for p in platforms)
  effective_rank = min(max_rank, len(available))

  prompt = PLATFORM_RANKING_PROMPT.format(
      demographics_block=demographics_block,
      region_block=region_block,
      self_description=demo.get("self_description", ""),
      examples=examples_str,
      platforms_block=platforms_block,
      max_rank=effective_rank,
  )

  lm = model.copy()
  with user():
    lm += prompt
  with assistant():
    lm += gen(
        max_tokens=512,
        temperature=temperature,
        name="ranking_json",
    )

  raw = lm["ranking_json"]
  text = re.sub(
      r"<think>.*?(</think>|$)", "", raw, flags=re.DOTALL
  ).strip()
  if text.startswith("```"):
    text = text.split("\n", 1)[1]
    text = text.rsplit("```", 1)[0]
    text = text.strip()

  try:
    ranking = json.loads(text)
  except json.JSONDecodeError:
    logging.warning(
        "Failed to parse platform ranking JSON for voter %s."
        " Raw: %.200s",
        voter_data["user_id"],
        text,
    )
    ranking = available[:effective_rank]

  valid = [r for r in ranking if r in set(available)]
  if len(valid) < effective_rank:
    for p in available:
      if p not in valid:
        valid.append(p)
      if len(valid) >= effective_rank:
        break
  ranking = valid[:effective_rank]

  district_name = district["name"] if district else "unassigned"

  return VoterBallot(
      user_id=voter_data["user_id"],
      district_name=district_name,
      ranking=ranking,
  )


def prepare_voters(
    num_voters: int,
    prism_sampler: PrismSampler,
    seed: int,
    region: Optional[Region] = None,
    voters_override: Optional[List[Dict[str, Any]]] = None,
) -> Tuple[
    List[Dict[str, Any]], List[Optional[Dict[str, Any]]]
]:
  """Sample voters from PRISM and assign districts.

  This is a reusable helper that separates voter sampling from
  querying, so the same voters can be used across multiple
  pipeline phases.

  Args:
    num_voters: Number of voters to sample.
    prism_sampler: An initialised ``PrismSampler`` instance.
    seed: Random seed for voter sampling.
    region: Optional ``Region`` — if provided, each voter is
      deterministically assigned a district.
    voters_override: If provided, use these pre-loaded personas
      instead of sampling from ``prism_sampler``.

  Returns:
    A tuple of ``(voters, voter_districts)`` where ``voters``
    is the list of raw PRISM voter dicts and
    ``voter_districts`` is the parallel list of district dicts
    (or ``None`` per voter if no region).
  """
  if voters_override is not None:
    voters = voters_override[:num_voters]
  else:
    voters = prism_sampler.sample(num_samples=num_voters, seed=seed)
  logging.info("Using %d voters (seed=%d).", len(voters), seed)

  voter_districts: List[Optional[Dict[str, Any]]] = []
  # Build a lookup by district name for O(1) access.
  district_by_name: Dict[str, Dict[str, Any]] = {}
  if region:
    district_by_name = {d["name"]: d for d in region.districts}

  for v in voters:
    if region:
      # Prefer the pre-assigned district from personas.json.
      assigned_name = v.get("district")
      if assigned_name and assigned_name in district_by_name:
        voter_districts.append(district_by_name[assigned_name])
      else:
        # Fallback: hash-based assignment.
        district = _assign_district(v["user_id"], region)
        voter_districts.append(district)
    else:
      voter_districts.append(None)

  return voters, voter_districts


def survey_voters(
    voters: List[Dict[str, Any]],
    voter_districts: List[Optional[Dict[str, Any]]],
    question: str,
    model: Any,
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
    region: Optional[Region] = None,
) -> List[VoterResponse]:
  """Generate voter opinions for a question using pre-sampled voters.

  Like ``generate_voter_responses`` but accepts pre-sampled voter
  data instead of sampling internally.  This allows the same set
  of voters to be reused across multiple questions.

  Args:
    voters: List of raw PRISM voter dicts (from
      ``prepare_voters``).
    voter_districts: Parallel list of district dicts (from
      ``prepare_voters``).
    question: The social-issue question text.
    model: A loaded PathFinder model instance.
    temperature: Sampling temperature.
    max_workers: Maximum concurrent LLM calls.
    region: Optional ``Region`` for prompt context.

  Returns:
    A list of ``VoterResponse`` objects, one per voter.
  """
  logging.info(
      "Surveying %d pre-sampled voters on: %.60s ...",
      len(voters),
      question,
  )

  workers = max_workers or len(voters)

  with futures.ThreadPoolExecutor(max_workers=workers) as pool:
    future_to_voter = {
        pool.submit(
            _query_voter,
            voter,
            question,
            model,
            temperature,
            region,
            district,
        ): voter
        for voter, district in zip(voters, voter_districts)
    }

    results: List[VoterResponse] = []
    for future in futures.as_completed(future_to_voter):
      voter = future_to_voter[future]
      try:
        result = future.result()
        results.append(result)
        logging.info(
            "Voter %s responded (%d chars).",
            result.user_id,
            len(result.response),
        )
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "Failed to generate response for voter %s",
            voter["user_id"],
        )

  logging.info(
      "Collected %d / %d voter responses.",
      len(results),
      len(voters),
  )
  return results


def generate_platform_rankings(
    voters: List[Dict[str, Any]],
    voter_districts: List[Optional[Dict[str, Any]]],
    platforms: List[PartyPlatform],
    model: Any,
    temperature: float = 0.7,
    max_rank: int = 3,
    max_workers: Optional[int] = None,
    region: Optional[Region] = None,
) -> List[VoterBallot]:
  """Voters rank parties based on platforms only (no issue).

  Each voter is shown the full platform summary for every party
  and asked to rank them by alignment with their values.  This
  is issue-independent — the same ballots are reused for seat
  allocation across all issues.

  Args:
    voters: List of raw PRISM voter dicts (from
      ``prepare_voters``).
    voter_districts: Parallel list of district dicts.
    platforms: List of ``PartyPlatform`` objects.
    model: A loaded PathFinder model instance.
    temperature: Sampling temperature.
    max_rank: Number of parties each voter ranks.
    max_workers: Concurrency limit.
    region: Optional ``Region`` for prompt context.

  Returns:
    A list of ``VoterBallot`` objects, one per voter.
  """
  logging.info(
      "Generating platform rankings for %d voters "
      "(max_rank=%d, %d parties)...",
      len(voters),
      max_rank,
      len(platforms),
  )

  workers = max_workers or len(voters)

  with futures.ThreadPoolExecutor(max_workers=workers) as pool:
    future_to_voter = {
        pool.submit(
            _query_platform_ranking,
            voter,
            district,
            platforms,
            model,
            temperature,
            max_rank,
            region,
        ): voter
        for voter, district in zip(voters, voter_districts)
    }

    ballots: List[VoterBallot] = []
    for future in futures.as_completed(future_to_voter):
      voter = future_to_voter[future]
      try:
        ballot = future.result()
        ballots.append(ballot)
        logging.info(
            "Voter %s platform ranking: %s",
            ballot.user_id,
            ballot.ranking,
        )
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "Failed to get platform ranking from %s",
            voter["user_id"],
        )

  logging.info(
      "Collected %d / %d platform rankings.",
      len(ballots),
      len(voters),
  )
  return ballots


def platform_to_policy_response(
    platform: PartyPlatform,
) -> PolicyResponse:
  """Convert a ``PartyPlatform`` to a ``PolicyResponse``.

  Used in platform-only mode where party policy generation is
  skipped and the static platform summary serves as the policy
  for deliberation.

  Args:
    platform: A ``PartyPlatform`` loaded from JSON.

  Returns:
    A ``PolicyResponse`` with the platform summary as the
    position statement and core values as key proposals.
  """
  return PolicyResponse(
      party_name=platform.party_name,
      ideology=platform.ideology,
      issue="general platform",
      position_statement=platform.summary(),
      key_proposals=list(platform.core_values),
      voter_alignment_score=0.0,
      reasoning="Static platform — no issue-specific generation.",
  )
