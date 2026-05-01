"""Comparative policy ranking across multiple voting systems.

Runs all configured electoral systems on the same set of ranked voter
ballots, produces a parliamentary deliberation under each system's seat
allocation, and then asks voters to rank the *voting systems* based on
how well the resulting adopted policy aligns with their preferences.

Results are persisted as JSON with the schema::

    {
      "<social issue text>": {
        "policies": {
          "<system_name>": "<adopted policy text>" | null
        },
        "parties": {
          "<ideology>": {"position_statement": "...", "key_proposals": [...]}
        },
        "voters": {
          "<user_id>": "<response text>"
        },
        "rankings": {
          "<user_id>": ["system_a", "system_b", ...]
        }
      }
    }

One file is maintained per LLM model name.
"""

from concurrent import futures
import json
import logging
import os
import re
from typing import Any
from typing import Dict
from typing import List
from typing import Optional
from typing import Tuple

from pathfinder import assistant
from pathfinder import gen
from pathfinder import user
from simulation.deliberate import DeliberationResult
from simulation.deliberate import run_deliberation
from simulation.policy_generator import PolicyResponse
from simulation.survey import _format_demographics_block
from simulation.survey import _format_examples_block
from simulation.survey import VoterResponse
from simulation.voting import ElectionResult
from simulation.voting import run_election
from simulation.voting import VoterBallot

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

POLICY_RANKING_PROMPT = """\
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

Different voting systems were used to form governments and pass \
legislation on this issue.  Below are the policies that were adopted \
under each system.  Compare them against your own views.

{policies_block}

Based on your values, background, and your original response, rank the \
voting systems from MOST preferred (the system whose adopted policy best \
reflects your views) to LEAST preferred.  Rank all {num_systems} systems.

Return ONLY a JSON array of voting system name strings, best first.  \
Example: ["system_a", "system_b", ...]

Return ONLY the JSON array — no other text."""


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


def _format_region_block(
    voter_response: VoterResponse,
) -> str:
  """Build a region/district context block from a voter response."""
  district = voter_response.district
  if not district:
    return ""
  return (
      f"\nYou live in the district of '{district['name']}'.\n"
      "District Characteristics:\n"
      f"- Wealth: {district['wealth']} (0-1 scale)\n"
      f"- Urbanisation: {district['urbanisation']} (0-1 scale)\n"
      f"- Primary Industry: {district['industry']}\n"
      "- Political Leaning: "
      f"{district.get('political_leaning', 'unknown')}\n"
      f"- Description: {district['description']}\n"
  )


def _format_policies_block(
    system_policies: Dict[str, Optional[str]],
) -> str:
  """Format adopted policies for the ranking prompt.

  Args:
    system_policies: ``{system_name: adopted_policy_text | None}``

  Returns:
    Formatted block listing each system's policy.
  """
  lines = []
  for system_name in sorted(system_policies):
    policy = system_policies[system_name]
    lines.append(f"=== {system_name} ===")
    if policy:
      lines.append(policy)
    else:
      lines.append("  (No bill was adopted under this system.)")
    lines.append("")
  return "\n".join(lines)


def _sanitize_model_name(model_path: str) -> str:
  """Derive a filesystem-safe name from a model path.

  Takes the basename of the path and replaces non-alphanumeric chars
  (except hyphens, underscores, dots) with underscores.
  """
  base = os.path.basename(model_path)
  return re.sub(r"[^\w\-.]", "_", base)


def _bill_to_text(
    deliberation: DeliberationResult,
) -> Optional[str]:
  """Extract the adopted bill text from a deliberation result."""
  if deliberation.adopted_bill is None:
    return None
  points = deliberation.adopted_bill.points
  return "\n".join(f"  {i + 1}. {pt}" for i, pt in enumerate(points))


# ---------------------------------------------------------------------------
# Phase 1: Run comparative elections
# ---------------------------------------------------------------------------


def run_comparative_elections(
    ballots: List[VoterBallot],
    district_seats: Dict[str, int],
    party_policies: Dict[str, PolicyResponse],
    model: Any,
    voting_systems: List[str],
    issue: str,
    temperature: float = 0.7,
    deliberation_max_rounds: int = 3,
    max_workers: Optional[int] = None,
) -> Dict[str, Tuple[ElectionResult, DeliberationResult]]:
  """Run election + deliberation for each voting system.

  Args:
    ballots: Ranked voter ballots from the survey phase.
    district_seats: ``{district_name: num_seats}``.
    party_policies: ``{ideology: PolicyResponse}`` from the party response
      phase.
    model: Loaded PathFinder model instance.
    voting_systems: List of voting system names to compare.
    issue: The social issue text under debate.
    temperature: LLM sampling temperature.
    deliberation_max_rounds: Max bill consideration rounds per system.
    max_workers: Concurrency limit for deliberation votes.

  Returns:
    ``{system_name: (ElectionResult, DeliberationResult)}``
  """
  results: Dict[str, Tuple[ElectionResult, DeliberationResult]] = {}

  for system_name in voting_systems:
    logging.info(
        "=== Comparative: running system '%s' ===",
        system_name,
    )

    # Election.
    election = run_election(
        ballots=ballots,
        district_seats=district_seats,
        system=system_name,
    )

    if not election.total_seats:
      logging.warning(
          "System '%s' produced no seats — skipping deliberation.",
          system_name,
      )
      delib = DeliberationResult(issue=issue)
      results[system_name] = (election, delib)
      continue

    # Deliberation.
    delib = run_deliberation(
        model=model,
        issue=issue,
        seat_allocation=election.total_seats,
        party_policies=party_policies,
        temperature=temperature,
        max_rounds=deliberation_max_rounds,
        max_workers=max_workers,
    )

    results[system_name] = (election, delib)
    adopted = "yes" if delib.adopted_bill else "no"
    logging.info(
        "System '%s': governing=%s, bill_adopted=%s",
        system_name,
        election.governing_party,
        adopted,
    )

  return results


# ---------------------------------------------------------------------------
# Phase 2: Voter ranking of system policies
# ---------------------------------------------------------------------------


def _query_voter_system_ranking(
    voter_response: VoterResponse,
    system_policies: Dict[str, Optional[str]],
    model: Any,
    temperature: float,
) -> Tuple[str, List[str]]:
  """Ask a single voter to rank voting systems.

  Returns ``(user_id, [system ranking])``.
  """
  demo = voter_response.demographics
  demographics_block = _format_demographics_block(demo)
  examples_str = _format_examples_block(demo.get("examples", []))
  region_block = _format_region_block(voter_response)
  policies_block = _format_policies_block(system_policies)
  system_names = sorted(system_policies.keys())

  prompt = POLICY_RANKING_PROMPT.format(
      demographics_block=demographics_block,
      region_block=region_block,
      self_description=demo.get("self_description", ""),
      examples=examples_str,
      question=voter_response.question,
      voter_response=voter_response.response,
      policies_block=policies_block,
      num_systems=len(system_names),
  )

  lm = model.copy()
  with user():
    lm += prompt
  with assistant():
    lm += gen(
        max_tokens=512,
        temperature=temperature,
        name="system_ranking_json",
    )

  raw = lm["system_ranking_json"]
  text = _strip_llm_wrapping(raw)

  try:
    ranking = json.loads(text)
  except json.JSONDecodeError:
    logging.warning(
        "Failed to parse system ranking JSON for voter %s. Raw: %.200s",
        voter_response.user_id,
        text,
    )
    ranking = system_names  # Fallback: alphabetical.

  # Validate: keep only recognised system names.
  valid = [s for s in ranking if s in system_policies]
  # Fill missing systems.
  for s in system_names:
    if s not in valid:
      valid.append(s)
  ranking = valid[: len(system_names)]

  return (voter_response.user_id, ranking)


def rank_policies(
    voter_responses: List[VoterResponse],
    comparative_results: Dict[str, Tuple[ElectionResult, DeliberationResult]],
    model: Any,
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
) -> Dict[str, List[str]]:
  """Ask all voters to rank voting systems by policy preference.

  Each voter is shown the adopted policy from every system's
  deliberation and asked to rank the *systems* from most to least
  preferred.

  Args:
    voter_responses: Voter responses from the survey phase.
    comparative_results: Output of ``run_comparative_elections()``.
    model: Loaded PathFinder model instance.
    temperature: LLM sampling temperature.
    max_workers: Concurrency limit.

  Returns:
    ``{user_id: [system_name_best, ..., system_name_worst]}``
  """
  # Build the system → adopted policy text mapping.
  system_policies: Dict[str, Optional[str]] = {}
  for system_name, (_, delib) in comparative_results.items():
    system_policies[system_name] = _bill_to_text(delib)

  logging.info(
      "Ranking policies: %d voters, %d systems.",
      len(voter_responses),
      len(system_policies),
  )

  workers = max_workers or len(voter_responses)

  with futures.ThreadPoolExecutor(max_workers=workers) as pool:
    future_to_voter = {
        pool.submit(
            _query_voter_system_ranking,
            vr,
            system_policies,
            model,
            temperature,
        ): vr
        for vr in voter_responses
    }

    rankings: Dict[str, List[str]] = {}
    for future in futures.as_completed(future_to_voter):
      vr = future_to_voter[future]
      try:
        user_id, ranking = future.result()
        rankings[user_id] = ranking
        logging.info(
            "Voter %s system ranking: %s",
            user_id,
            ranking,
        )
      except Exception:  # pylint: disable=broad-except
        logging.exception(
            "Failed to get system ranking from voter %s",
            vr.user_id,
        )

  logging.info(
      "Collected %d / %d system rankings.",
      len(rankings),
      len(voter_responses),
  )
  return rankings


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_rankings(
    rankings: Dict[str, List[str]],
    issue: str,
    model_path: str,
    output_dir: str,
    system_policies: Optional[Dict[str, Optional[str]]] = None,
    voter_responses: Optional[List[VoterResponse]] = None,
    party_responses: Optional[Dict[str, PolicyResponse]] = None,
) -> str:
  """Persist voter system rankings to a per-model JSON file.

  The file is created or incrementally updated with the schema::

      {
        "<social issue>": {
          "policies": {
            "<system_name>": "<adopted policy text>" | null
          },
          "parties": {
            "<ideology>": {
              "position_statement": "...",
              "key_proposals": ["..."]
            }
          },
          "voters": {
            "<user_id>": "<response text>"
          },
          "rankings": {
            "<user_id>": ["system_a", "system_b", ...]
          }
        }
      }

  Args:
    rankings: ``{user_id: [system ranking]}``.
    issue: Social issue text (used as top-level key).
    model_path: Model path — basename used for the filename.
    output_dir: Directory to write the JSON file into.
    system_policies: Optional ``{system_name: policy_text | None}``. When
      provided, stored under ``<issue> → "policies"``.
    voter_responses: Optional list of VoterResponse objects. When provided,
      stored under ``<issue> → "voters"``.
    party_responses: Optional ``{ideology: PolicyResponse}``. When provided,
      stored under ``<issue> → "parties"``.

  Returns:
    The absolute path to the written JSON file.
  """
  os.makedirs(output_dir, exist_ok=True)

  filename = _sanitize_model_name(model_path) + ".json"
  filepath = os.path.join(output_dir, filename)

  # Load existing data if present.
  existing: Dict[str, Any] = {}
  if os.path.exists(filepath):
    try:
      with open(filepath, "r") as f:
        existing = json.load(f)
      logging.info(
          "Loaded existing results from %s (%d issues).",
          filepath,
          len(existing),
      )
    except (json.JSONDecodeError, OSError) as e:
      logging.warning(
          "Could not read existing results file %s: %s. Starting fresh.",
          filepath,
          e,
      )

  # Merge new data under the issue key.
  if issue not in existing:
    existing[issue] = {}

  # Store adopted policies per system.
  if system_policies is not None:
    existing[issue]["policies"] = {s: p for s, p in system_policies.items()}

  # Store party responses.
  if party_responses is not None:
    parties_dict: Dict[str, Any] = {}
    for ideology, pr in party_responses.items():
      parties_dict[ideology] = {
          "position_statement": pr.position_statement,
          "key_proposals": pr.key_proposals,
      }
    existing[issue]["parties"] = parties_dict

  # Store voter responses with personalization data.
  if voter_responses is not None:
    voters_dict: Dict[str, Any] = {}
    for vr in voter_responses:
      demo = vr.demographics or {}
      voters_dict[vr.user_id] = {
          "response": vr.response,
          "demographics": {
              k: v for k, v in demo.items()
              if k not in ("examples", "self_description")
          },
          "self_description": demo.get("self_description", ""),
          "examples": demo.get("examples", []),
      }
    existing[issue]["voters"] = voters_dict

  # Store voter system rankings.
  existing[issue]["rankings"] = rankings

  with open(filepath, "w") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

  logging.info(
      "Saved rankings to %s (%d voters for issue '%.60s').",
      filepath,
      len(rankings),
      issue,
  )
  return filepath
