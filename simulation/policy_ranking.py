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
        "ballots": {
          "<user_id>": {"district": "...", "ranking": [...]}
        },
        "rankings": {
          "<user_id>": ["system_a", "system_b", ...]
        },
        "scores": {
          "<user_id>": {"system_a": 4.2, "system_b": 2.0}
        }
      }
    }

One file is maintained per LLM model name.
"""

from concurrent import futures
import hashlib
import json
import logging
import os
import random
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
from simulation.deliberate import draft_baseline_bill
from simulation.deliberate import draft_baseline_informed_bill
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

You must do TWO things:

1. **Rank** the voting systems from MOST preferred (the system whose \
adopted policy best reflects your views) to LEAST preferred.  Rank \
all {num_systems} systems.

2. **Score** each system on a Likert scale from 1.0 to 5.0 indicating \
how well its policy matches your preferences:
   1.0 = does not match at all
   3.0 = neutral / partially matches
   5.0 = perfect match
Scores must be in increments of 0.1 (e.g. 1.0, 2.3, 4.5).  \
Values like 2.25 or 4.333 are NOT allowed.

Return ONLY a JSON object with two keys:
- "ranking": an array of voting system name strings, best first
- "scores": an object mapping each system name to its score

Example:
{{"ranking": ["system_a", "system_b"], \
"scores": {{"system_a": 4.2, "system_b": 2.0}}}}

Return ONLY the JSON object — no other text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _strip_llm_wrapping(raw: str) -> str:
  """Remove <think> blocks and markdown fences from LLM output."""
  text = re.sub(r"<think>.*?(</think>|$)", "", raw, flags=re.DOTALL).strip()
  if text.startswith("```"):
    parts = text.split("\n", 1)
    text = parts[1] if len(parts) > 1 else parts[0][3:]
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
    voter_seed: Optional[int] = None,
) -> str:
  """Format adopted policies for the ranking prompt.

  When ``voter_seed`` is provided, the systems are presented in
  a randomised order (seeded per-voter) to avoid ordering bias.

  Args:
    system_policies: ``{system_name: adopted_policy_text | None}``
    voter_seed: Optional per-voter seed for shuffling.

  Returns:
    Formatted block listing each system's policy.
  """
  system_names = list(system_policies.keys())
  if voter_seed is not None:
    random.Random(voter_seed).shuffle(system_names)
  else:
    system_names.sort()
  lines = []
  for system_name in system_names:
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
    voter_responses: Optional[List] = None,
    districts: Optional[List] = None,
) -> Tuple[
    Dict[str, Tuple[ElectionResult, DeliberationResult]],
    Dict[str, Dict[str, Any]],
]:
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
    voter_responses: Optional voter responses for constituency context.
    districts: Optional list of district dicts for member assignment.

  Returns:
    A tuple of ``(results, government_infos)`` where:
    - ``results``: ``{system_name: (ElectionResult, DeliberationResult)}``
    - ``government_infos``: ``{system_name: {coalition, opposition, ...}}``
  """
  results: Dict[str, Tuple[ElectionResult, DeliberationResult]] = {}
  government_infos: Dict[str, Dict[str, Any]] = {}

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
    delib, gov_info = run_deliberation(
        model=model,
        issue=issue,
        seat_allocation=election.total_seats,
        party_policies=party_policies,
        temperature=temperature,
        max_rounds=deliberation_max_rounds,
        max_workers=max_workers,
        voter_responses=voter_responses,
        districts=districts,
    )

    results[system_name] = (election, delib)
    government_infos[system_name] = gov_info
    adopted = "yes" if delib.adopted_bill else "no"
    logging.info(
        "System '%s': governing=%s, coalition=%s, bill_adopted=%s",
        system_name,
        election.governing_party,
        list(gov_info.get("coalition", {}).keys()),
        adopted,
    )

  # --- Baseline bill (no deliberation) ----------------------------------
  logging.info(
      "=== Comparative: generating baseline bill "
      "(no deliberation) ==="
  )
  try:
    baseline_bill = draft_baseline_bill(
        model=model,
        issue=issue,
        temperature=temperature,
    )
    baseline_delib = DeliberationResult(
        issue=issue,
        adopted_bill=baseline_bill,
    )
  except Exception:  # pylint: disable=broad-except
    logging.exception("Failed to generate baseline bill.")
    baseline_delib = DeliberationResult(issue=issue)

  # Use a sentinel ElectionResult with no seats for the baseline.
  baseline_election = ElectionResult(
      voting_system="baseline",
      district_results=[],
      total_seats={},
      governing_party="none",
  )
  results["baseline"] = (baseline_election, baseline_delib)
  logging.info(
      "Baseline bill adopted: %s",
      "yes" if baseline_delib.adopted_bill else "no",
  )

  # --- Baseline-informed bill (policies + ballots, no election) ----------
  logging.info(
      "=== Comparative: generating baseline-informed bill "
      "(policies + ballots, no election/deliberation) ==="
  )
  try:
    informed_bill = draft_baseline_informed_bill(
        model=model,
        issue=issue,
        party_policies=party_policies,
        ballots=ballots,
        temperature=temperature,
    )
    informed_delib = DeliberationResult(
        issue=issue,
        adopted_bill=informed_bill,
    )
  except Exception:  # pylint: disable=broad-except
    logging.exception(
        "Failed to generate baseline-informed bill."
    )
    informed_delib = DeliberationResult(issue=issue)

  informed_election = ElectionResult(
      voting_system="baseline_informed",
      district_results=[],
      total_seats={},
      governing_party="none",
  )
  results["baseline_informed"] = (
      informed_election, informed_delib
  )
  logging.info(
      "Baseline-informed bill adopted: %s",
      "yes" if informed_delib.adopted_bill else "no",
  )

  return results, government_infos


# ---------------------------------------------------------------------------
# Phase 2: Voter ranking of system policies
# ---------------------------------------------------------------------------


def _query_voter_system_ranking(
    voter_response: VoterResponse,
    system_policies: Dict[str, Optional[str]],
    model: Any,
    temperature: float,
) -> Tuple[str, List[str], Dict[str, float]]:
  """Ask a single voter to rank and score voting systems.

  Returns ``(user_id, [system ranking], {system: score})``.
  """
  demo = voter_response.demographics
  demographics_block = _format_demographics_block(demo)
  examples_str = _format_examples_block(demo.get("examples", []))
  region_block = _format_region_block(voter_response)
  voter_seed = int.from_bytes(
      hashlib.md5(
          f"{voter_response.user_id}:system_rank:"
          f"{voter_response.question}".encode()
      ).digest(),
      "big",
  )
  policies_block = _format_policies_block(
      system_policies, voter_seed=voter_seed
  )
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
        max_tokens=4096,
        temperature=temperature,
        name="system_ranking_json",
    )

  raw = lm["system_ranking_json"]
  text = _strip_llm_wrapping(raw)

  logging.debug(
      "Voter %s system ranking raw (first 300 chars): %.300s",
      voter_response.user_id,
      text,
  )

  # Parse: expect {"ranking": [...], "scores": {...}}
  # Fall back to bare list for backward compat.
  ranking = system_names  # default
  scores: Dict[str, float] = {}  # default

  data = None
  ranking_match = None
  try:
    data = json.loads(text)
    logging.debug(
        "Voter %s parsed JSON type=%s keys=%s",
        voter_response.user_id,
        type(data).__name__,
        list(data.keys())[:5] if isinstance(data, dict) else "N/A",
    )
  except json.JSONDecodeError:
    # The JSON is likely truncated mid-scores.  Try to recover.
    # Strategy 1: try common closing suffixes.
    for suffix in ('}}', '"}', '"}}', '}', ']}'):
      try:
        data = json.loads(text + suffix)
        logging.info(
            "Repaired truncated system ranking JSON for voter %s "
            "(added '%s').",
            voter_response.user_id,
            suffix,
        )
        break
      except json.JSONDecodeError:
        pass

    if data is None:
      # Strategy 2: extract the ranking array even if the outer
      # object is broken.  The ranking is almost always complete;
      # only the scores dict gets truncated.
      ranking_match = re.search(
          r'"ranking"\s*:\s*(\[.*?\])', text, re.DOTALL
      )
      if ranking_match:
        try:
          ranking = json.loads(ranking_match.group(1))
          logging.info(
              "Extracted ranking array from truncated JSON for "
              "voter %s.",
              voter_response.user_id,
          )
        except json.JSONDecodeError:
          pass
      # Also try to salvage partial scores.
      scores_match = re.search(
          r'"scores"\s*:\s*\{(.*)', text, re.DOTALL
      )
      if scores_match:
        partial = scores_match.group(1)
        # Extract key-value pairs that are complete.
        for kv in re.finditer(
            r'"([^"]+)"\s*:\s*([\d.]+)', partial
        ):
          try:
            scores[kv.group(1)] = float(kv.group(2))
          except ValueError:
            pass

    if data is None and not ranking_match:
      logging.warning(
          "Failed to parse system ranking JSON for voter %s."
          " Raw: %.200s",
          voter_response.user_id,
          text,
      )

  if isinstance(data, dict):
    if "ranking" in data or "scores" in data:
      # Expected format: {"ranking": [...], "scores": {...}}
      ranking = data.get("ranking", system_names)
      scores = data.get("scores", scores)
    else:
      # Check if the dict keys are system names → flat scores dict.
      # e.g. {"stv": 4.8, "dhondt": 4.5, ...}
      system_set = set(system_names)
      dict_keys = set(data.keys())
      if dict_keys & system_set:
        logging.info(
            "Voter %s returned flat scores dict (keys: %s).",
            voter_response.user_id,
            sorted(dict_keys & system_set)[:4],
        )
        scores = data
        # Derive ranking from scores (highest first).
        scored = {k: v for k, v in data.items() if k in system_set}
        if scored:
          ranking = sorted(scored, key=lambda k: -scored[k])
      else:
        logging.warning(
            "Voter %s returned dict with unrecognised keys: %s",
            voter_response.user_id,
            sorted(data.keys())[:5],
        )
  elif isinstance(data, list):
    # Legacy bare-array response.
    ranking = data

  # Validate ranking: keep only recognised system names.
  valid = [s for s in ranking if s in system_policies]
  for s in system_names:
    if s not in valid:
      valid.append(s)
  ranking = valid[: len(system_names)]

  # If we have a ranking but no scores, derive scores from position.
  # Maps rank 0 → 5.0, last → 1.0 linearly.
  if not scores and ranking != system_names:
    n = len(ranking)
    for i, s in enumerate(ranking):
      scores[s] = round(5.0 - (4.0 * i / max(n - 1, 1)), 1)
    logging.info(
        "Derived scores from ranking for voter %s.",
        voter_response.user_id,
    )

  # Validate scores: clamp to [1.0, 5.0], round to 0.1.
  validated_scores: Dict[str, float] = {}
  for s in system_names:
    raw_score = scores.get(s)
    if raw_score is not None:
      try:
        val = round(float(raw_score), 1)
        val = max(1.0, min(5.0, val))
      except (ValueError, TypeError):
        val = 3.0  # neutral default
    else:
      val = 3.0  # neutral default
    validated_scores[s] = val

  return (voter_response.user_id, ranking, validated_scores)


def rank_policies(
    voter_responses: List[VoterResponse],
    comparative_results: Dict[
        str, Tuple[ElectionResult, DeliberationResult]
    ],
    model: Any,
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
) -> Tuple[
    Dict[str, List[str]], Dict[str, Dict[str, float]]
]:
  """Ask all voters to rank and score voting systems.

  Each voter is shown the adopted policy from every system's
  deliberation and asked to rank the *systems* from most to least
  preferred, and to score each system on a 1.0–5.0 Likert scale.

  Args:
    voter_responses: Voter responses from the survey phase.
    comparative_results: Output of ``run_comparative_elections()``.
    model: Loaded PathFinder model instance.
    temperature: LLM sampling temperature.
    max_workers: Concurrency limit.

  Returns:
    A tuple of ``(rankings, scores)`` where:
    - ``rankings``: ``{user_id: [system_best, ..., worst]}``
    - ``scores``: ``{user_id: {system: float}}``
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
    all_scores: Dict[str, Dict[str, float]] = {}
    for future in futures.as_completed(future_to_voter):
      vr = future_to_voter[future]
      try:
        uid, ranking, scores = future.result()
        rankings[uid] = ranking
        all_scores[uid] = scores
        logging.info(
            "Voter %s system ranking: %s  scores: %s",
            uid,
            ranking,
            scores,
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
  return rankings, all_scores


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------


def save_rankings(
    rankings: Dict[str, List[str]],
    issue: str,
    model_path: str,
    output_dir: str,
    system_policies: Optional[Dict[str, Optional[str]]] = None,
    party_responses: Optional[
        Dict[str, PolicyResponse]
    ] = None,
    scores: Optional[Dict[str, Dict[str, float]]] = None,
    ballots: Optional[List[VoterBallot]] = None,
    deliberation_rounds: Optional[Dict[str, int]] = None,
    dataset_name: Optional[str] = None,
    voting_mode: Optional[str] = None,
    voter_opinions: Optional[Dict[str, str]] = None,
    government_infos: Optional[Dict[str, Dict[str, Any]]] = None,
) -> str:
  """Persist voter system rankings and scores to JSON.

  The file is created or incrementally updated with the schema::

      {
        "<social issue>": {
          "policies": {
            "<system_name>": "<adopted policy text>" | null
          },
          "deliberation_rounds": {
            "<system_name>": <int>
          },
          "parties": {
            "<ideology>": {
              "position_statement": "...",
              "key_proposals": ["..."]
            }
          },
          "ballots": {
            "<user_id>": {
              "district": "Ironforge Centre",
              "ranking": ["liberal", "socialist", "conservative"]
            }
          },
          "rankings": {
            "<user_id>": ["system_a", "system_b", ...]
          },
          "scores": {
            "<user_id>": {
              "system_a": 4.2,
              "system_b": 2.0
            }
          }
        }
      }

  Args:
    rankings: ``{user_id: [system ranking]}``.
    issue: Social issue text (used as top-level key).
    model_path: Model path — basename used for the filename.
    output_dir: Directory to write the JSON file into.
    system_policies: Optional ``{system_name: policy_text}``.
    party_responses: Optional ``{ideology: PolicyResponse}``.
    scores: Optional ``{user_id: {system: float}}``.
    ballots: Optional list of phase-5 voter party ballots.
    deliberation_rounds: Optional ``{system_name: num_rounds}``.
    dataset_name: Optional shorthand for the social-issues dataset
      (e.g. ``"diverse-12"``).  Included in the filename.
    voting_mode: Optional voting mode (``"issue"`` or
      ``"platform"``).  Included in the filename.
    voter_opinions: Optional ``{user_id: opinion_text}``.  Each
      voter's original free-text response on the issue.

  Returns:
    The absolute path to the written JSON file.
  """
  os.makedirs(output_dir, exist_ok=True)

  # Build filename: [{dataset}.][{mode}.]{model}.json
  parts = []
  if dataset_name:
    parts.append(dataset_name)
  if voting_mode:
    parts.append(voting_mode)
  parts.append(_sanitize_model_name(model_path))
  filename = ".".join(parts) + ".json"
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

  # Store number of deliberation rounds per system.
  if deliberation_rounds is not None:
    existing[issue]["deliberation_rounds"] = deliberation_rounds

  # Store party responses.
  if party_responses is not None:
    parties_dict: Dict[str, Any] = {}
    for ideology, pr in party_responses.items():
      parties_dict[ideology] = {
          "position_statement": pr.position_statement,
          "key_proposals": pr.key_proposals,
      }
    existing[issue]["parties"] = parties_dict



  # Store voter party ballots (phase 5 rankings).
  if ballots is not None:
    ballots_dict: Dict[str, Any] = {}
    for b in ballots:
      ballots_dict[b.user_id] = {
          "district": b.district_name,
          "ranking": b.ranking,
      }
    existing[issue]["ballots"] = ballots_dict

  # Store voter system rankings.
  existing[issue]["rankings"] = rankings

  # Store Likert scores per voter per system.
  if scores is not None:
    existing[issue]["scores"] = scores

  # Store voter opinions (free-text responses on the issue).
  if voter_opinions is not None:
    existing[issue]["opinions"] = voter_opinions

  # Store government/coalition info per voting system.
  if government_infos is not None:
    existing[issue]["government"] = government_infos

  with open(filepath, "w") as f:
    json.dump(existing, f, indent=2, ensure_ascii=False)

  logging.info(
      "Saved rankings to %s (%d voters for issue '%.60s').",
      filepath,
      len(rankings),
      issue,
  )
  return filepath
