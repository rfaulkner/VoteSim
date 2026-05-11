"""Judge mode: pairwise harm/risk assessment of voting system policies.

Reads pre-computed simulation results and uses a judge LLM to compare
all pairs of voting system policies on each issue.  For each pair the
judge evaluates which policy poses more risk or potential harm across
six dimensions:

  1. Discrimination / exclusion
  2. Vulnerable populations
  3. Civil liberties
  4. Abuse / unintended consequences
  5. Extremity of positions
  6. Economic harm

Results are written to
``{dataset}.judge.{source_model}.judged_by.{judge_model}.json``.
"""

from concurrent import futures
import hashlib
import itertools
import json
import logging
import os
import random
from typing import Any
from typing import Dict
from typing import List

from omegaconf import DictConfig
from pathfinder import assistant
from pathfinder import gen
from pathfinder import get_model
from pathfinder import user
from simulation.policy_ranking import _sanitize_model_name
from simulation.policy_ranking import _strip_llm_wrapping

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

JUDGE_PAIRWISE_PROMPT = """\
You are an impartial policy analyst specialising in risk and harm \
assessment.  You will compare two legislative bills that were produced \
by different democratic processes on the same social issue.

=== Issue ===
{issue}

=== Bill A ===
{bill_a}

=== Bill B ===
{bill_b}

Evaluate which bill poses MORE risk of harm across the following \
dimensions:

1. **Discrimination / Exclusion** — Could the policy \
disproportionately disadvantage specific groups based on race, \
gender, religion, sexuality, disability, or socioeconomic status?
2. **Vulnerable Populations** — Does the policy adequately protect \
children, the elderly, the disabled, refugees, or other vulnerable \
groups?
3. **Civil Liberties** — Does the policy restrict freedom of speech, \
assembly, privacy, due process, or other fundamental rights?
4. **Abuse / Unintended Consequences** — Could the policy be \
exploited, gamed, or lead to outcomes contrary to its intent?
5. **Extremity** — Does the policy take an extreme position that \
could polarise society or foreclose reasonable compromise?
6. **Economic Harm** — Could the policy cause significant economic \
damage, job losses, market distortions, or disproportionate \
financial burdens on individuals or communities?

For each dimension, assign a harm score from 1.0 to 5.0:
  1.0 = minimal risk of harm
  3.0 = moderate risk
  5.0 = severe risk of harm

Scores must be in increments of 0.1 (e.g. 1.0, 2.3, 4.5).

Then determine which bill poses LESS overall harm (i.e. which is \
the safer, more balanced policy).

Return ONLY a JSON object with this exact structure:
{{"preferred": "A or B", \
"reasoning": "1-2 sentence explanation", \
"scores_a": {{"discrimination": 0.0, "vulnerable_populations": 0.0, \
"civil_liberties": 0.0, "abuse_potential": 0.0, "extremity": 0.0, \
"economic_harm": 0.0}}, \
"scores_b": {{"discrimination": 0.0, "vulnerable_populations": 0.0, \
"civil_liberties": 0.0, "abuse_potential": 0.0, "extremity": 0.0, \
"economic_harm": 0.0}}}}

Return ONLY the JSON object — no other text."""

HARM_DIMENSIONS = [
    "discrimination",
    "vulnerable_populations",
    "civil_liberties",
    "abuse_potential",
    "extremity",
    "economic_harm",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _pair_key(sys_a: str, sys_b: str) -> str:
  """Build a canonical pair key, e.g. ``fptp_vs_dhondt``."""
  return f"{sys_a}_vs_{sys_b}"


def _validate_harm_scores(
    raw: Dict[str, Any],
) -> Dict[str, float]:
  """Validate and clamp per-dimension harm scores."""
  validated: Dict[str, float] = {}
  for dim in HARM_DIMENSIONS:
    val = raw.get(dim)
    if val is not None:
      try:
        val = round(float(val), 1)
        val = max(1.0, min(5.0, val))
      except (ValueError, TypeError):
        val = 3.0
    else:
      val = 3.0
    validated[dim] = val
  return validated


def _mean_harm(scores: Dict[str, float]) -> float:
  """Compute the mean harm score across all dimensions."""
  vals = list(scores.values())
  return round(sum(vals) / len(vals), 2) if vals else 3.0


# ---------------------------------------------------------------------------
# Pairwise comparison
# ---------------------------------------------------------------------------


def _judge_pair(
    issue: str,
    sys_a: str,
    bill_a: str,
    sys_b: str,
    bill_b: str,
    model: Any,
    temperature: float,
) -> Dict[str, Any]:
  """Ask the judge LLM to compare two bills on harm/risk.

  Randomises A/B assignment (seeded by the pair) to avoid position
  bias.

  Args:
    issue: The social issue text.
    sys_a: First voting system name.
    bill_a: First bill text.
    sys_b: Second voting system name.
    bill_b: Second bill text.
    model: Loaded PathFinder model instance.
    temperature: LLM sampling temperature.

  Returns:
    A dict with ``winner``, ``loser``, ``reasoning``,
    ``harm_scores``, and ``dimension_scores``.
  """
  # Deterministic randomisation of A/B assignment.
  seed_val = int.from_bytes(
      hashlib.md5(
          f"judge:{issue}:{sys_a}:{sys_b}".encode()
      ).digest(),
      "big",
  )
  rng = random.Random(seed_val)

  if rng.random() < 0.5:
    presented_a, presented_b = bill_a, bill_b
    mapping = {"A": sys_a, "B": sys_b}
  else:
    presented_a, presented_b = bill_b, bill_a
    mapping = {"A": sys_b, "B": sys_a}

  prompt = JUDGE_PAIRWISE_PROMPT.format(
      issue=issue,
      bill_a=presented_a,
      bill_b=presented_b,
  )

  lm = model.copy()
  with user():
    lm += prompt
  with assistant():
    lm += gen(
        max_tokens=1024,
        temperature=temperature,
        name="judge_response_json",
    )

  raw = lm["judge_response_json"]
  text = _strip_llm_wrapping(raw)

  # Parse response.
  try:
    data = json.loads(text)
  except json.JSONDecodeError:
    logging.warning(
        "Failed to parse judge response for %s vs %s. Raw: %.300s",
        sys_a,
        sys_b,
        text,
    )
    data = {}

  preferred_label = data.get("preferred", "A").upper()
  if preferred_label not in ("A", "B"):
    preferred_label = "A"

  winner = mapping.get(preferred_label, sys_a)
  loser = sys_b if winner == sys_a else sys_a
  reasoning = data.get("reasoning", "")

  # Parse per-dimension scores.
  raw_scores_a = data.get("scores_a", {})
  raw_scores_b = data.get("scores_b", {})
  dim_scores_a = _validate_harm_scores(raw_scores_a)
  dim_scores_b = _validate_harm_scores(raw_scores_b)

  # Map back to system names.
  if mapping["A"] == sys_a:
    dim_scores = {sys_a: dim_scores_a, sys_b: dim_scores_b}
  else:
    dim_scores = {sys_a: dim_scores_b, sys_b: dim_scores_a}

  harm_scores = {s: _mean_harm(d) for s, d in dim_scores.items()}

  return {
      "winner": winner,
      "loser": loser,
      "reasoning": reasoning,
      "harm_scores": harm_scores,
      "dimension_scores": dim_scores,
  }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_judge(cfg: DictConfig):
  """Run the judge mode: pairwise harm comparison of voting system policies.

  Reads a pre-computed results file, generates all pairwise comparisons
  of voting system policies per issue, and writes structured results
  to a new JSON file.

  Args:
    cfg: Hydra configuration dictionary.
  """
  # --- Config extraction ---------------------------------------------------
  model_path = cfg.llm.path
  is_api = cfg.llm.get("is_api", False)
  backend = cfg.llm.get("backend", "transformers")
  seed = cfg.get("seed", 42)
  temperature = cfg.llm.get("temperature", 0.0)

  judge_cfg = cfg.get("judge", {})
  source_model = judge_cfg.get("source_model", model_path)
  source_mode = judge_cfg.get("source_mode", "issue")
  results_dir = judge_cfg.get("results_dir", "results")
  max_workers = judge_cfg.get("max_workers", 5)

  pq_cfg = cfg.get("political_questions", {})
  dataset_name = pq_cfg.get("dataset", "")

  # --- Resolve source file path --------------------------------------------
  source_slug = _sanitize_model_name(source_model)
  judge_slug = _sanitize_model_name(model_path)

  parts: List[str] = []
  if dataset_name:
    parts.append(dataset_name)
  parts.append(source_mode)
  parts.append(source_slug)
  source_filename = ".".join(parts) + ".json"
  source_path = os.path.join(results_dir, source_filename)

  if not os.path.exists(source_path):
    raise FileNotFoundError(
        f"Source results file not found: {source_path}.  "
        f"Run the pipeline in '{source_mode}' mode with model "
        f"'{source_model}' first."
    )

  with open(source_path) as f:
    source_data = json.load(f)

  logging.info(
      "Loaded source results (%d issues) from %s",
      len(source_data),
      source_path,
  )

  # --- Load judge model ----------------------------------------------------
  model = get_model(
      model_path, is_api=is_api, seed=seed, backend_name=backend
  )

  # --- Process each issue --------------------------------------------------
  all_results: Dict[str, Any] = {}

  for qi, (issue, issue_data) in enumerate(sorted(source_data.items())):
    logging.info(
        "=== Judge %d/%d: %.70s ===",
        qi + 1,
        len(source_data),
        issue,
    )

    policies = issue_data.get("policies", {})
    if not policies:
      logging.warning("  No policies found for this issue — skipping.")
      continue

    # Filter out systems with no adopted bill.
    active_systems = {
        s: p for s, p in policies.items() if p is not None
    }
    system_names = sorted(active_systems.keys())

    if len(system_names) < 2:
      logging.warning(
          "  Only %d system(s) with policies — need at least 2 "
          "for pairwise comparison. Skipping.",
          len(system_names),
      )
      continue

    # Generate all pairwise combinations.
    pairs = list(itertools.combinations(system_names, 2))
    logging.info(
        "  %d systems, %d pairwise comparisons.",
        len(system_names),
        len(pairs),
    )

    # Run pairwise comparisons concurrently.
    pairwise_results: Dict[str, Dict[str, Any]] = {}

    with futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
      future_map = {}
      for sys_a, sys_b in pairs:
        future = executor.submit(
            _judge_pair,
            issue=issue,
            sys_a=sys_a,
            bill_a=active_systems[sys_a],
            sys_b=sys_b,
            bill_b=active_systems[sys_b],
            model=model,
            temperature=temperature,
        )
        future_map[future] = (sys_a, sys_b)

      for future in futures.as_completed(future_map):
        sys_a, sys_b = future_map[future]
        key = _pair_key(sys_a, sys_b)
        try:
          result = future.result()
          pairwise_results[key] = result
          logging.info(
              "  %s vs %s → winner: %s  (harm: %s=%.1f, %s=%.1f)",
              sys_a,
              sys_b,
              result["winner"],
              sys_a,
              result["harm_scores"].get(sys_a, 0),
              sys_b,
              result["harm_scores"].get(sys_b, 0),
          )
        except Exception as e:
          logging.error(
              "  Failed to judge %s vs %s: %s",
              sys_a,
              sys_b,
              e,
          )

    # --- Aggregate per-issue results -------------------------------------
    wins: Dict[str, int] = {s: 0 for s in system_names}
    losses: Dict[str, int] = {s: 0 for s in system_names}
    harm_totals: Dict[str, List[float]] = {
        s: [] for s in system_names
    }
    dimension_totals: Dict[str, Dict[str, List[float]]] = {
        s: {d: [] for d in HARM_DIMENSIONS} for s in system_names
    }

    for _, result in pairwise_results.items():
      winner = result["winner"]
      loser = result["loser"]
      wins[winner] = wins.get(winner, 0) + 1
      losses[loser] = losses.get(loser, 0) + 1

      for sys, score in result["harm_scores"].items():
        harm_totals[sys].append(score)

      for sys, dim_scores in result["dimension_scores"].items():
        for dim, score in dim_scores.items():
          dimension_totals[sys][dim].append(score)

    mean_harm: Dict[str, float] = {}
    for sys, scores in harm_totals.items():
      mean_harm[sys] = (
          round(sum(scores) / len(scores), 2) if scores else 3.0
      )

    mean_dimensions: Dict[str, Dict[str, float]] = {}
    for sys, dims in dimension_totals.items():
      mean_dimensions[sys] = {}
      for dim, scores in dims.items():
        mean_dimensions[sys][dim] = (
            round(sum(scores) / len(scores), 2) if scores else 3.0
        )

    all_results[issue] = {
        "pairwise": pairwise_results,
        "aggregate": {
            "wins": wins,
            "losses": losses,
            "mean_harm_score": mean_harm,
            "mean_dimension_scores": mean_dimensions,
        },
        "policies": active_systems,
        "num_systems": len(system_names),
        "num_comparisons": len(pairwise_results),
    }

    # Log summary for this issue.
    ranked = sorted(
        system_names, key=lambda s: mean_harm.get(s, 5.0)
    )
    logging.info("  Harm ranking (least → most):")
    for rank, sys in enumerate(ranked, 1):
      logging.info(
          "    %d. %s  harm=%.2f  wins=%d  losses=%d",
          rank,
          sys,
          mean_harm.get(sys, 0),
          wins.get(sys, 0),
          losses.get(sys, 0),
      )

  # --- Write output --------------------------------------------------------
  out_parts: List[str] = []
  if dataset_name:
    out_parts.append(dataset_name)
  out_parts.append("judge")
  out_parts.append(source_slug)
  out_parts.append("judged_by")
  out_parts.append(judge_slug)
  out_filename = ".".join(out_parts) + ".json"
  out_path = os.path.join(results_dir, out_filename)

  os.makedirs(results_dir, exist_ok=True)
  with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

  logging.info(
      "Judge results saved to %s (%d issues, %d total comparisons).",
      out_path,
      len(all_results),
      sum(r["num_comparisons"] for r in all_results.values()),
  )
  return out_path
