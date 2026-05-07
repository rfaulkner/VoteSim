"""Compare mode: ask voters to choose between issue-mode and platform-mode bills.

Reads the two pre-computed results files (``{dataset}.issue.{model}.json``
and ``{dataset}.platform.{model}.json``), finds common issues and voters,
loads personas, and concurrently asks each voter which bill they prefer.

Results are written to ``{dataset}.compare.{model}.json``.
"""

from collections import Counter
from concurrent import futures
import hashlib
import json
import logging
import os
import random
from typing import Any
from typing import Dict
from typing import Optional
from typing import Tuple

from omegaconf import DictConfig
from pathfinder import assistant
from pathfinder import gen
from pathfinder import get_model
from pathfinder import user
from simulation.policy_ranking import _sanitize_model_name
from simulation.policy_ranking import _strip_llm_wrapping
from simulation.survey import _format_demographics_block
from simulation.survey import _format_examples_block

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

BILL_COMPARISON_PROMPT = """\
You are a person with the following background:
{demographics_block}
{region_block}
Here is how you describe your own values and outlook:
  "{self_description}"

Here are some statements you have made in the past that reflect your views:
{examples}

You are being asked about the following social issue:
  "{question}"
{opinion_block}
Two different democratic processes were used to form governments and pass \
legislation on this issue.  Below are the two bills that were adopted.  \
Compare them against your own views and values.

=== Bill A ===
{bill_a}

=== Bill B ===
{bill_b}

Which bill better reflects your preferences and values?  You MUST choose \
one.  Return ONLY a JSON object:
{{"preferred": "A"}} or {{"preferred": "B"}}

Return ONLY the JSON object — no other text."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_region_block_from_district(
    district: Optional[Dict[str, Any]],
) -> str:
  """Build a region/district context block from a persona's district."""
  if not district:
    return ""
  return (
      f"\nYou live in the district of '{district['name']}'.\n"
      "District Characteristics:\n"
      f"- Wealth: {district.get('wealth', 'unknown')} (0-1 scale)\n"
      f"- Urbanisation: {district.get('urbanisation', 'unknown')}"
      " (0-1 scale)\n"
      f"- Primary Industry: {district.get('industry', 'unknown')}\n"
      f"- Political Leaning:"
      f" {district.get('political_leaning', 'unknown')}\n"
      f"- Description: {district.get('description', '')}\n"
  )


def _query_bill_preference(
    user_id: str,
    demographics: Dict[str, Any],
    district: Optional[Dict[str, Any]],
    question: str,
    opinion: Optional[str],
    bill_issue: str,
    bill_platform: str,
    system_issue: str,
    system_platform: str,
    model: Any,
    temperature: float,
) -> Tuple[str, str, str]:
  """Ask a voter which bill they prefer.

  Returns ``(user_id, preferred_mode, preferred_system)`` where
  ``preferred_mode`` is ``"issue"`` or ``"platform"``.
  """
  demographics_block = _format_demographics_block(demographics)
  examples_str = _format_examples_block(
      demographics.get("examples", [])
  )
  region_block = _format_region_block_from_district(district)
  self_desc = demographics.get("self_description", "")

  # Build opinion block if available.
  if opinion:
    opinion_block = (
        f'\nYou previously shared your opinion on this issue:\n'
        f'  "{opinion}"\n'
    )
  else:
    opinion_block = ""

  # Randomly assign bills to A/B to avoid position bias.
  seed_val = int.from_bytes(
      hashlib.md5(
          f"{user_id}:compare:{question}".encode()
      ).digest(),
      "big",
  )
  rng = random.Random(seed_val)
  if rng.random() < 0.5:
    # A = issue, B = platform
    bill_a, bill_b = bill_issue, bill_platform
    mapping = {"A": ("issue", system_issue),
               "B": ("platform", system_platform)}
  else:
    # A = platform, B = issue
    bill_a, bill_b = bill_platform, bill_issue
    mapping = {"A": ("platform", system_platform),
               "B": ("issue", system_issue)}

  prompt = BILL_COMPARISON_PROMPT.format(
      demographics_block=demographics_block,
      region_block=region_block,
      self_description=self_desc,
      examples=examples_str,
      question=question,
      opinion_block=opinion_block,
      bill_a=bill_a,
      bill_b=bill_b,
  )

  lm = model.copy()
  with user():
    lm += prompt
  with assistant():
    lm += gen(
        max_tokens=64,
        temperature=temperature,
        name="bill_preference_json",
    )

  raw = lm["bill_preference_json"]
  text = _strip_llm_wrapping(raw)

  try:
    data = json.loads(text)
    preferred = data.get("preferred", "A").upper()
  except (json.JSONDecodeError, AttributeError):
    logging.warning(
        "Failed to parse bill preference for voter %s. Raw: %.200s",
        user_id,
        text,
    )
    preferred = "A"  # default fallback

  if preferred not in mapping:
    preferred = "A"

  mode, system_name = mapping[preferred]
  return (user_id, mode, system_name)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def run_compare(cfg: DictConfig):
  """Run the compare mode: ask voters to choose between issue/platform bills.

  Reads the two results files, finds common issues and voters, loads
  personas, and concurrently queries each voter for their bill preference.

  Args:
    cfg: Hydra configuration dictionary.
  """
  # --- Config extraction ---------------------------------------------------
  model_path = cfg.llm.path
  is_api = cfg.llm.get("is_api", False)
  backend = cfg.llm.get("backend", "transformers")
  seed = cfg.get("seed", 42)
  temperature = cfg.llm.get("temperature", 0.0)

  pipe_cfg = cfg.get("pipeline", {})
  results_dir = pipe_cfg.get("results_dir", "results")
  max_workers = pipe_cfg.get("max_workers", 5)
  personas_path = pipe_cfg.get(
      "personas_path", "dataset/personas/personas.json"
  )

  pq_cfg = cfg.get("political_questions", {})
  dataset_name = pq_cfg.get("dataset", "")

  # --- Resolve input file paths -------------------------------------------
  model_slug = _sanitize_model_name(model_path)

  def _build_path(mode: str) -> str:
    parts = []
    if dataset_name:
      parts.append(dataset_name)
    parts.append(mode)
    parts.append(model_slug)
    return os.path.join(results_dir, ".".join(parts) + ".json")

  issue_path = _build_path("issue")
  platform_path = _build_path("platform")

  for path, label in [(issue_path, "issue"), (platform_path, "platform")]:
    if not os.path.exists(path):
      raise FileNotFoundError(
          f"{label}-mode results file not found: {path}.  "
          f"Run the pipeline in '{label}' mode first."
      )

  with open(issue_path) as f:
    issue_data = json.load(f)
  with open(platform_path) as f:
    platform_data = json.load(f)

  logging.info(
      "Loaded issue results (%d issues) from %s",
      len(issue_data),
      issue_path,
  )
  logging.info(
      "Loaded platform results (%d issues) from %s",
      len(platform_data),
      platform_path,
  )

  # --- Find common issues --------------------------------------------------
  common_issues = sorted(set(issue_data.keys()) & set(platform_data.keys()))
  if not common_issues:
    logging.error("No common issues found between the two results files.")
    return

  logging.info(
      "Found %d common issue(s) for comparison.", len(common_issues)
  )

  # --- Load personas -------------------------------------------------------
  with open(personas_path) as f:
    personas_list = json.load(f)
  personas = {p["user_id"]: p for p in personas_list}
  logging.info("Loaded %d personas from %s.", len(personas), personas_path)

  # --- Load model ----------------------------------------------------------
  model = get_model(
      model_path, is_api=is_api, seed=seed, backend_name=backend
  )

  # --- Process each issue --------------------------------------------------
  all_results: Dict[str, Any] = {}

  for qi, question in enumerate(common_issues):
    logging.info(
        "=== Compare %d/%d: %.70s ===",
        qi + 1,
        len(common_issues),
        question,
    )

    ie = issue_data[question]
    pe = platform_data[question]

    # Find common voters with rankings in both modes.
    common_voters = sorted(
        set(ie.get("rankings", {}).keys())
        & set(pe.get("rankings", {}).keys())
    )
    logging.info("  %d common voters for this issue.", len(common_voters))

    if not common_voters:
      continue

    # Prepare per-voter arguments.
    issue_policies = ie.get("policies", {})
    platform_policies = pe.get("policies", {})
    issue_opinions = ie.get("opinions", {})
    platform_opinions = pe.get("opinions", {})

    voter_args = []
    for uid in common_voters:
      # Get top-ranked system from each mode.
      i_ranking = ie["rankings"].get(uid, [])
      p_ranking = pe["rankings"].get(uid, [])
      if not i_ranking or not p_ranking:
        continue

      top_issue_sys = i_ranking[0]
      top_plat_sys = p_ranking[0]

      bill_issue = issue_policies.get(top_issue_sys)
      bill_platform = platform_policies.get(top_plat_sys)

      if not bill_issue or not bill_platform:
        continue

      # Get persona.
      persona = personas.get(uid)
      if not persona:
        logging.warning("  No persona found for %s, skipping.", uid)
        continue

      # Prefer issue-mode opinion, fall back to platform-mode opinion.
      opinion = issue_opinions.get(uid) or platform_opinions.get(uid)

      voter_args.append((
          uid,
          persona.get("demographics", {}),
          persona.get("district"),
          question,
          opinion,
          bill_issue,
          bill_platform,
          top_issue_sys,
          top_plat_sys,
      ))

    # Concurrent voter queries.
    mode_counts = Counter()  # "issue" | "platform" -> count
    issue_sys_counts = Counter()  # system -> count (when issue chosen)
    plat_sys_counts = Counter()  # system -> count (when platform chosen)
    total = 0

    with futures.ThreadPoolExecutor(
        max_workers=max_workers
    ) as executor:
      future_map = {}
      for args in voter_args:
        future = executor.submit(
            _query_bill_preference,
            *args,
            model=model,
            temperature=temperature,
        )
        future_map[future] = args[0]  # user_id

      for future in futures.as_completed(future_map):
        uid = future_map[future]
        try:
          _, preferred_mode, preferred_sys = future.result()
          mode_counts[preferred_mode] += 1
          if preferred_mode == "issue":
            issue_sys_counts[preferred_sys] += 1
          else:
            plat_sys_counts[preferred_sys] += 1
          total += 1
        except Exception as e:
          logging.error(
              "  Failed to get preference from voter %s: %s",
              uid,
              e,
          )

    if total == 0:
      logging.warning("  No valid preferences collected for this issue.")
      continue

    # Build per-issue result.
    issue_rate = mode_counts.get("issue", 0) / total
    plat_rate = mode_counts.get("platform", 0) / total

    issue_total = mode_counts.get("issue", 0)
    plat_total = mode_counts.get("platform", 0)

    all_results[question] = {
        "issue_voting": {
            "preference_rate": round(issue_rate, 4),
            "systems": {
                s: round(c / issue_total, 4)
                for s, c in issue_sys_counts.most_common()
            } if issue_total > 0 else {},
        },
        "platform_voting": {
            "preference_rate": round(plat_rate, 4),
            "systems": {
                s: round(c / plat_total, 4)
                for s, c in plat_sys_counts.most_common()
            } if plat_total > 0 else {},
        },
        "num_voters": total,
    }

    logging.info(
        "  Issue: %.1f%% (%d) | Platform: %.1f%% (%d)",
        issue_rate * 100,
        mode_counts.get("issue", 0),
        plat_rate * 100,
        mode_counts.get("platform", 0),
    )

  # --- Write output --------------------------------------------------------
  out_parts = []
  if dataset_name:
    out_parts.append(dataset_name)
  out_parts.append("compare")
  out_parts.append(model_slug)
  out_filename = ".".join(out_parts) + ".json"
  out_path = os.path.join(results_dir, out_filename)

  os.makedirs(results_dir, exist_ok=True)
  with open(out_path, "w") as f:
    json.dump(all_results, f, indent=2, ensure_ascii=False)

  logging.info(
      "Comparison results saved to %s (%d issues).",
      out_path,
      len(all_results),
  )
  return out_path
