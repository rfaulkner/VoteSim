"""Voting-round pipeline: voter survey → party policy responses.

Orchestrates a single round of the VoteSim experiment:
1. Optionally load or generate a ``Region`` with districts.
2. Sample a social-issue question from ``PoliticalQuestionSampler``.
3. Sample *k* voters from PRISM, assign them to districts, and concurrently
   elicit their opinions on the issue.
4. Concurrently generate policy responses from every party platform,
   conditioned on the voter responses, the issue, and regional context.
"""

import logging
import os

from dataclasses import dataclass
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from pathfinder import get_model
from simulation.district_generator import Region
from simulation.district_generator import RegionGenerator
from simulation.policy_generator import PolicyResponse
from simulation.political_sampler import PoliticalQuestionSampler
from simulation.prism_sampler import PrismSampler
from simulation.survey import VoterResponse
from simulation.survey import generate_party_responses
from simulation.survey import generate_voter_responses


@dataclass
class VotingRoundResult:
  """Container for the outputs of a single voting round."""

  question: str
  voter_responses: List[VoterResponse]
  party_responses: Dict[str, PolicyResponse]
  region: Optional[Region] = None

  def summary(self) -> str:
    """Return a human-readable summary of the round."""
    lines = [
        "=" * 60,
        "VOTING ROUND SUMMARY",
        "=" * 60,
        f"\nSocial Issue:\n  {self.question}",
    ]

    if self.region:
      lines.append(
          f"\nRegion: {self.region.region_name}"
          f" ({len(self.region.districts)} districts)"
      )

    lines.append(f"\n--- Voter Responses ({len(self.voter_responses)}) ---")
    for i, vr in enumerate(self.voter_responses, 1):
      district_name = vr.district["name"] if vr.district else "N/A"
      lines.append(f"\n  Voter {i} [{vr.user_id}] (district: {district_name})")
      lines.append(f"    {vr.response}")

    lines.append(
        f"\n--- Party Responses ({len(self.party_responses)}) ---"
    )
    for ideology in sorted(self.party_responses):
      pr = self.party_responses[ideology]
      lines.append(f"\n  {pr.party_name} ({pr.ideology})")
      lines.append(f"    Position: {pr.position_statement}")
      lines.append(f"    Proposals:")
      for proposal in pr.key_proposals:
        lines.append(f"      - {proposal}")
      lines.append(
          f"    Voter alignment: {pr.voter_alignment_score:.2f}"
      )
      lines.append(f"    Reasoning: {pr.reasoning}")

    lines.append("\n" + "=" * 60)
    return "\n".join(lines)


def _load_or_generate_region(
    region_description: str,
    num_districts: int,
    model_path: str,
    is_api: bool,
    seed: int,
    backend: str,
    cache_path: Optional[str] = None,
) -> Region:
  """Load a cached region or generate a new one via LLM."""
  region_file = (
      os.path.join(cache_path, "region.json") if cache_path else None
  )

  if region_file and os.path.exists(region_file):
    logging.info("Loading cached region from %s", region_file)
    return Region.from_json(region_file)

  logging.info("Generating new region (%d districts)...", num_districts)
  generator = RegionGenerator(
      model_path, is_api=is_api, seed=seed, backend=backend
  )
  region = generator.generate(region_description, num_districts)

  if region_file:
    region.to_json(region_file)

  return region


def run_voting_round(
    num_voters: int,
    model_path: str,
    prism_dataset_dir: str = "dataset/prism",
    political_questions_path: str = (
        "dataset/political_questions/political-questions.csv"
    ),
    party_dir: str = "dataset/party",
    topic: str = "social",
    seed: int = 42,
    is_api: bool = False,
    backend: str = "transformers",
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
    region_description: Optional[str] = None,
    num_districts: int = 5,
    region_cache_path: Optional[str] = None,
    model: Optional[Any] = None,
    parties: Optional[List[str]] = None,
) -> VotingRoundResult:
  """Execute a full voting round: voters respond → parties respond.

  Args:
    num_voters: Number of voters to sample from PRISM.
    model_path: Path or name of the LLM to use.
    prism_dataset_dir: Path to the PRISM dataset directory.
    political_questions_path: Path to the political-questions CSV.
    party_dir: Directory containing party platform JSON files.
    topic: Topic category to filter questions by (e.g. ``"social"``).
    seed: Random seed for all sampling and generation (reproducibility).
    is_api: Whether the model is accessed via API.
    backend: LLM backend name.
    temperature: Sampling temperature for LLM generation.
    max_workers: Max concurrent LLM calls per stage.
    region_description: If provided, a region is loaded/generated and voters are
      assigned districts.  Example: ``"the Midwestern United States"``.
    num_districts: Number of districts in the generated region.
    region_cache_path: Directory to cache the generated region JSON.
    model: Optional pre-loaded PathFinder model instance.  If ``None``,
      the model is loaded once from ``model_path``.
    parties: Optional list of ideology names to include (e.g.
      ``["liberal", "conservative"]``).  If ``None``, all platforms are used.

  Returns:
    A ``VotingRoundResult`` containing voter responses, party responses,
    the sampled question, and the region (if any).
  """
  # -- 0. Model (load once) ------------------------------------------------
  if model is None:
    logging.info("Loading model '%s' (backend=%s)...", model_path, backend)
    model = get_model(
        model_path, is_api=is_api, seed=seed, backend_name=backend
    )

  # -- 1. Region -----------------------------------------------------------
  region: Optional[Region] = None
  if region_description:
    region = _load_or_generate_region(
        region_description=region_description,
        num_districts=num_districts,
        model_path=model_path,
        is_api=is_api,
        seed=seed,
        backend=backend,
        cache_path=region_cache_path,
    )

  # -- 2. Sample question --------------------------------------------------
  pq_sampler = PoliticalQuestionSampler(political_questions_path)
  question = pq_sampler.sample_question_text(topic=topic, seed=seed)
  logging.info("Sampled question [%s]: %s", topic, question)

  # -- 3. Voter survey (concurrent) ----------------------------------------
  prism = PrismSampler(prism_dataset_dir)
  voter_responses = generate_voter_responses(
      k=num_voters,
      question=question,
      prism_sampler=prism,
      model=model,
      seed=seed,
      temperature=temperature,
      max_workers=max_workers,
      region=region,
  )

  # -- 4. Party responses (concurrent) ------------------------------------
  party_responses = generate_party_responses(
      issue=question,
      voter_responses=voter_responses,
      party_dir=party_dir,
      model=model,
      temperature=temperature,
      max_workers=max_workers,
      region=region,
      parties=parties,
  )

  result = VotingRoundResult(
      question=question,
      voter_responses=voter_responses,
      party_responses=party_responses,
      region=region,
  )
  logging.info("Voting round complete.")
  return result

