"""Voting-round pipeline: survey → rank → election → deliberation → compare.

Orchestrates a single round of the VoteSim experiment:
1. Optionally load or generate a ``Region`` with districts.
2. Sample a social-issue question from ``PoliticalQuestionSampler``.
3. Sample *k* voters from PRISM, assign them to districts, and
   concurrently elicit their opinions on the issue.
4. Concurrently generate policy responses from every party platform,
   conditioned on the voter responses, the issue, and regional
   context.
5. Each voter ranks the parties based on their policy proposals.
6. Apply a voting system to determine seat allocation and the
   governing party.
7. Parliamentary deliberation — bill drafting, amendment, and vote.
8. Comparative policy ranking — run all configured voting systems,
   deliberate under each, and ask voters to rank the resulting
   policies.  Results are persisted as per-model JSON.
"""

from dataclasses import dataclass
from dataclasses import field
import json
import logging
import os
from typing import Any
from typing import Dict
from typing import List
from typing import Optional

from simulation.deliberate import DeliberationResult
from simulation.deliberate import run_deliberation
from simulation.policy_ranking import _bill_to_text
from simulation.policy_ranking import rank_policies
from simulation.policy_ranking import run_comparative_elections
from simulation.policy_ranking import save_rankings

from pathfinder import get_model
from simulation.district_generator import Region
from simulation.district_generator import RegionGenerator
from simulation.policy_generator import PartyPlatform
from simulation.policy_generator import PolicyResponse
from simulation.political_sampler import PoliticalQuestionSampler
from simulation.prism_sampler import load_or_create_personas
from simulation.prism_sampler import PrismSampler
from simulation.survey import generate_party_responses
from simulation.survey import generate_platform_rankings
from simulation.survey import generate_voter_rankings
from simulation.survey import generate_voter_responses
from simulation.survey import platform_to_policy_response
from simulation.survey import prepare_voters
from simulation.survey import survey_voters
from simulation.survey import VoterResponse
from simulation.voting import allocate_district_seats
from simulation.voting import ElectionResult
from simulation.voting import run_election
from simulation.voting import VoterBallot


def _read_district_seats(
    districts: List[Dict[str, Any]],
    max_seats_per_district: Optional[int] = None,
) -> Dict[str, int]:
  """Read seat counts from district dicts.

  Uses the ``"seats"`` key if present in each district, falling
  back to ``allocate_district_seats()`` otherwise.

  Args:
    districts: List of district dicts from a loaded region.
    max_seats_per_district: If set, cap each district to at most
      this many seats.

  Returns:
    ``{district_name: num_seats}``
  """
  if all("seats" in d for d in districts):
    seats = {d["name"]: d["seats"] for d in districts}
  else:
    seats = allocate_district_seats(districts)
  if max_seats_per_district is not None:
    seats = {k: min(v, max_seats_per_district) for k, v in seats.items()}
  return seats


@dataclass
class VotingRoundResult:
  """Container for the outputs of a single voting round."""

  question: str
  voter_responses: List[VoterResponse]
  party_responses: Dict[str, PolicyResponse]
  ballots: List[VoterBallot] = field(default_factory=list)
  election: Optional[ElectionResult] = None
  region: Optional[Region] = None
  deliberation: Optional[DeliberationResult] = None
  comparative_rankings: Optional[Dict[str, List[str]]] = None
  rankings_file: Optional[str] = None

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

    lines.append(f"\n--- Party Responses ({len(self.party_responses)}) ---")
    for ideology in sorted(self.party_responses):
      pr = self.party_responses[ideology]
      lines.append(f"\n  {pr.party_name} ({pr.ideology})")
      lines.append(f"    Position: {pr.position_statement}")
      lines.append("    Proposals:")
      for proposal in pr.key_proposals:
        lines.append(f"      - {proposal}")
      lines.append(f"    Voter alignment: {pr.voter_alignment_score:.2f}")
      lines.append(f"    Reasoning: {pr.reasoning}")

    if self.ballots:
      lines.append(f"\n--- Voter Rankings ({len(self.ballots)}) ---")
      for ballot in self.ballots:
        lines.append(
            f"  {ballot.user_id} ({ballot.district_name}): "
            f"{' > '.join(ballot.ranking)}"
        )

    if self.election:
      lines.append(f"\n{self.election.summary()}")

    if self.deliberation:
      lines.append(f"\n{self.deliberation.summary()}")

    if self.comparative_rankings:
      lines.append(
          f"\n--- Comparative Rankings "
          f"({len(self.comparative_rankings)} voters) ---"
      )
      for uid in sorted(self.comparative_rankings):
        lines.append(
            f"  {uid}: "
            f"{' > '.join(self.comparative_rankings[uid])}"
        )
      if self.rankings_file:
        lines.append(f"\n  Results saved to: {self.rankings_file}")

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
  region_file = os.path.join(cache_path, "region.json") if cache_path else None

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
    question_override: Optional[str] = None,
    seed: int = 42,
    is_api: bool = False,
    backend: str = "transformers",
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
    region_description: Optional[str] = None,
    num_districts: int = 5,
    region_cache_path: Optional[str] = None,
    parties: Optional[List[str]] = None,
    voting_system: str = "sntv",
    max_rank: int = 3,
    deliberation_enabled: bool = True,
    deliberation_max_rounds: int = 3,
    voting_systems: Optional[List[str]] = None,
    results_dir: Optional[str] = None,
    personas_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    voting_mode: Optional[str] = None,
    max_seats_per_district: Optional[int] = None,
    personas_tag: Optional[str] = None,
) -> VotingRoundResult:
  """Execute a full voting round with election.

  Pipeline phases:
    1. Region generation (optional)
    2. Sample political question
    3. Voter opinion survey (concurrent)
    4. Party policy responses (concurrent)
    5. Voter ranking of parties (concurrent)
    6. Election — seat allocation via the configured voting system
    7. Parliamentary deliberation — bill drafting, amendment, and
       vote (optional, enabled by default)
    8. Comparative policy ranking — run all configured voting
       systems, deliberate under each, and ask voters to rank
       the resulting policies (optional, requires
       ``voting_systems``).

  Args:
    num_voters: Number of voters to sample from PRISM.
    model_path: Path or name of the LLM to use.
    prism_dataset_dir: Path to the PRISM dataset directory.
    political_questions_path: Path to the political-questions CSV.
    party_dir: Directory containing party platform JSON files.
    topic: Topic category to filter questions by.
    question_override: If provided, use this exact question text
      instead of sampling from the dataset.
    seed: Random seed for reproducibility.
    is_api: Whether the model is accessed via API.
    backend: LLM backend name.
    temperature: Sampling temperature for LLM generation.
    max_workers: Max concurrent LLM calls per stage.
    region_description: If provided, a region is loaded/generated
      and voters are assigned districts.
    num_districts: Number of districts in the generated region.
    region_cache_path: Directory to cache the generated region JSON.
    parties: Optional list of ideology names to include.
    voting_system: Primary electoral system for phases 6-7.
    max_rank: Maximum number of parties each voter ranks.
    deliberation_enabled: Whether to run the deliberation phase.
    deliberation_max_rounds: Max bill consideration attempts.
    voting_systems: List of voting system names for comparative
      ranking (phase 8).  If ``None`` or empty, phase 8 is
      skipped.
    results_dir: Directory to persist per-model ranking JSON.
      If ``None``, rankings are computed but not saved.

  Returns:
    A ``VotingRoundResult`` with all phases' outputs.
  """
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
  if question_override:
    question = question_override
  else:
    pq_sampler = PoliticalQuestionSampler(political_questions_path)
    question = pq_sampler.sample_question_text(topic=topic, seed=seed)
  logging.info("Sampled question [%s]: %s", topic, question)

  # -- 3. Voter survey (concurrent) ----------------------------------------
  logging.info("Loading model '%s' (backend=%s)...", model_path, backend)
  model = get_model(model_path, is_api=is_api, seed=seed, backend_name=backend)

  prism = PrismSampler(prism_dataset_dir)
  voters = load_or_create_personas(
      prism_dataset_dir=prism_dataset_dir,
      personas_path=personas_path,
      num_personas=num_voters,
      seed=seed,
  ) if personas_path else prism.sample(num_samples=num_voters, seed=seed)
  voter_responses = generate_voter_responses(
      k=num_voters,
      question=question,
      prism_sampler=prism,
      model=model,
      seed=seed,
      temperature=temperature,
      max_workers=max_workers,
      region=region,
      voters_override=voters,
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

  # -- 5. Voter ranking (concurrent) --------------------------------------
  ballots = generate_voter_rankings(
      voter_responses=voter_responses,
      party_responses=party_responses,
      model=model,
      temperature=temperature,
      max_rank=max_rank,
      max_workers=max_workers,
      region=region,
  )

  # -- 6. Election — seat allocation --------------------------------------
  election: Optional[ElectionResult] = None
  if region and ballots:
    district_seats = _read_district_seats(region.districts, max_seats_per_district)
    logging.info("District seat allocation: %s", district_seats)
    election = run_election(
        ballots=ballots,
        district_seats=district_seats,
        system=voting_system,
    )

  # -- 7. Deliberation — parliamentary bill process ------------------------
  deliberation: Optional[DeliberationResult] = None
  if deliberation_enabled and election and election.total_seats:
    logging.info("Starting parliamentary deliberation...")
    deliberation, _ = run_deliberation(
        model=model,
        issue=question,
        seat_allocation=election.total_seats,
        party_policies=party_responses,
        temperature=temperature,
        max_rounds=deliberation_max_rounds,
        max_workers=max_workers,
        voter_responses=voter_responses,
        districts=region.districts if region else None,
    )

  # -- 8. Comparative policy ranking ------------------------------------
  comparative_rankings: Optional[Dict[str, List[str]]] = None
  rankings_file: Optional[str] = None
  if voting_systems and region and ballots:
    district_seats = _read_district_seats(region.districts, max_seats_per_district)
    logging.info(
        "Starting comparative policy ranking across %d systems: %s",
        len(voting_systems),
        voting_systems,
    )

    comp_results, government_infos = run_comparative_elections(
        ballots=ballots,
        district_seats=district_seats,
        party_policies=party_responses,
        model=model,
        voting_systems=voting_systems,
        issue=question,
        temperature=temperature,
        deliberation_max_rounds=deliberation_max_rounds,
        max_workers=max_workers,
        voter_responses=voter_responses,
        districts=region.districts if region else None,
    )

    comparative_rankings, comparative_scores = rank_policies(
        voter_responses=voter_responses,
        comparative_results=comp_results,
        model=model,
        temperature=temperature,
        max_workers=max_workers,
    )

    if results_dir:
      # Build system → adopted policy text mapping for persistence.
      sys_policies = {
          s: _bill_to_text(delib)
          for s, (_, delib) in comp_results.items()
      }
      delib_rounds = {
          s: len(delib.rounds)
          for s, (_, delib) in comp_results.items()
      }
      rankings_file = save_rankings(
          rankings=comparative_rankings,
          issue=question,
          model_path=model_path,
          output_dir=results_dir,
          system_policies=sys_policies,
          party_responses=party_responses,
          scores=comparative_scores,
          ballots=ballots,
          deliberation_rounds=delib_rounds,
          dataset_name=dataset_name,
          voting_mode=voting_mode,
          voter_opinions={
              vr.user_id: vr.response for vr in voter_responses
          },
          government_infos=government_infos,
          personas_tag=personas_tag,
      )

  result = VotingRoundResult(
      question=question,
      voter_responses=voter_responses,
      party_responses=party_responses,
      ballots=ballots,
      election=election,
      region=region,
      deliberation=deliberation,
      comparative_rankings=comparative_rankings,
      rankings_file=rankings_file,
  )
  logging.info("Voting round complete.")
  return result



# ---------------------------------------------------------------------------
# Survey-only mode
# ---------------------------------------------------------------------------


def run_survey_only(
    num_voters: int,
    model_path: str,
    questions: List[str],
    prism_dataset_dir: str = "dataset/prism",
    party_dir: str = "dataset/party",
    seed: int = 42,
    is_api: bool = False,
    backend: str = "transformers",
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
    region_description: Optional[str] = None,
    num_districts: int = 5,
    region_cache_path: Optional[str] = None,
    personas_path: Optional[str] = None,
    parties: Optional[List[str]] = None,
    max_rank: int = 3,
    results_dir: Optional[str] = None,
    dataset_name: Optional[str] = None,
    max_seats_per_district: Optional[int] = None,
    personas_tag: Optional[str] = None,
) -> List[VotingRoundResult]:
  """Run the pipeline through the voter ranking phase (phases 1-5).

  This mode runs voter survey, party responses, and voter ranking
  but skips election, deliberation, and comparative policy ranking.
  It is useful for collecting voter opinion distributions and ballot
  data without the cost of the full pipeline.

  Flow:
    **Once (before the question loop):**
      1. Generate or load region.
      2. Load model, sample voters, assign districts.

    **Per question:**
      3. Voter opinion survey.
      4. Party policy responses.
      5. Voter ranking of parties → ballots.

  All issues are written to a single consolidated JSON file when
  ``results_dir`` is provided.

  Args:
    num_voters: Number of voters to sample from PRISM.
    model_path: Path or name of the LLM to use.
    questions: List of social-issue question texts to process.
    prism_dataset_dir: Path to the PRISM dataset directory.
    party_dir: Directory containing party platform JSON files.
    seed: Random seed for reproducibility.
    is_api: Whether the model is accessed via API.
    backend: LLM backend name.
    temperature: Sampling temperature.
    max_workers: Max concurrent LLM calls per stage.
    region_description: If provided, a region is generated.
    num_districts: Number of districts in the region.
    region_cache_path: Directory to cache the region JSON.
    personas_path: Path to a pre-built personas JSON file.
    parties: Optional list of ideology names to include.
    max_rank: Number of parties each voter ranks.
    results_dir: Directory to persist the consolidated JSON.
    dataset_name: Optional dataset label for output metadata.

  Returns:
    A list of ``VotingRoundResult`` objects, one per question.
  """
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

  # -- 2. Load model, sample voters, assign districts ---------------------
  logging.info(
      "Loading model '%s' (backend=%s)...", model_path, backend
  )
  model = get_model(
      model_path, is_api=is_api, seed=seed, backend_name=backend
  )

  prism = PrismSampler(prism_dataset_dir)
  personas = load_or_create_personas(
      prism_dataset_dir=prism_dataset_dir,
      personas_path=personas_path,
      num_personas=num_voters,
      seed=seed,
  ) if personas_path else prism.sample(num_samples=num_voters, seed=seed)
  voters, voter_districts = prepare_voters(
      num_voters=num_voters,
      prism_sampler=prism,
      seed=seed,
      region=region,
      voters_override=personas,
  )

  # Consolidated output: all issues go into one dict, written at the end.
  all_issues: Dict[str, Any] = {}

  # -- Per-question loop ---------------------------------------------------
  results: List[VotingRoundResult] = []

  for qi, question in enumerate(questions):
    logging.info(
        "=== [Survey mode] Question %d/%d: %s ===",
        qi + 1,
        len(questions),
        question,
    )

    # 3. Voter opinion survey.
    voter_responses = survey_voters(
        voters=voters,
        voter_districts=voter_districts,
        question=question,
        model=model,
        temperature=temperature,
        max_workers=max_workers,
        region=region,
    )

    # 4. Party policy responses.
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

    # 5. Voter ranking → ballots.
    ballots = generate_voter_rankings(
        voter_responses=voter_responses,
        party_responses=party_responses,
        model=model,
        temperature=temperature,
        max_rank=max_rank,
        max_workers=max_workers,
        region=region,
    )

    # Accumulate issue data for the consolidated output file.
    all_issues[question] = {
        "voter_opinions": {
            vr.user_id: vr.response for vr in voter_responses
        },
        "party_responses": {
            ideology: {
                "party_name": pr.party_name,
                "position_statement": pr.position_statement,
                "key_proposals": pr.key_proposals,
                "voter_alignment_score": pr.voter_alignment_score,
                "reasoning": pr.reasoning,
            }
            for ideology, pr in party_responses.items()
        },
        "ballots": {
            b.user_id: {
                "ranking": b.ranking,
                "district": b.district_name,
            }
            for b in ballots
        },
    }

    result = VotingRoundResult(
        question=question,
        voter_responses=voter_responses,
        party_responses=party_responses,
        ballots=ballots,
        region=region,
    )
    results.append(result)
    logging.info(
        "Question %d complete: %d voter responses, %d ballots.",
        qi + 1,
        len(voter_responses),
        len(ballots),
    )

  # -- Write consolidated output file ------------------------------------
  if results_dir:
    os.makedirs(results_dir, exist_ok=True)
    model_slug = os.path.basename(model_path).replace("/", "_")
    _parts = ["survey"]
    if dataset_name:
      _parts.append(dataset_name)
    if personas_tag:
      _parts.append(personas_tag)
    _parts.append(model_slug)
    out_path = os.path.join(results_dir, ".".join(_parts) + ".json")

    output = {
        "model": model_path,
        "dataset": dataset_name,
        "num_voters": num_voters,
        "num_questions": len(questions),
        "issues": all_issues,
    }
    with open(out_path, "w") as f:
      json.dump(output, f, indent=2)
    logging.info(
        "Consolidated survey results (%d issues) saved to %s",
        len(all_issues),
        out_path,
    )

  logging.info(
      "Survey mode finished: processed %d question(s).",
      len(results),
  )
  return results


# ---------------------------------------------------------------------------
# Platform-only voting mode
# ---------------------------------------------------------------------------


def _load_party_platforms(
    party_dir: str,
    parties: Optional[List[str]] = None,
) -> List[PartyPlatform]:
  """Load party platforms from JSON files in a directory.

  Args:
    party_dir: Directory containing party platform JSON files.
    parties: Optional list of ideology names to include.  If
      ``None``, all platforms are loaded.

  Returns:
    A list of ``PartyPlatform`` objects.

  Raises:
    FileNotFoundError: If no valid platforms are found.
  """
  party_filter = {p.lower() for p in parties} if parties else None
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
        f"No valid party platform JSON files in {party_dir}"
    )
  return platforms


def run_platform_mode(
    num_voters: int,
    model_path: str,
    questions: List[str],
    prism_dataset_dir: str = "dataset/prism",
    party_dir: str = "dataset/party",
    seed: int = 42,
    is_api: bool = False,
    backend: str = "transformers",
    temperature: float = 0.7,
    max_workers: Optional[int] = None,
    region_description: Optional[str] = None,
    num_districts: int = 5,
    region_cache_path: Optional[str] = None,
    parties: Optional[List[str]] = None,
    max_rank: int = 3,
    deliberation_enabled: bool = True,
    deliberation_max_rounds: int = 3,
    voting_systems: Optional[List[str]] = None,
    results_dir: Optional[str] = None,
    personas_path: Optional[str] = None,
    dataset_name: Optional[str] = None,
    voting_mode: Optional[str] = None,
    max_seats_per_district: Optional[int] = None,
    personas_tag: Optional[str] = None,
) -> List[VotingRoundResult]:
  """Run the pipeline in platform-only voting mode.

  In this mode voters rank parties **once** based on their general
  platforms (no issue-specific policy generation), producing a
  single set of ballots and a fixed seat allocation per voting
  system.  Deliberation and comparative policy ranking then run
  per-issue using those fixed seats and platform-derived policies.

  Flow:
    **Once (before the question loop):**
      1. Generate or load region.
      2. Load model, sample voters, assign districts.
      3. Load party platforms.
      4. Platform-based ranking → ballots.

    **Per question:**
      5. Voter opinion survey (for use in bill ranking).
      6. Comparative elections — reuses fixed ballots, runs
         deliberation per voting system, plus baselines.
      7. Voters rank the resulting bills.
      8. Persist results.

  Args:
    num_voters: Number of voters to sample from PRISM.
    model_path: Path or name of the LLM to use.
    questions: List of social-issue question texts to process.
    prism_dataset_dir: Path to the PRISM dataset directory.
    party_dir: Directory containing party platform JSON files.
    seed: Random seed for reproducibility.
    is_api: Whether the model is accessed via API.
    backend: LLM backend name.
    temperature: Sampling temperature.
    max_workers: Max concurrent LLM calls per stage.
    region_description: If provided, a region is generated.
    num_districts: Number of districts in the region.
    region_cache_path: Directory to cache the region JSON.
    parties: Optional list of ideology names to include.
    max_rank: Number of parties each voter ranks.
    deliberation_enabled: Whether to run deliberation.
    deliberation_max_rounds: Max bill consideration attempts.
    voting_systems: List of voting system names for comparative
      ranking.  If ``None`` or empty, only single-system
      results are produced.
    results_dir: Directory to persist per-model ranking JSON.

  Returns:
    A list of ``VotingRoundResult`` objects, one per question.
  """
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

  # -- 2. Load model, sample voters, assign districts ---------------------
  logging.info(
      "Loading model '%s' (backend=%s)...", model_path, backend
  )
  model = get_model(
      model_path, is_api=is_api, seed=seed, backend_name=backend
  )

  prism = PrismSampler(prism_dataset_dir)
  personas = load_or_create_personas(
      prism_dataset_dir=prism_dataset_dir,
      personas_path=personas_path,
      num_personas=num_voters,
      seed=seed,
  ) if personas_path else prism.sample(num_samples=num_voters, seed=seed)
  voters, voter_districts = prepare_voters(
      num_voters=num_voters,
      prism_sampler=prism,
      seed=seed,
      region=region,
      voters_override=personas,
  )

  # -- 3. Load party platforms --------------------------------------------
  platforms = _load_party_platforms(party_dir, parties)
  logging.info(
      "Loaded %d party platforms: %s",
      len(platforms),
      [p.ideology for p in platforms],
  )

  # Build platform-derived PolicyResponse dict for deliberation.
  platform_policies: Dict[str, PolicyResponse] = {
      p.ideology: platform_to_policy_response(p)
      for p in platforms
  }

  # -- 4. Platform-based ranking → ballots (ONCE) -------------------------
  ballots = generate_platform_rankings(
      voters=voters,
      voter_districts=voter_districts,
      platforms=platforms,
      model=model,
      temperature=temperature,
      max_rank=max_rank,
      max_workers=max_workers,
      region=region,
  )
  logging.info(
      "Platform rankings complete: %d ballots.", len(ballots)
  )

  # Pre-compute district seats for elections.
  district_seats: Optional[Dict[str, int]] = None
  if region:
    district_seats = _read_district_seats(region.districts, max_seats_per_district)
    logging.info("District seat allocation: %s", district_seats)

  # -- Per-question loop ---------------------------------------------------
  results: List[VotingRoundResult] = []

  for qi, question in enumerate(questions):
    logging.info(
        "=== [Platform mode] Question %d/%d: %s ===",
        qi + 1,
        len(questions),
        question,
    )

    # 5. Voter opinion survey (for bill ranking later).
    voter_responses = survey_voters(
        voters=voters,
        voter_districts=voter_districts,
        question=question,
        model=model,
        temperature=temperature,
        max_workers=max_workers,
        region=region,
    )

    # 6-7. Comparative elections + deliberation + bill ranking.
    comparative_rankings: Optional[Dict[str, List[str]]] = None
    rankings_file: Optional[str] = None

    if voting_systems and district_seats and ballots:
      logging.info(
          "Comparative ranking across %d systems: %s",
          len(voting_systems),
          voting_systems,
      )

      effective_max_rounds = (
          deliberation_max_rounds if deliberation_enabled else 0
      )

      comp_results, government_infos = run_comparative_elections(
          ballots=ballots,
          district_seats=district_seats,
          party_policies=platform_policies,
          model=model,
          voting_systems=voting_systems,
          issue=question,
          temperature=temperature,
          deliberation_max_rounds=effective_max_rounds,
          max_workers=max_workers,
          voter_responses=voter_responses,
          districts=region.districts if region else None,
      )

      comparative_rankings, comparative_scores = rank_policies(
          voter_responses=voter_responses,
          comparative_results=comp_results,
          model=model,
          temperature=temperature,
          max_workers=max_workers,
      )

      if results_dir:
        sys_policies = {
            s: _bill_to_text(delib)
            for s, (_, delib) in comp_results.items()
        }
        delib_rounds = {
            s: len(delib.rounds)
            for s, (_, delib) in comp_results.items()
        }
        rankings_file = save_rankings(
            rankings=comparative_rankings,
            issue=question,
            model_path=model_path,
            output_dir=results_dir,
            system_policies=sys_policies,
            party_responses=platform_policies,
            scores=comparative_scores,
            ballots=ballots,
            deliberation_rounds=delib_rounds,
            dataset_name=dataset_name,
            voting_mode=voting_mode,
            voter_opinions={
                vr.user_id: vr.response for vr in voter_responses
            },
            government_infos=government_infos,
            personas_tag=personas_tag,
        )

    result = VotingRoundResult(
        question=question,
        voter_responses=voter_responses,
        party_responses=platform_policies,
        ballots=ballots,
        election=None,
        region=region,
        deliberation=None,
        comparative_rankings=comparative_rankings,
        rankings_file=rankings_file,
    )
    results.append(result)
    logging.info(
        "Question %d complete: %d voter responses, "
        "%d systems ranked.",
        qi + 1,
        len(voter_responses),
        len(comparative_rankings) if comparative_rankings else 0,
    )

  logging.info(
      "Platform mode finished: processed %d question(s).",
      len(results),
  )
  return results
