"""Functions to run personalized queries using LLMs."""

import hashlib
import logging
import os
from typing import Optional

from omegaconf import DictConfig
from pathfinder import assistant, gen, get_model, system, user
from simulation.district_generator import Region, RegionGenerator
from simulation.pipeline import run_platform_mode
from simulation.pipeline import run_survey_only
from simulation.pipeline import run_voting_round
from simulation.political_sampler import PoliticalQuestionSampler
from simulation.prism_sampler import PrismSampler

DEFAULT_NUM_DISTRICTS = 3
DEFAULT_NUM_VOTERS = 10
DEFAULT_PRISM_DATASET_DIR = "dataset/prism"
DEFAULT_POLITICAL_QUESTIONS_PATH = (
    "dataset/political_questions/political-questions.csv"
)
DEFAULT_PARTY_DIR = "dataset/party"
DEFAULT_PERSONAS_PATH = "dataset/personas/personas.json"
DEFAULT_SEED = 42
DEFAULT_TEMPERATURE = 0.0
DEFAULT_TOP_P = 1.0

# Maps short dataset names to CSV paths.
DATASET_MAP = {
    "diverse-12": "dataset/political_questions/diverse-12.csv",
    "diverse-20": "dataset/political_questions/diverse-20.csv",
    "divisive-12": "dataset/political_questions/divisive-12.csv",
    "harm-12": "dataset/political_questions/harm-12.csv",
    "harm-20": "dataset/political_questions/harm-20.csv",
    "legacy": DEFAULT_POLITICAL_QUESTIONS_PATH,
}


def run_query(cfg: DictConfig):
  """Runs a single personalized query using the PRISM dataset and specified LLM."""

  logging.info("Starting highly personalized query.")

  dataset_dir = cfg.get(
      "dataset_dir",
      DEFAULT_PRISM_DATASET_DIR,
  )
  sampler = PrismSampler(dataset_dir)

  samples = sampler.sample(num_samples=1, seed=cfg.get("seed", 42))
  if not samples:
    logging.error("No samples found in dataset.")
    return

  sample = samples[0]
  demo = sample["demographics"]
  examples = sample["examples"]

  logging.info("Sampled User ID: %s", sample["user_id"])
  logging.info("Sampled Demographics: %s", demo)
  logging.info("Sampled Examples: %s", examples)

  is_api = cfg.llm.get("is_api", False)
  seed = cfg.get("seed", DEFAULT_SEED)
  backend = cfg.llm.get("backend", "transformers")

  # Load or generate region
  region_cfg = cfg.get("region")
  region = None
  district = None

  if region_cfg:
    cache_path = region_cfg.get("cache_path")
    region_file = (
        os.path.join(cache_path, "region.json") if cache_path else None
    )

    if region_file and os.path.exists(region_file):
      logging.info("Loading cached region from %s", region_file)
      region = Region.from_json(region_file)
    else:
      logging.info("Generating new region...")
      # We need a model to generate the region.
      # We can use the same model as for the query.
      generator = RegionGenerator(
          cfg.llm.path, is_api=is_api, seed=seed, backend=backend
      )
      region = generator.generate(
          region_cfg.description, region_cfg.num_districts
      )
      if region_file:
        region.to_json(region_file)

    # Assign a district to the user deterministically based on user_id
    if region:
      user_id_hash = hashlib.md5(sample["user_id"].encode()).digest()
      user_seed = int.from_bytes(user_id_hash, "big") % 10000
      sampled_districts = region.sample(1, seed=user_seed)
      if sampled_districts:
        district = sampled_districts[0]
        logging.info("Assigned District: %s", district["name"])

  model = get_model(
      cfg.llm.path, is_api=is_api, seed=seed, backend_name=backend
  )

  lm = model

  # Personalize using system prompt
  persona_desc = (
      f"You are a person with the following demographics: Age: {demo['age']},"
      f" Gender: {demo['gender']}, Employment: {demo['employment']}, Education:"
      f" {demo['education']}."
  )

  region_desc = ""
  if region and district:
    region_desc = (
        f"\nYou live in the region of '{region.region_name}' (inspired by"
        f" {region.description}).\nSpecifically, you live in the district of"
        f" '{district['name']}'.\nDistrict Characteristics:\n- Wealth:"
        f" {district['wealth']} (0-1 scale)\n- Urbanisation:"
        f" {district['urbanisation']} (0-1 scale)\n- Primary Industry:"
        f" {district['industry']}\n- Political Leaning:"
        f" {district.get('political_leaning', 'unknown')}\n- Description:"
        f" {district['description']}\n"
    )

  examples_str = "\n".join([f"- {ex}" for ex in examples])
  full_system_prompt = (
      f"{persona_desc}{region_desc}\nHere are some examples of things you have"
      f" said in the past:\n{examples_str}\nAlways respond in character and do"
      " not explicitly reference the fact that you are being personalized in"
      " your response. Do not say 'As an AI' or similar."
  )

  # Determine the prompt: either sample a political question or use the static
  # prompt from config.
  pq_cfg = cfg.get("political_questions")
  if pq_cfg:
    pq_path = pq_cfg.get(
        "path", "dataset/political_questions/political-questions.csv"
    )
    pq_topic = pq_cfg.get("topic", "social")
    pq_sampler = PoliticalQuestionSampler(pq_path)
    prompt_text = pq_sampler.sample_question_text(topic=pq_topic, seed=seed)
    logging.info("Sampled political question [%s]: %s", pq_topic, prompt_text)
  else:
    prompt_text = cfg.prompt

  with system():
    lm += full_system_prompt
  with user():
    lm += prompt_text
  with assistant():
    lm += gen(max_tokens=2048, name="response")

  response = lm["response"]

  logging.info("Prompt: %s", prompt_text)
  logging.info("Model Response:\n%s", response)


def run_pipeline(cfg: DictConfig):
  """Run a full voting-round pipeline: voters respond → parties respond.

  Reads configuration from ``cfg`` and delegates to
  ``pipeline.run_voting_round``.  Logs the complete summary of voter
  responses and party policy proposals.

  Expected config keys (all have sensible defaults):
    - ``seed``: int — master random seed.
    - ``llm.path``: str — model path / name.
    - ``llm.is_api``: bool — whether model is API-based.
    - ``llm.backend``: str — LLM backend name.
    - ``pipeline.k``: int — number of voters to sample.
    - ``pipeline.temperature``: float — LLM temperature.
    - ``pipeline.max_workers``: int — concurrency limit.
    - ``pipeline.topic``: str — question topic filter.
    - ``pipeline.voting_systems``: list[str] — systems for
      comparative policy ranking (phase 8).
    - ``pipeline.results_dir``: str — output dir for per-model
      ranking JSON files.
    - ``dataset_dir``: str — PRISM dataset path.
    - ``political_questions.dataset``: str — dataset name
      (``diverse-12``, ``diverse-20``, ``harm-12``, ``harm-20``,
      or ``legacy``).
    - ``political_questions.question_indices``: list[int] — optional
      0-based indices to select from the dataset. Omit for all.
    - ``party_dir``: str — directory of party platform JSONs.
    - ``region.description``: str — region description.
    - ``region.num_districts``: int — number of districts.
    - ``region.cache_path``: str — where to cache region JSON.

  The pipeline iterates over every question in the dataset (or the
  specified subset), running a full voting round for each.  Results
  accumulate in the per-model JSON file under separate issue keys.

  Args:
    cfg: The configuration dictionary, typically loaded from ``config.yaml``.

  Returns:
    The ``VotingRoundResult`` for the last question processed, or ``None``
    if no questions were available.
  """
  logging.info("Starting voting-round pipeline.")

  seed = cfg.get("seed", DEFAULT_SEED)
  is_api = cfg.llm.get("is_api", False)
  backend = cfg.llm.get("backend", "transformers")
  model_path = cfg.llm.path
  debug = cfg.get("debug", False)

  # Pipeline-specific settings.
  pipe_cfg = cfg.get("pipeline", {})
  num_voters = pipe_cfg.get("num_voters", DEFAULT_NUM_VOTERS)
  temperature = cfg.llm.get("temperature", DEFAULT_TEMPERATURE)
  max_workers = pipe_cfg.get("max_workers", None)
  topic = pipe_cfg.get("topic", "social")
  parties = pipe_cfg.get("parties", None)
  if parties is not None:
    parties = list(parties)
  voting_system = pipe_cfg.get("voting_system", "sntv")
  max_rank = pipe_cfg.get("max_rank", 3)

  # Comparative ranking settings.
  voting_systems = pipe_cfg.get("voting_systems", None)
  if voting_systems is not None:
    voting_systems = list(voting_systems)
  results_dir = pipe_cfg.get("results_dir", None)

  # Deliberation settings.
  delib_cfg = pipe_cfg.get("deliberation", {})
  deliberation_enabled = delib_cfg.get("enabled", True)
  deliberation_max_rounds = delib_cfg.get("max_rounds", 3)

  # --- Debug overrides ---------------------------------------------------
  if debug:
    num_voters = min(num_voters, 10)
    if parties and len(parties) > 2:
      parties = parties[:2]
    if voting_systems and len(voting_systems) > 2:
      voting_systems = voting_systems[:2]
    max_workers = min(max_workers or 2, 2)
    logging.info(
        "DEBUG overrides: num_voters=%d, parties=%s, "
        "voting_systems=%s, max_workers=%d",
        num_voters,
        parties,
        voting_systems,
        max_workers,
    )

  # Dataset paths.
  prism_dir = cfg.get("dataset_dir", DEFAULT_PRISM_DATASET_DIR)
  pq_cfg = cfg.get("political_questions", {})
  # Resolve dataset name → CSV path (fall back to explicit "path" key).
  pq_dataset = pq_cfg.get("dataset", "")
  if pq_dataset in DATASET_MAP:
    pq_path = DATASET_MAP[pq_dataset]
  else:
    pq_path = pq_cfg.get("path", DEFAULT_POLITICAL_QUESTIONS_PATH)
  question_indices = pq_cfg.get("question_indices", None)
  party_dir = cfg.get("party_dir", DEFAULT_PARTY_DIR)
  personas_path = pipe_cfg.get("personas_path", DEFAULT_PERSONAS_PATH)

  # Derive a short personas tag from the file stem so that runs using
  # different persona files write to distinct output filenames.
  # e.g. "personas_635.json" → "p635", "personas_800.json" → "p800".
  # The default "personas.json" produces no tag (None) so existing
  # filenames are unchanged.
  _personas_stem = os.path.splitext(os.path.basename(personas_path))[0]
  if _personas_stem == "personas":
    personas_tag: Optional[str] = None
  else:
    # Strip the leading "personas" / "personas_" prefix if present.
    _suffix = _personas_stem.removeprefix("personas_").removeprefix("personas")
    personas_tag = f"p{_suffix}" if _suffix else _personas_stem

  # Region settings (optional).
  region_cfg = cfg.get("region")
  region_description = None
  num_districts = 5
  region_cache_path = None
  if region_cfg:
    region_description = region_cfg.get("description")
    num_districts = region_cfg.get("num_districts", 5)
    region_cache_path = region_cfg.get("cache_path")

  max_seats_per_district = None  # no cap by default
  if debug:
    max_seats_per_district = 1
    logging.info("DEBUG override: max_seats_per_district=%d", max_seats_per_district)

  # Load questions from the dataset.
  pq_sampler = PoliticalQuestionSampler(pq_path)
  all_questions = pq_sampler.questions  # list of dicts with 'question' key
  if question_indices is not None:
    selected = []
    for idx in question_indices:
      if 0 <= idx < len(all_questions):
        selected.append(all_questions[idx])
      else:
        logging.warning(
            "question_indices contains out-of-range index %d "
            "(dataset has %d questions). Skipping.",
            idx,
            len(all_questions),
        )
    all_questions = selected

  # In debug mode, limit to first question only.
  if debug and len(all_questions) > 1:
    logging.info("DEBUG: limiting to first question only.")
    all_questions = all_questions[:1]

  logging.info(
      "Running pipeline over %d question(s) from '%s'.",
      len(all_questions),
      pq_path,
  )

  # --- Mode dispatch --------------------------------------------------
  voting_mode = pipe_cfg.get("voting_mode", "issue")

  if voting_mode == "survey":
    logging.info("Using survey mode (phases 1-5: through voter ranking).")
    question_texts = [q["question"] for q in all_questions]
    survey_results = run_survey_only(
        num_voters=num_voters,
        model_path=model_path,
        questions=question_texts,
        prism_dataset_dir=prism_dir,
        party_dir=party_dir,
        seed=seed,
        is_api=is_api,
        backend=backend,
        temperature=temperature,
        max_workers=max_workers,
        region_description=region_description,
        num_districts=num_districts,
        region_cache_path=region_cache_path,
        personas_path=personas_path,
        parties=parties,
        max_rank=max_rank,
        results_dir=results_dir,
        dataset_name=pq_dataset or None,
        max_seats_per_district=max_seats_per_district,
        personas_tag=personas_tag,
    )
    logging.info(
        "Pipeline finished (survey mode): %d question(s).",
        len(survey_results),
    )
    return survey_results[-1] if survey_results else None

  if voting_mode == "platform":
    logging.info("Using platform-only voting mode.")
    question_texts = [q["question"] for q in all_questions]
    platform_results = run_platform_mode(
        num_voters=num_voters,
        model_path=model_path,
        questions=question_texts,
        prism_dataset_dir=prism_dir,
        party_dir=party_dir,
        seed=seed,
        is_api=is_api,
        backend=backend,
        temperature=temperature,
        max_workers=max_workers,
        region_description=region_description,
        num_districts=num_districts,
        region_cache_path=region_cache_path,
        parties=parties,
        max_rank=max_rank,
        deliberation_enabled=deliberation_enabled,
        deliberation_max_rounds=deliberation_max_rounds,
        voting_systems=voting_systems,
        results_dir=results_dir,
        personas_path=personas_path,
        dataset_name=pq_dataset or None,
        voting_mode=voting_mode,
        max_seats_per_district=max_seats_per_district,
        personas_tag=personas_tag,
    )
    logging.info(
        "Pipeline finished (platform mode): %d question(s).",
        len(platform_results),
    )
    return platform_results[-1] if platform_results else None

  # --- Default "issue" mode (unchanged) --------------------------------

  results = []
  for qi, q_entry in enumerate(all_questions):
    question_text = q_entry["question"]
    logging.info(
        "=== Question %d/%d: %s ===",
        qi + 1,
        len(all_questions),
        question_text,
    )
    result = run_voting_round(
        num_voters=num_voters,
        model_path=model_path,
        prism_dataset_dir=prism_dir,
        political_questions_path=pq_path,
        party_dir=party_dir,
        topic=topic,
        question_override=question_text,
        seed=seed + qi,  # vary seed per question for voter diversity
        is_api=is_api,
        backend=backend,
        temperature=temperature,
        max_workers=max_workers,
        region_description=region_description,
        num_districts=num_districts,
        region_cache_path=region_cache_path,
        parties=parties,
        voting_system=voting_system,
        max_rank=max_rank,
        deliberation_enabled=deliberation_enabled,
        deliberation_max_rounds=deliberation_max_rounds,
        voting_systems=voting_systems,
        results_dir=results_dir,
        personas_path=personas_path,
        dataset_name=pq_dataset or None,
        voting_mode=voting_mode,
        max_seats_per_district=max_seats_per_district,
        personas_tag=personas_tag,
    )
    results.append(result)

    # Report results for this question.
    logging.info(result.summary())
    governing = (
        result.election.governing_party if result.election else "N/A"
    )
    logging.info(
        "Question %d complete: %d voters, %d parties, %d ballots. "
        "Governing party: %s",
        qi + 1,
        len(result.voter_responses),
        len(result.party_responses),
        len(result.ballots),
        governing,
    )

  logging.info(
      "Pipeline finished: processed %d question(s).", len(results)
  )
  return results[-1] if results else None

