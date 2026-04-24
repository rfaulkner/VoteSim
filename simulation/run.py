"""Functions to run personalized queries using LLMs."""

import hashlib
import logging
import os

from omegaconf import DictConfig
from pathfinder import assistant, gen, get_model, system, user
from simulation.district_generator import Region, RegionGenerator
from simulation.prism_sampler import PrismSampler


def run_query(cfg: DictConfig):
  """Runs a single personalized query using the PRISM dataset and specified LLM."""

  logging.info("Starting highly personalized query.")

  dataset_dir = cfg.get(
      "dataset_dir",
      "dataset/prism",
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
  seed = cfg.get("seed", 42)
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

  with system():
    lm += full_system_prompt
  with user():
    lm += cfg.prompt
  with assistant():
    lm += gen(max_tokens=2048, name="response")

  response = lm["response"]

  logging.info("Model Response:\n%s", response)

