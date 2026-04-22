"""Functions to run personalized queries using LLMs."""

import logging
import os
from omegaconf import DictConfig
from pathfinder import assistant, gen, get_model, system, user
from simulation.prism_sampler import PrismSampler


def run_query(cfg: DictConfig):
  """Runs a personalized query using LLMs and the PRISM dataset for persona generation."""

  logging.info("Starting personalized query.")

  dataset_dir = cfg.get(
      "dataset_dir",
      "/google/src/cloud/rfaulk/votesim/google3/experimental/users/rfaulk/VoteSim/dataset/prism",
  )
  sampler = PrismSampler(dataset_dir)

  # Sample one persona
  samples = sampler.sample(num_samples=1, seed=cfg.get("seed", 42))
  if not samples:
    logging.error("No samples found in dataset.")
    return

  sample = samples[0]
  demo = sample["demographics"]

  logging.info("Sampled Demographics: %s", demo)

  is_api = cfg.llm.get("is_api", False)
  seed = cfg.get("seed", 42)
  backend = cfg.llm.get("backend", "transformers")

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

  with system():
    lm += persona_desc
  with user():
    lm += cfg.prompt
  with assistant():
    lm += gen(max_tokens=100, name="response")

  response = lm["response"]

  logging.info("Model Response:\n%s", response)
