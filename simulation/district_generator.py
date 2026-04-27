"""Generator and sampler for regions composed of districts with varied characteristics."""

import json
import logging
import os
import random
import re
from typing import Any, Dict, List, Optional

from pathfinder import assistant
from pathfinder import gen
from pathfinder import get_model
from pathfinder import user

DISTRICT_PROMPT = """You are a world-building assistant. Generate a realistic region inspired by {description}.

The region has exactly {num_districts} districts. For each district, provide a JSON object with these fields:
- "name": string (a plausible district name)
- "population": int (estimated population)
- "wealth": float 0-1 (relative wealth, 1 = wealthiest in region)
- "urbanisation": float 0-1 (1 = fully urban, 0 = fully rural)
- "infrastructure": float 0-1 (quality of roads, transit, broadband)
- "employment_rate": float 0-1
- "industry": string (dominant industry)
- "education_level": float 0-1 (share of population with post-secondary education)
- "political_leaning": string (e.g., "progressive", "conservative", "moderate", "swing")
- "demographics_diversity": float 0-1 (1 = highly diverse)
- "median_age": int
- "crime_rate": float 0-1 (relative to region)
- "description": string (1-2 sentence narrative description)

The districts should be spatially and culturally connected as parts of a single governed region, but vary realistically in their characteristics. Include a mix of urban cores, suburbs, small towns, and rural areas as appropriate for the analog.

Return ONLY a JSON object with two keys:
- "region_name": string
- "districts": array of {num_districts} district objects

No other text, just valid JSON."""


class Region:
  """A generated region containing districts with varied characteristics."""

  def __init__(
      self,
      region_name: str,
      districts: List[Dict[str, Any]],
      description: str,
      seed: int,
  ):
    self.region_name = region_name
    self.districts = districts
    self.description = description
    self.seed = seed

  def sample(self, n: int, seed: int) -> List[Dict[str, Any]]:
    """Randomly sample n districts."""
    random.seed(seed)
    return random.sample(self.districts, min(n, len(self.districts)))

  def get_district(self, name: str) -> Optional[Dict[str, Any]]:
    """Lookup a district by name (case-insensitive)."""
    for d in self.districts:
      if d["name"].lower() == name.lower():
        return d
    return None

  def all_districts(self) -> List[Dict[str, Any]]:
    """Return all districts."""
    return list(self.districts)

  def to_json(self, path: str):
    """Save region to a JSON file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
      json.dump(
          {
              "region_name": self.region_name,
              "description": self.description,
              "seed": self.seed,
              "districts": self.districts,
          },
          f,
          indent=2,
      )
    logging.info("Saved region to %s", path)

  @classmethod
  def from_json(cls, path: str) -> "Region":
    """Load a region from a JSON file."""
    with open(path, "r") as f:
      data = json.load(f)
    return cls(
        region_name=data["region_name"],
        districts=data["districts"],
        description=data.get("description", ""),
        seed=data.get("seed", 0),
    )


class RegionGenerator:
  """Uses an LLM to generate a region with realistic districts."""

  def __init__(
      self,
      model_path: str,
      is_api: bool = False,
      seed: int = 42,
      backend: str = "transformers",
  ):
    self.model = get_model(
        model_path, is_api=is_api, seed=seed, backend_name=backend
    )
    self.seed = seed

  def generate(
      self, description: str, num_districts: int, temperature: float = 0.7
  ) -> Region:
    """Generate a region with the specified number of districts."""
    prompt = DISTRICT_PROMPT.format(
        description=description, num_districts=num_districts
    )

    lm = self.model
    with user():
      lm += prompt
    with assistant():
      lm += gen(max_tokens=16384, temperature=temperature, name="region_json")

    raw = lm["region_json"]
    logging.info("Raw LLM output length: %d chars", len(raw))
    logging.info("Raw LLM output (first 500 chars): %.500s", raw)

    # Strip <think>...</think> blocks (e.g. Qwen3 chain-of-thought).
    # Also handle unclosed <think> when model runs out of tokens mid-thought.
    text = re.sub(r"<think>.*?(</think>|$)", "", raw, flags=re.DOTALL).strip()

    # Strip markdown fences if present
    if text.startswith("```"):
      text = text.split("\n", 1)[1]
      text = text.rsplit("```", 1)[0]
      text = text.strip()

    if not text:
      raise ValueError(
          "LLM returned empty response. Cannot parse region JSON. "
          f"Raw output was: {raw!r:.200}"
      )

    try:
      data = json.loads(text)
    except json.JSONDecodeError as e:
      logging.error("Failed to parse LLM output as JSON. Content: %.500s", text)
      raise ValueError(
          f"LLM output is not valid JSON: {e}. "
          f"First 200 chars of cleaned text: {text[:200]!r}"
      ) from e
    region = Region(
        region_name=data["region_name"],
        districts=data["districts"],
        description=description,
        seed=self.seed,
    )
    logging.info(
        "Generated region '%s' with %d districts.",
        region.region_name,
        len(region.districts),
    )
    return region
