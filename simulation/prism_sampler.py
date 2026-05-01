"""Functions to sample from the PRISM dataset and create personalization prompts."""

import json
import os
import random
from typing import Any, Dict, List, Optional


class PrismSampler:
  """Samples personalization prompts and demographics from the PRISM dataset.

  Builds a rich persona for each voter by combining:
  - Demographics (age, gender, education, employment, ethnicity, religion,
    location, marital status)
  - Self-description (the user's own values/worldview statement)
  - Statement-only examples from the user's conversation history (questions
    are filtered out so that the persona block contains only declarative
    opinions and statements)

  By default, 5-10 examples are sampled per voter to provide richer
  personalization while keeping prompt length manageable.
  """

  MIN_EXAMPLES = 5
  MAX_EXAMPLES = 10

  def __init__(self, dataset_dir: str):
    self.dataset_dir = dataset_dir
    self.user_data: Dict[str, Dict[str, Any]] = {}
    self.user_statements: Dict[str, List[str]] = {}
    self.samples: List[str] = []
    self._load_data()

  def _load_data(self):
    """Loads the PRISM dataset from the specified directory."""
    survey_path = os.path.join(self.dataset_dir, "survey.jsonl")
    with open(survey_path, "r") as f:
      for line in f:
        data = json.loads(line)
        uid = data["user_id"]

        # Extract ethnicity/religion/location from nested dicts.
        ethnicity = _extract_simplified(data.get("ethnicity"))
        religion = _extract_simplified(data.get("religion"))
        location = _extract_location(data.get("location"))

        self.user_data[uid] = {
            "age": data.get("age"),
            "gender": data.get("gender"),
            "employment": data.get("employment_status"),
            "education": data.get("education"),
            "ethnicity": ethnicity,
            "religion": religion,
            "location": location,
            "marital_status": data.get("marital_status"),
            "self_description": data.get("self_description", ""),
        }

    # Load conversations and collect statements (non-question prompts).
    conv_path = os.path.join(self.dataset_dir, "conversations.jsonl")
    with open(conv_path, "r") as f:
      for line in f:
        data = json.loads(line)
        uid = data.get("user_id")
        prompt = data.get("opening_prompt", "").strip()
        if not uid or not prompt:
          continue
        # Keep only statements — skip entries that are questions or
        # trivially short greetings.
        if _is_question(prompt) or len(prompt) < 15:
          continue
        if uid not in self.user_statements:
          self.user_statements[uid] = []
        self.user_statements[uid].append(prompt)

    # Filter users who have both demographics and at least some statements.
    self.samples = [
        u for u in self.user_data if u in self.user_statements
    ]

  def sample(
      self,
      num_samples: int,
      seed: int,
      min_examples: Optional[int] = None,
      max_examples: Optional[int] = None,
  ) -> List[Dict[str, Any]]:
    """Samples personas with demographics and statement examples.

    Args:
      num_samples: Number of voters to sample.
      seed: Random seed for reproducibility.
      min_examples: Min statements per voter (default MIN_EXAMPLES).
      max_examples: Max statements per voter (default MAX_EXAMPLES).

    Returns:
      List of dicts with 'user_id', 'demographics', and 'examples'.
    """
    rng = random.Random(seed)
    lo = min_examples if min_examples is not None else self.MIN_EXAMPLES
    hi = max_examples if max_examples is not None else self.MAX_EXAMPLES

    sampled_ids = rng.sample(
        self.samples, min(num_samples, len(self.samples))
    )

    results = []
    for uid in sampled_ids:
      statements = self.user_statements[uid]
      # Sample between lo and hi examples (clamped to available count).
      n = rng.randint(lo, min(hi, max(lo, len(statements))))
      n = min(n, len(statements))
      examples = rng.sample(statements, n)

      results.append({
          "user_id": uid,
          "demographics": self.user_data[uid],
          "examples": examples,
      })
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _is_question(text: str) -> bool:
  """Return True if the text looks like a question rather than a statement."""
  return text.rstrip().endswith("?")


def _extract_simplified(nested: Any) -> str:
  """Pull the 'simplified' or 'categorised' label from a nested dict."""
  if not isinstance(nested, dict):
    return ""
  val = nested.get("simplified") or nested.get("categorised", "")
  if val and val.lower() in ("prefer not to say", "unknown"):
    return ""
  return val


def _extract_location(nested: Any) -> str:
  """Build a concise location string from the PRISM location dict."""
  if not isinstance(nested, dict):
    return ""
  country = nested.get("reside_country", "")
  return country
