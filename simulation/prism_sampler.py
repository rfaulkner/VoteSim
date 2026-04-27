"""Functions to sample from the PRISM dataset and create personalization prompts."""

import json
import os
import random
from typing import Any, Dict, List


class PrismSampler:
  """A class to sample personalization prompts and demographics from the PRISM dataset."""

  def __init__(self, dataset_dir: str):
    self.dataset_dir = dataset_dir
    self.user_data = {}
    self.user_prompts = {}
    self.samples = []
    self._load_data()

  def _load_data(self):
    """Loads the PRISM dataset from the specified directory."""
    survey_path = os.path.join(self.dataset_dir, "survey.jsonl")
    with open(survey_path, "r") as f:
      for line in f:
        data = json.loads(line)
        self.user_data[data["user_id"]] = {
            "age": data.get("age"),
            "gender": data.get("gender"),
            "employment": data.get("employment_status"),
            "education": data.get("education"),
        }

    # Load conversations and group by user
    conv_path = os.path.join(self.dataset_dir, "conversations.jsonl")
    with open(conv_path, "r") as f:
      for line in f:
        data = json.loads(line)
        user_id = data.get("user_id")
        if user_id and "opening_prompt" in data:
          if user_id not in self.user_prompts:
            self.user_prompts[user_id] = []
          self.user_prompts[user_id].append(data["opening_prompt"])

    # Filter users who have both demographics and prompts
    self.samples = [u for u in self.user_data if u in self.user_prompts]

  def sample(self, num_samples: int, seed: int) -> List[Dict[str, Any]]:
    """Samples persona with demographics and example prompts."""
    random.seed(seed)
    sampled_user_ids = random.sample(
        self.samples, min(num_samples, len(self.samples))
    )

    results = []
    for u_id in sampled_user_ids:
      prompts = self.user_prompts[u_id]
      num_examples = min(3, len(prompts))
      examples = random.sample(prompts, num_examples)

      results.append({
          "user_id": u_id,
          "demographics": self.user_data[u_id],
          "examples": examples,
      })
    return results
