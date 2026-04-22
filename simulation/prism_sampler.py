"""A class to sample personalization prompts and demographics from the PRISM dataset."""

import json
import os
import random
from typing import Dict, List


class PrismSampler:
  """A class to sample personalization prompts and demographics from the PRISM dataset."""

  def __init__(self, dataset_dir: str):
    self.dataset_dir = dataset_dir
    self.user_data = {}
    self.samples = []
    self._load_data()

  def _load_data(self):
    """Loads survey data and conversations from the PRISM dataset and links them."""
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

    # Load conversations and link with demographics
    conv_path = os.path.join(self.dataset_dir, "conversations.jsonl")
    with open(conv_path, "r") as f:
      for line in f:
        data = json.loads(line)
        user_id = data.get("user_id")
        if user_id in self.user_data and "opening_prompt" in data:
          self.samples.append({
              "prompt": data["opening_prompt"],
              "demographics": self.user_data[user_id]
          })

  def sample(self, num_samples: int, seed: int) -> List[Dict]:
    """Samples persona-prompt pairs."""
    random.seed(seed)
    return random.sample(self.samples, min(num_samples, len(self.samples)))
