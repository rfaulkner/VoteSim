"""Sampler for political/social questions from the promptfoo dataset."""

import csv
import os
import random
from typing import Any, Dict, List, Optional


# Available topic categories in the dataset.
VALID_TOPICS = frozenset({"economic", "social"})

# Default topic when none is specified.
DEFAULT_TOPIC = "social"


class PoliticalQuestionSampler:
  """Samples political questions from the promptfoo/political-questions dataset.

  Questions can be filtered by topic axis (e.g. "social", "economic") to
  select domain-relevant prompts for the simulation.
  """

  def __init__(self, dataset_path: str):
    """Initialise from a CSV file.

    Args:
      dataset_path: Path to the political-questions.csv file.
    """
    self.dataset_path = dataset_path
    self.questions: List[Dict[str, Any]] = []
    self._by_topic: Dict[str, List[Dict[str, Any]]] = {}
    self._load_data()

  def _load_data(self):
    """Loads the CSV and indexes rows by topic axis."""
    if not os.path.exists(self.dataset_path):
      raise FileNotFoundError(
          f"Political questions dataset not found at: {self.dataset_path}"
      )

    with open(self.dataset_path, "r", encoding="utf-8") as f:
      reader = csv.DictReader(f)
      for row in reader:
        # Skip blank rows (the CSV has empty lines between sections).
        if not row.get("id"):
          continue
        entry = {
            "id": row["id"].strip(),
            "question": row["question"].strip(),
            "source": row.get("source", "").strip(),
            "topic": row.get("axis", "").strip().lower(),
        }
        self.questions.append(entry)
        self._by_topic.setdefault(entry["topic"], []).append(entry)

  @property
  def topics(self) -> List[str]:
    """Returns sorted list of available topic categories."""
    return sorted(self._by_topic.keys())

  def count(self, topic: Optional[str] = None) -> int:
    """Returns the number of questions, optionally filtered by topic."""
    if topic is None:
      return len(self.questions)
    return len(self._by_topic.get(topic.lower(), []))

  def sample(
      self,
      num_samples: int = 1,
      topic: Optional[str] = None,
      seed: Optional[int] = None,
  ) -> List[Dict[str, Any]]:
    """Sample questions, optionally filtered by topic axis.

    Args:
      num_samples: Number of questions to sample.
      topic: Topic category to filter by (e.g. "social", "economic").
        If None, samples from all questions.
      seed: Random seed for reproducibility. If None, uses current random
        state.

    Returns:
      List of question dicts with keys: id, question, source, topic.

    Raises:
      ValueError: If the specified topic is not found in the dataset.
    """
    if seed is not None:
      random.seed(seed)

    if topic is not None:
      topic_key = topic.lower()
      if topic_key not in self._by_topic:
        raise ValueError(
            f"Topic '{topic}' not found. Available topics: {self.topics}"
        )
      pool = self._by_topic[topic_key]
    else:
      pool = self.questions

    n = min(num_samples, len(pool))
    return random.sample(pool, n)

  def sample_question_text(
      self,
      topic: Optional[str] = None,
      seed: Optional[int] = None,
  ) -> str:
    """Convenience method: sample a single question and return its text.

    Args:
      topic: Topic category to filter by.
      seed: Random seed for reproducibility.

    Returns:
      The question text string.
    """
    samples = self.sample(num_samples=1, topic=topic, seed=seed)
    return samples[0]["question"]

