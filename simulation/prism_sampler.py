"""Functions to sample from the PRISM dataset and create personalization prompts."""

import json
import logging
import os
import random
from typing import Any, Dict, List, Optional


DEFAULT_NUM_PERSONAS = 120

# Countries where English is a primary or official language.
_ENGLISH_SPEAKING_COUNTRIES = frozenset({
    "Australia",
    "Canada",
    "Ireland",
    "New Zealand",
    "South Africa",
    "United Kingdom",
    "United States",
})


class PrismSampler:
  """Samples personalization prompts and demographics from the PRISM dataset.

  Builds a rich persona for each voter by combining:
  - Demographics (age, gender, education, employment, ethnicity, religion,
    marital status)
  - Self-description (the user's own values/worldview statement)
  - Statement-only examples from the user's conversation history (questions
    are filtered out so that the persona block contains only declarative
    opinions and statements)

  Only personas from English-speaking countries are included (location is
  used for filtering but not stored in demographics, since the voter's
  geographic context comes from their assigned district).

  By default, 5-10 examples are sampled per voter to provide richer
  personalization while keeping prompt length manageable.
  """

  MIN_EXAMPLES = 2
  MAX_EXAMPLES = 10

  def __init__(self, dataset_dir: str):
    self.dataset_dir = dataset_dir
    self.user_data: Dict[str, Dict[str, Any]] = {}
    self.user_statements: Dict[str, List[str]] = {}  # general statements
    self.user_political: Dict[str, List[str]] = {}   # political/social statements
    self.samples: List[str] = []
    self._load_data()

  def _load_data(self):
    """Loads the PRISM dataset from the specified directory."""
    survey_path = os.path.join(self.dataset_dir, "survey.jsonl")
    with open(survey_path, "r") as f:
      for line in f:
        data = json.loads(line)
        uid = data["user_id"]

        # Filter to English-speaking countries only.
        location = data.get("location", {})
        country = (
            location.get("reside_country", "")
            if isinstance(location, dict)
            else ""
        )
        if country not in _ENGLISH_SPEAKING_COUNTRIES:
          continue

        # Extract ethnicity/religion from nested dicts.
        ethnicity = _extract_simplified(data.get("ethnicity"))
        religion = _extract_simplified(data.get("religion"))

        self.user_data[uid] = {
            "age": data.get("age"),
            "gender": data.get("gender"),
            "employment": data.get("employment_status"),
            "education": data.get("education"),
            "ethnicity": ethnicity,
            "religion": religion,
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
        # Skip questions, task requests, terse topic names, and trivially
        # short greetings.
        if (_is_question(prompt)
            or _is_imperative_request(prompt)
            or _is_too_short_for_belief(prompt)):
          continue
        if len(prompt) < 15:
          continue
        # Only include users we already loaded (i.e. English-speaking).
        if uid not in self.user_data:
          continue
        if uid not in self.user_statements:
          self.user_statements[uid] = []
          self.user_political[uid] = []
        self.user_statements[uid].append(prompt)
        if _is_political_topic(prompt):
          self.user_political[uid].append(prompt)

    # Filter users who have both demographics and at least MIN_EXAMPLES
    # statements (ensures every persona has meaningful examples).
    self.samples = [
        u for u in self.user_data
        if u in self.user_statements
        and len(self.user_statements[u]) >= self.MIN_EXAMPLES
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
      all_stmts = self.user_statements[uid]
      political = self.user_political.get(uid, [])
      general = [s for s in all_stmts if s not in set(political)]

      # Prioritise political/social statements, then fill with general.
      n = rng.randint(lo, min(hi, max(lo, len(all_stmts))))
      n = min(n, len(all_stmts))

      # Take as many political statements as we can, then fill the rest.
      n_political = min(len(political), n)
      chosen_political = rng.sample(political, n_political)
      remaining = n - n_political
      n_general = min(len(general), remaining)
      chosen_general = rng.sample(general, n_general) if n_general > 0 else []
      examples = chosen_political + chosen_general
      rng.shuffle(examples)

      results.append({
          "user_id": uid,
          "demographics": self.user_data[uid],
          "examples": examples,
      })
    return results


# ---------------------------------------------------------------------------
# Cached persona loading
# ---------------------------------------------------------------------------


def load_or_create_personas(
    prism_dataset_dir: str,
    personas_path: str = "dataset/personas/personas.json",
    num_personas: int = DEFAULT_NUM_PERSONAS,
    seed: int = 42,
) -> List[Dict[str, Any]]:
  """Load a cached persona set or create one from PRISM.

  On first call the function samples ``num_personas`` voters from the
  PRISM dataset, writes the result to ``personas_path``, and returns
  the list.  On subsequent calls (or on different machines sharing the
  same ``personas_path``), the cached file is loaded directly without
  touching PRISM.

  Args:
    prism_dataset_dir: Path to the PRISM dataset directory.
    personas_path: Where to store / load the cached personas JSON.
    num_personas: Number of personas to sample when creating the file.
    seed: Random seed for sampling (only used when creating).

  Returns:
    A list of persona dicts, each with ``user_id``,
    ``demographics``, and ``examples`` keys.
  """
  if os.path.exists(personas_path):
    logging.info("Loading cached personas from %s", personas_path)
    with open(personas_path, "r") as f:
      personas = json.load(f)
    logging.info("Loaded %d cached personas.", len(personas))
    return personas

  logging.info(
      "No cached personas at %s — sampling %d from PRISM.",
      personas_path,
      num_personas,
  )
  sampler = PrismSampler(prism_dataset_dir)
  personas = sampler.sample(num_samples=num_personas, seed=seed)

  # Ensure the output directory exists.
  os.makedirs(os.path.dirname(personas_path), exist_ok=True)
  with open(personas_path, "w") as f:
    json.dump(personas, f, indent=2)
  logging.info(
      "Cached %d personas to %s.", len(personas), personas_path
  )
  return personas


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Leading verbs that signal a task / command rather than a belief or opinion.
_IMPERATIVE_VERBS = frozenset({
    "write", "give", "help", "create", "make", "generate", "list",
    "plan", "design", "build", "draw", "find", "search", "show",
    "suggest", "recommend", "provide", "explain", "describe",
    "summarize", "summarise", "translate", "calculate", "compare",
    "tell", "prepare", "draft", "compose", "outline", "send",
    "book", "please", "rank", "rate", "convert", "imagine",
    "present", "define", "justify", "debate",
})

# Phrase prefixes that indicate a request rather than a statement.
_REQUEST_PREFIXES = (
    "i want to ",
    "i want you ",
    "i want concrete ",
    "i need ",
    "i would like ",
    "i'm looking for ",
    "please ",
    "give me ",
    "help me ",
    "tell me ",
    "can you ",
    "could you ",
    "write me ",
    "write a ",
    "write an ",
    "create a ",
    "create an ",
    "show me ",
    "find me ",
    "provide me ",
    "suggest ",
    "recommend ",
    "i need help ",
    "i need a ",
    "i need some ",
    "i need you ",
    "need to complete ",
    "talk about ",
    "let's talk ",
    "let's play ",
    "lets talk ",
    "hello ",
    "hey ",
    "hi ",
    "five ideal ",
    "date night ",
    "one paragraph ",
    "i have a flask ",
    "i have art block",
    "i have adopted ",
    "i have a second ",
)

# Substrings that appear anywhere in a prompt and signal a request.
_REQUEST_SUBSTRINGS = (
    "have you any advice",
    "please create",
    "please provide",
    "please give",
    "please help",
    "got any ideas",
    "any suggestions",
)


def _is_imperative_request(text: str) -> bool:
  """Return True if the text is a task/command rather than a belief.

  Filters out prompts like 'Write a story...', 'Give me a recipe...',
  'Tell me about...', 'I need help...' which do not reveal personal
  beliefs, opinions, or values useful for voter personalisation.
  """
  lower = text.strip().lower()
  first_word = lower.split()[0] if lower else ""
  if first_word in _IMPERATIVE_VERBS:
    return True
  if any(lower.startswith(prefix) for prefix in _REQUEST_PREFIXES):
    return True
  return any(sub in lower for sub in _REQUEST_SUBSTRINGS)


def _is_too_short_for_belief(text: str) -> bool:
  """Return True if the text is too terse to express a belief.

  Filters out entries that are bare topic names (e.g. 'Capital
  punishment', 'political issues') rather than actual statements
  of opinion. Requires at least 5 words for a statement to be
  considered a genuine expression of belief.
  """
  words = text.strip().split()
  return len(words) < 5

def _is_question(text: str) -> bool:
  """Return True if the text looks like a question rather than a statement.

  Checks for trailing '?' and question-word-initial patterns (e.g.
  'who is ...', 'what do ...') even without punctuation.
  """
  stripped = text.strip()
  if stripped.endswith("?"):
    return True
  first_word = stripped.lower().split()[0] if stripped else ""
  return first_word in _QUESTION_WORDS


# Words that signal a question when they appear at the start of a prompt.
_QUESTION_WORDS = frozenset({
    "who", "what", "where", "when", "why", "how",
    "is", "are", "do", "does", "did",
    "can", "could", "would", "should", "will", "shall",
    "which", "whats", "whos", "hows", "wheres",
    "what's", "who's", "where's", "how's", "when's",
})


# Keywords that indicate a statement touches political or social topics.
_POLITICAL_KEYWORDS = frozenset({
    "government", "tax", "policy", "vote", "election", "party",
    "law", "rights", "right", "freedom", "democracy", "justice",
    "immigration", "healthcare", "climate", "economy", "welfare",
    "housing", "education", "military", "gun", "abortion",
    "inequality", "poverty", "liberal", "conservative", "socialist",
    "capitalism", "regulation", "police", "prison", "reform",
    "equality", "discrimination", "religion", "war", "peace",
    "trade", "environment", "energy", "nuclear", "drugs", "crime",
    "security", "censorship", "speech", "privacy", "surveillance",
    "politics", "political", "racism", "sexism", "feminist",
    "protest", "activism", "union", "unions", "wealth",
    "corruption", "constitution", "ideology", "progressive",
    "libertarian", "nationalist", "populist", "parliament",
    "congress", "senate", "president", "minister", "mayor",
    "border", "refugee", "asylum", "sanctions", "diplomacy",
    "patriot", "sovereignty", "social", "society", "civic",
    "euthanasia", "death penalty", "capital punishment",
    "homelessness", "affordable", "wages", "income",
    "diversity", "inclusion", "sustainability",
})


def _is_political_topic(text: str) -> bool:
  """Return True if the text touches political or social topics."""
  lower = text.lower()
  return any(kw in lower for kw in _POLITICAL_KEYWORDS)


def _extract_simplified(nested: Any) -> str:
  """Pull the 'simplified' or 'categorised' label from a nested dict."""
  if not isinstance(nested, dict):
    return ""
  val = nested.get("simplified") or nested.get("categorised", "")
  if val and val.lower() in ("prefer not to say", "unknown"):
    return ""
  return val
