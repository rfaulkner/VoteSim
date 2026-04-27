"""Generator for ideology-specific policy responses to social issues.

Given a party platform, a set of social issues, and free-form voter responses,
this module uses an LLM to produce structured policy proposals that reflect the
party's ideology while accounting for constituent sentiment.
"""

import json
import logging
import os
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pathfinder import assistant
from pathfinder import gen
from pathfinder import get_model
from pathfinder import user

# ---------------------------------------------------------------------------
# Prompt template
# ---------------------------------------------------------------------------

POLICY_PROMPT = """You are a senior political policy advisor for {party_name}, \
a {ideology} party.

=== PARTY PLATFORM ===
{platform_summary}

=== SOCIAL ISSUE(S) TO ADDRESS ===
{issues_block}

=== VOTER RESPONSES FROM YOUR CONSTITUENCY ===
The following are free-form responses from potential voters in your district. \
Use them to gauge constituent sentiment and tailor your proposals accordingly.

{voter_block}

=== INSTRUCTIONS ===
For EACH social issue listed above, draft a policy response as a JSON object \
with these fields:
- "issue": string — the issue name exactly as given above
- "position_statement": string — 1-2 sentence official party position
- "key_proposals": array of 3-5 strings — concrete, actionable policy proposals
- "voter_alignment_score": float 0-1 — estimated alignment between this \
response and the voter sentiments provided (1 = perfect alignment)
- "reasoning": string — brief explanation of how the party platform and voter \
feedback shaped this response

Return a JSON array of these objects (one per issue).  Return ONLY valid JSON \
— no markdown fences, no commentary."""


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

class PartyPlatform:
  """A political party's baseline policy platform loaded from JSON."""

  def __init__(self, data: Dict[str, Any], source_path: Optional[str] = None):
    self.data = data
    self.source_path = source_path
    self.party_name: str = data["party_name"]
    self.ideology: str = data["ideology"]
    self.core_values: List[str] = data.get("core_values", [])

  # -- I/O ----------------------------------------------------------------

  @classmethod
  def from_json(cls, path: str) -> "PartyPlatform":
    """Load a platform from a JSON file."""
    with open(path, "r", encoding="utf-8") as f:
      data = json.load(f)
    return cls(data, source_path=path)

  def to_json(self, path: str) -> None:
    """Save the platform to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
      json.dump(self.data, f, indent=2)
    logging.info("Saved platform to %s", path)

  # -- Summary for prompt injection ----------------------------------------

  def summary(self) -> str:
    """Return a human-readable summary suitable for LLM prompt injection."""
    lines = [
        f"Party: {self.party_name} ({self.ideology})",
        f"Core Values: {', '.join(self.core_values)}",
    ]
    # Walk the top-level policy sections and flatten them.
    for section_key in (
        "economic_policy",
        "social_policy",
        "governance",
        "environment",
        "security",
    ):
      section = self.data.get(section_key)
      if not section:
        continue
      header = section_key.replace("_", " ").title()
      lines.append(f"\n{header}:")
      for k, v in section.items():
        if isinstance(v, list):
          v = "; ".join(v)
        lines.append(f"  - {k.replace('_', ' ').title()}: {v}")
    return "\n".join(lines)

  def __repr__(self) -> str:
    return f"PartyPlatform({self.party_name!r}, ideology={self.ideology!r})"


@dataclass
class PolicyResponse:
  """A single policy response for one social issue."""

  party_name: str
  ideology: str
  issue: str
  position_statement: str
  key_proposals: List[str]
  voter_alignment_score: float
  reasoning: str

  def to_dict(self) -> Dict[str, Any]:
    return {
        "party_name": self.party_name,
        "ideology": self.ideology,
        "issue": self.issue,
        "position_statement": self.position_statement,
        "key_proposals": self.key_proposals,
        "voter_alignment_score": self.voter_alignment_score,
        "reasoning": self.reasoning,
    }


# ---------------------------------------------------------------------------
# Generator
# ---------------------------------------------------------------------------

class PolicyGenerator:
  """Uses an LLM to generate policy responses grounded in a party platform."""

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

  # -- Core generation -----------------------------------------------------

  def generate(
      self,
      platform: PartyPlatform,
      issues: List[str],
      voter_responses: List[str],
      temperature: float = 0.7,
  ) -> List[PolicyResponse]:
    """Generate policy responses for the given issues.

    Args:
      platform: The party platform to ground the response in.
      issues: One or more social issue names / descriptions.
      voter_responses: Free-form text responses from potential voters
        expressing their views on the issue(s).
      temperature: Sampling temperature for the LLM.

    Returns:
      A list of PolicyResponse objects, one per issue.
    """
    issues_block = "\n".join(f"- {issue}" for issue in issues)
    voter_block = "\n".join(
        f'  Voter {i + 1}: "{resp}"' for i, resp in enumerate(voter_responses)
    )
    if not voter_responses:
      voter_block = "  (No voter responses provided.)"

    prompt = POLICY_PROMPT.format(
        party_name=platform.party_name,
        ideology=platform.ideology,
        platform_summary=platform.summary(),
        issues_block=issues_block,
        voter_block=voter_block,
    )

    lm = self.model
    with user():
      lm += prompt
    with assistant():
      lm += gen(max_tokens=8192, temperature=temperature, name="policy_json")

    raw = lm["policy_json"]
    logging.info("Raw policy output length: %d chars", len(raw))
    logging.info("Raw policy output (first 500 chars): %.500s", raw)

    parsed = self._parse_response(raw, platform)
    logging.info(
        "Generated %d policy response(s) for %s.",
        len(parsed),
        platform.party_name,
    )
    return parsed

  # -- Batch helper --------------------------------------------------------

  def generate_for_all_parties(
      self,
      party_dir: str,
      issues: List[str],
      voter_responses: List[str],
      temperature: float = 0.7,
  ) -> Dict[str, List[PolicyResponse]]:
    """Generate policy responses for every party platform in a directory.

    Loads all ``*.json`` files from *party_dir*, runs :meth:`generate` for
    each, and returns a dict keyed by ideology.

    Args:
      party_dir: Path to directory containing party platform JSON files.
      issues: Social issue(s) to address.
      voter_responses: Free-form voter responses.
      temperature: Sampling temperature.

    Returns:
      ``{ideology: [PolicyResponse, ...], ...}``
    """
    results: Dict[str, List[PolicyResponse]] = {}
    for filename in sorted(os.listdir(party_dir)):
      if not filename.endswith(".json"):
        continue
      path = os.path.join(party_dir, filename)
      try:
        platform = PartyPlatform.from_json(path)
      except (json.JSONDecodeError, KeyError) as e:
        logging.warning("Skipping %s: %s", path, e)
        continue
      logging.info(
          "Generating policy for %s (%s)...",
          platform.party_name,
          platform.ideology,
      )
      results[platform.ideology] = self.generate(
          platform, issues, voter_responses, temperature
      )
    return results

  # -- Parsing helpers -----------------------------------------------------

  @staticmethod
  def _parse_response(
      raw: str, platform: PartyPlatform
  ) -> List[PolicyResponse]:
    """Parse the raw LLM output into PolicyResponse objects."""
    # Strip <think>...</think> blocks (chain-of-thought wrappers).
    text = re.sub(r"<think>.*?(</think>|$)", "", raw, flags=re.DOTALL).strip()

    # Strip markdown fences if present.
    if text.startswith("```"):
      text = text.split("\n", 1)[1]
      text = text.rsplit("```", 1)[0]
      text = text.strip()

    if not text:
      raise ValueError(
          "LLM returned empty response for policy generation. "
          f"Raw output: {raw!r:.200}"
      )

    try:
      data = json.loads(text)
    except json.JSONDecodeError as e:
      logging.error(
          "Failed to parse policy JSON. Content: %.500s", text
      )
      raise ValueError(
          f"LLM output is not valid JSON: {e}. "
          f"First 200 chars: {text[:200]!r}"
      ) from e

    # Accept both a bare list and a wrapped {"policies": [...]} object.
    if isinstance(data, dict):
      data = data.get("policies", data.get("responses", [data]))
    if not isinstance(data, list):
      data = [data]

    responses = []
    for item in data:
      responses.append(
          PolicyResponse(
              party_name=platform.party_name,
              ideology=platform.ideology,
              issue=item.get("issue", "unknown"),
              position_statement=item.get("position_statement", ""),
              key_proposals=item.get("key_proposals", []),
              voter_alignment_score=float(
                  item.get("voter_alignment_score", 0.0)
              ),
              reasoning=item.get("reasoning", ""),
          )
      )
    return responses
