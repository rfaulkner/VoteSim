#!/usr/bin/env python3
"""Generate 800 voter personas directly from the PRISM dataset.

Samples English-speaking users from PRISM survey + conversations data,
keeping their real demographics, self_descriptions, and example comments.
Strictly filters out questions, imperatives, and short prompts.
Better to have fewer high-quality examples than noisy ones.

Usage:
    python generate_from_prism.py
"""

from collections import Counter
import json
import os
import random
import re
from typing import Any, Dict, List

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PRISM_DIR = os.path.join(SCRIPT_DIR, "..", "prism")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "personas.json")

TARGET_COUNT = 800
SEED = 42

ENGLISH_SPEAKING_COUNTRIES = frozenset({
    "Australia",
    "Canada",
    "Ireland",
    "New Zealand",
    "South Africa",
    "United Kingdom",
    "United States",
})

DISTRICTS = [
    "Steelhaven",
    "Ironforge Centre",
    "Ashford",
    "Lakeshore",
    "Silverpine West",
    "Millhaven",
    "Bramblewood",
    "Dunmore",
]
DISTRICT_WEIGHTS = [0.22, 0.18, 0.16, 0.12, 0.12, 0.08, 0.07, 0.05]

# ── Filtering ──────────────────────────────────────────────────────────────

_QUESTION_WORDS = frozenset({
    "who",
    "what",
    "where",
    "when",
    "why",
    "how",
    "is",
    "are",
    "do",
    "does",
    "did",
    "can",
    "could",
    "would",
    "should",
    "will",
    "shall",
    "which",
    "whats",
    "whos",
    "hows",
    "wheres",
    "what's",
    "who's",
    "where's",
    "how's",
    "when's",
    "have",
    "has",
})

_IMPERATIVE_VERBS = frozenset({
    "write",
    "give",
    "help",
    "create",
    "make",
    "generate",
    "list",
    "plan",
    "design",
    "build",
    "draw",
    "find",
    "search",
    "show",
    "suggest",
    "recommend",
    "provide",
    "explain",
    "describe",
    "summarize",
    "summarise",
    "translate",
    "calculate",
    "compare",
    "tell",
    "prepare",
    "draft",
    "compose",
    "outline",
    "send",
    "book",
    "please",
    "rank",
    "rate",
    "convert",
    "imagine",
    "present",
    "define",
    "justify",
    "debate",
    "name",
})

_REQUEST_PREFIXES = (
    "i want to ",
    "i want you ",
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
    "talk about ",
    "let's talk ",
    "let's play ",
    "lets talk ",
    "hello",
    "hey ",
    "hey,",
    "hi ",
    "hi,",
    "good morning",
    "good evening",
)


def _is_question(text: str) -> bool:
  stripped = text.strip()
  if stripped.endswith("?"):
    return True
  first_word = stripped.lower().split()[0] if stripped else ""
  return first_word in _QUESTION_WORDS


def _is_imperative(text: str) -> bool:
  lower = text.strip().lower()
  first_word = lower.split()[0] if lower else ""
  if first_word in _IMPERATIVE_VERBS:
    return True
  return any(lower.startswith(p) for p in _REQUEST_PREFIXES)


def _is_too_short(text: str) -> bool:
  return len(text.strip().split()) < 5


def _is_good_example(text: str) -> bool:
  """Return True if the text is a statement of opinion/belief worth keeping."""
  text = text.strip()
  if not text:
    return False
  if _is_too_short(text):
    return False
  if _is_question(text):
    return False
  if _is_imperative(text):
    return False
  return True


_POLITICAL_KEYWORDS = frozenset({
    "government",
    "tax",
    "taxes",
    "policy",
    "vote",
    "election",
    "party",
    "law",
    "rights",
    "right",
    "freedom",
    "democracy",
    "justice",
    "immigration",
    "immigrant",
    "healthcare",
    "climate",
    "economy",
    "welfare",
    "housing",
    "education",
    "military",
    "gun",
    "abortion",
    "inequality",
    "poverty",
    "liberal",
    "conservative",
    "socialist",
    "capitalism",
    "regulation",
    "police",
    "prison",
    "reform",
    "equality",
    "discrimination",
    "religion",
    "war",
    "peace",
    "trade",
    "environment",
    "energy",
    "nuclear",
    "drugs",
    "crime",
    "security",
    "censorship",
    "speech",
    "privacy",
    "surveillance",
    "politics",
    "political",
    "racism",
    "sexism",
    "feminist",
    "protest",
    "activism",
    "union",
    "wealth",
    "corruption",
    "constitution",
    "ideology",
    "progressive",
    "libertarian",
    "nationalist",
    "populist",
    "parliament",
    "congress",
    "senate",
    "president",
    "minister",
    "mayor",
    "border",
    "refugee",
    "asylum",
    "sanctions",
    "diplomacy",
    "patriot",
    "sovereignty",
    "social",
    "society",
    "civic",
    "euthanasia",
    "death penalty",
    "homelessness",
    "homeless",
    "affordable",
    "wages",
    "income",
    "diversity",
    "inclusion",
    "sustainability",
    "nhs",
    "trans",
    "transgender",
    "lgbtq",
    "gay",
    "lesbian",
    "racial",
    "race",
    "ethnic",
    "minority",
    "pandemic",
    "covid",
    "lockdown",
    "vaccine",
    "cost of living",
    "inflation",
    "recession",
    "billionaire",
    "corporation",
    "monopoly",
    "human rights",
    "women's rights",
    "pro-life",
    "pro-choice",
    "mental health",
    "disability",
    "fossil fuel",
    "renewable",
    "carbon",
    "media",
    "misinformation",
    "propaganda",
    "bias",
    "oppression",
    "privilege",
    "terrorism",
    "extremism",
    "israel",
    "palestine",
    "china",
    "russia",
    "ukraine",
    "married",
    "marriage",
    "family",
    "values",
    "faith",
    "church",
    "mosque",
    "bible",
    "god",
    "atheist",
    "agnostic",
})


def _is_sociopolitical(text: str) -> bool:
  lower = text.lower()
  return any(kw in lower for kw in _POLITICAL_KEYWORDS)


def _extract_simplified(nested: Any) -> str:
  if not isinstance(nested, dict):
    return ""
  val = nested.get("simplified") or nested.get("categorised", "")
  if val and val.lower() in ("prefer not to say", "unknown"):
    return ""
  return val


def main():
  rng = random.Random(SEED)

  # 1. Load survey data
  users: Dict[str, Dict] = {}
  with open(os.path.join(PRISM_DIR, "survey.jsonl")) as f:
    for line in f:
      d = json.loads(line)
      loc = d.get("location", {})
      country = loc.get("reside_country", "") if isinstance(loc, dict) else ""
      if country not in ENGLISH_SPEAKING_COUNTRIES:
        continue
      uid = d["user_id"]
      users[uid] = {
          "demographics": {
              "age": d.get("age", ""),
              "gender": d.get("gender", ""),
              "employment": d.get("employment_status", ""),
              "education": d.get("education", ""),
              "ethnicity": _extract_simplified(d.get("ethnicity")),
              "religion": _extract_simplified(d.get("religion")),
              "marital_status": d.get("marital_status", ""),
              "self_description": d.get("self_description", ""),
          },
          "political": [],
          "general": [],
      }

  print(f"Loaded {len(users)} English-speaking PRISM users")

  # 2. Load conversations — strict filtering
  with open(os.path.join(PRISM_DIR, "conversations.jsonl")) as f:
    for line in f:
      d = json.loads(line)
      uid = d.get("user_id")
      prompt = d.get("opening_prompt", "").strip()
      if not uid or not prompt or uid not in users:
        continue
      if not _is_good_example(prompt):
        continue
      if _is_sociopolitical(prompt):
        users[uid]["political"].append(prompt)
      else:
        users[uid]["general"].append(prompt)

  # 3. Build candidate list — any user with at least 1 good example + desc
  candidates = []
  for uid, data in users.items():
    n_examples = len(data["political"]) + len(data["general"])
    desc = data["demographics"].get("self_description", "").strip()
    if n_examples >= 1 and desc:
      candidates.append(uid)

  print(f"Candidates with >= 1 good example: {len(candidates)}")

  # Stats on example counts
  ex_dist = Counter()
  for uid in candidates:
    n = len(users[uid]["political"]) + len(users[uid]["general"])
    ex_dist[n] += 1
  print("Example count distribution among candidates:")
  for k in sorted(ex_dist):
    print(f"  {k}: {ex_dist[k]} users")

  # 4. Sample 800
  sampled = rng.sample(candidates, min(TARGET_COUNT, len(candidates)))
  print(f"\nSampled {len(sampled)} users")

  # 5. Assign districts
  total_w = sum(DISTRICT_WEIGHTS)
  norm_w = [w / total_w for w in DISTRICT_WEIGHTS]
  dcounts = {d: int(w * len(sampled)) for d, w in zip(DISTRICTS, norm_w)}
  rem = len(sampled) - sum(dcounts.values())
  resid = {
      d: (w * len(sampled)) - dcounts[d] for d, w in zip(DISTRICTS, norm_w)
  }
  for d in sorted(resid, key=resid.get, reverse=True)[:rem]:
    dcounts[d] += 1
  dlist = []
  for d, c in dcounts.items():
    dlist.extend([d] * c)
  rng.shuffle(dlist)

  # 6. Build personas — prioritise political examples
  personas = []
  for i, uid in enumerate(sampled):
    data = users[uid]
    pol = list(data["political"])
    gen = list(data["general"])

    # Take all political, then fill with general, cap at 5
    rng.shuffle(pol)
    rng.shuffle(gen)
    chosen = pol + gen
    chosen = chosen[:5]  # max 5

    personas.append({
        "user_id": uid,
        "demographics": data["demographics"],
        "examples": chosen,
        "district": dlist[i],
    })

  # 7. Summary
  print(f"\n{'='*60}")
  print(f"Total personas: {len(personas)}")

  ec = Counter(len(p["examples"]) for p in personas)
  print(f"\nExamples per persona:")
  for k, v in sorted(ec.items()):
    print(f"  {k}: {v} personas")

  for field in ["age", "gender", "ethnicity", "religion"]:
    counts = Counter(p["demographics"][field] for p in personas)
    print(f"\n{field}:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1])[:6]:
      print(f"  {k or '(empty)'}: {v} ({v/len(personas)*100:.1f}%)")

  print(f"\ndistrict:")
  for k, v in sorted(
      Counter(p["district"] for p in personas).items(), key=lambda x: -x[1]
  ):
    print(f"  {k}: {v} ({v/len(personas)*100:.1f}%)")

  # 8. Save
  with open(OUTPUT_FILE, "w") as f:
    json.dump(personas, f, indent=2, ensure_ascii=False)
  print(f"\nSaved {len(personas)} personas to {OUTPUT_FILE}")


if __name__ == "__main__":
  main()
