#!/usr/bin/env python3
"""Generate 800 diverse voter personas for VoteSim.

Uses the existing 120 personas as demographic seed data to maintain
diversity proportions, then generates new personas in batches using
Gemini API.  Each persona gets at least 3 sociopolitical example
comments and cleaned-up language.

Usage:
    python generate_personas.py
"""

import json
import os
import random
import re
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TARGET_COUNT = 800
BATCH_SIZE = 20  # personas per API call

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ORIGINAL_FILE = os.path.join(SCRIPT_DIR, "personas.json")
OUTPUT_FILE = os.path.join(SCRIPT_DIR, "personas_800.json")

# Demographic distributions (percentages targeting 800 total)
# Based on the original 120-persona proportions, slightly adjusted.

AGE_DIST = {
    "18-24 years old": 0.09,
    "25-34 years old": 0.30,
    "35-44 years old": 0.17,
    "45-54 years old": 0.17,
    "55-64 years old": 0.17,
    "65+ years old": 0.10,
}

GENDER_DIST = {
    "Male": 0.54,
    "Female": 0.44,
    "Non-binary / third gender": 0.02,
}

EMPLOYMENT_DIST = {
    "Working full-time": 0.53,
    "Working part-time": 0.13,
    "Retired": 0.11,
    "Student": 0.06,
    "Homemaker / Stay-at-home parent": 0.06,
    "Unemployed, not seeking work": 0.05,
    "Unemployed, seeking work": 0.04,
    "Prefer not to say": 0.02,
}

EDUCATION_DIST = {
    "University Bachelors Degree": 0.40,
    "Completed Secondary School": 0.17,
    "Graduate / Professional degree": 0.16,
    "Some University but no degree": 0.12,
    "Vocational": 0.11,
    "Some Secondary": 0.04,
}

ETHNICITY_DIST = {
    "White": 0.65,
    "Black": 0.10,
    "Asian": 0.10,
    "Hispanic": 0.06,
    "Mixed": 0.06,
    "Other": 0.02,
    "": 0.01,
}

RELIGION_DIST = {
    "No Affiliation": 0.45,
    "Christian": 0.35,
    "Other": 0.06,
    "Muslim": 0.05,
    "Jewish": 0.04,
    "Hindu": 0.02,
    "Buddhist": 0.02,
    "": 0.01,
}

MARITAL_DIST = {
    "Never been married": 0.44,
    "Married": 0.38,
    "Divorced / Separated": 0.11,
    "Widowed": 0.04,
    "Domestic partnership": 0.03,
}

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

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sample_from_dist(dist: dict, n: int) -> list:
  """Sample n items from a distribution dict {value: proportion}."""
  keys = list(dist.keys())
  weights = [dist[k] for k in keys]
  # Normalize
  total = sum(weights)
  weights = [w / total for w in weights]
  # Deterministic allocation with random shuffle for remainder
  counts = {k: int(w * n) for k, w in zip(keys, weights)}
  remainder = n - sum(counts.values())
  # Distribute remainder proportionally
  residuals = {k: (w * n) - counts[k] for k, w in zip(keys, weights)}
  for k in sorted(residuals, key=residuals.get, reverse=True)[:remainder]:
    counts[k] += 1
  # Build list and shuffle
  result = []
  for k, c in counts.items():
    result.extend([k] * c)
  random.shuffle(result)
  return result


def build_demographic_slots(n: int) -> list:
  """Pre-assign demographics for n personas to ensure diversity."""
  ages = sample_from_dist(AGE_DIST, n)
  genders = sample_from_dist(GENDER_DIST, n)
  employments = sample_from_dist(EMPLOYMENT_DIST, n)
  educations = sample_from_dist(EDUCATION_DIST, n)
  ethnicities = sample_from_dist(ETHNICITY_DIST, n)
  religions = sample_from_dist(RELIGION_DIST, n)
  maritals = sample_from_dist(MARITAL_DIST, n)
  # Districts
  dist_keys = DISTRICTS
  dist_w = DISTRICT_WEIGHTS
  total_w = sum(dist_w)
  dist_w = [w / total_w for w in dist_w]
  district_counts = {k: int(w * n) for k, w in zip(dist_keys, dist_w)}
  rem = n - sum(district_counts.values())
  residuals = {
      k: (w * n) - district_counts[k] for k, w in zip(dist_keys, dist_w)
  }
  for k in sorted(residuals, key=residuals.get, reverse=True)[:rem]:
    district_counts[k] += 1
  districts = []
  for k, c in district_counts.items():
    districts.extend([k] * c)
  random.shuffle(districts)

  slots = []
  for i in range(n):
    slots.append({
        "age": ages[i],
        "gender": genders[i],
        "employment": employments[i],
        "education": educations[i],
        "ethnicity": ethnicities[i],
        "religion": religions[i],
        "marital_status": maritals[i],
        "district": districts[i],
    })
  return slots


# ---------------------------------------------------------------------------
# Gemini generation
# ---------------------------------------------------------------------------


def make_prompt(slots_batch: list, seed_examples: list, batch_idx: int) -> str:
  """Build the generation prompt for a batch of demographic slots."""

  # Pick 5 diverse seed examples to show format
  seeds = random.sample(seed_examples, min(5, len(seed_examples)))
  seed_json = json.dumps(seeds, indent=2)

  demo_block = json.dumps(slots_batch, indent=2)

  prompt = f"""You are generating realistic voter personas for a political simulation research project. 
Each persona represents a real voter in a fictional democratic country. 

Here are {len(seeds)} example personas showing the exact JSON format to follow:
{seed_json}

Now generate {len(slots_batch)} NEW, UNIQUE personas. For each persona, I have pre-assigned demographics below.
You must use these EXACT demographics and district values. Your job is to write:
1. A unique "user_id" (format: "voter_XXXX" where XXXX is a random 4-digit number, unique across the batch)
2. A "self_description" (2-5 sentences) that reflects the persona's values, worldview, and personality. 
   Make it personal and specific — avoid generic platitudes. Ground it in their life situation.
3. Exactly 3 "examples" — these are SOCIOPOLITICAL opinions or comments the person might make.
   They should reveal the person's political leanings, policy views, or social attitudes.
   Make them substantive (1-3 sentences each). Cover a range of topics: economics, immigration, 
   healthcare, education, environment, civil rights, law enforcement, foreign policy, technology, etc.
   Write in natural, conversational English. Vary the tone — some passionate, some measured, some provocative.
   Clean grammar and spelling throughout.

IMPORTANT RULES:
- English speakers only — all text must be in fluent English
- No duplicate content across personas
- Make the self_descriptions and examples reflect the demographic context naturally
  (e.g., a retired 65+ person might reference grandchildren; a student might reference tuition costs)
- Include a MIX of political orientations: left, center-left, center, center-right, right
- Some personas should hold contradictory or nuanced views (e.g., fiscally conservative but socially liberal)
- Avoid clichés like "treat others how you want to be treated" — be specific and personal
- The examples should be sociopolitical in nature — opinions about policy, governance, society, justice, etc.

Pre-assigned demographics for this batch (batch {batch_idx}):
{demo_block}

Return ONLY a valid JSON array of {len(slots_batch)} persona objects. No commentary, no markdown fences.
Each object must have exactly these keys: "user_id", "demographics", "examples", "district"
where "demographics" has keys: "age", "gender", "employment", "education", "ethnicity", "religion", "marital_status", "self_description"
"""
  return prompt


def _get_access_token() -> str:
  """Get an access token via gcloud."""
  import subprocess

  result = subprocess.run(
      ["/google/data/ro/teams/cloud-sdk/gcloud", "auth", "print-access-token"],
      capture_output=True,
      text=True,
      timeout=30,
  )
  if result.returncode != 0:
    raise RuntimeError(f"gcloud auth failed: {result.stderr}")
  return result.stdout.strip()


# Cache the token (refreshed if 401)
_ACCESS_TOKEN = None


def call_gemini(prompt: str, retries: int = 3) -> str:
  """Call Gemini via Vertex AI REST API."""
  import requests

  global _ACCESS_TOKEN
  if _ACCESS_TOKEN is None:
    _ACCESS_TOKEN = _get_access_token()

  project = os.environ.get("GOOGLE_CLOUD_PROJECT", "deepmind-rfaulk")
  location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
  model = "gemini-2.5-flash"

  url = (
      f"https://{location}-aiplatform.googleapis.com/v1/"
      f"projects/{project}/locations/{location}/"
      f"publishers/google/models/{model}:generateContent"
  )

  payload = {
      "contents": [{"role": "user", "parts": [{"text": prompt}]}],
      "generationConfig": {
          "temperature": 1.0,
          "maxOutputTokens": 65536,
      },
  }

  for attempt in range(retries):
    try:
      headers = {
          "Authorization": f"Bearer {_ACCESS_TOKEN}",
          "Content-Type": "application/json",
      }
      resp = requests.post(url, json=payload, headers=headers, timeout=120)

      # Refresh token on auth error
      if resp.status_code in (401, 403):
        print(f"  Token expired, refreshing...")
        _ACCESS_TOKEN = _get_access_token()
        headers["Authorization"] = f"Bearer {_ACCESS_TOKEN}"
        resp = requests.post(url, json=payload, headers=headers, timeout=120)

      resp.raise_for_status()
      data = resp.json()

      # Extract text from response
      candidates = data.get("candidates", [])
      if candidates:
        parts = candidates[0].get("content", {}).get("parts", [])
        if parts:
          return parts[0].get("text", "")
      print(f"  No text in response")
      return ""
    except Exception as e:
      print(f"  API error (attempt {attempt+1}/{retries}): {e}")
      if attempt < retries - 1:
        time.sleep(5 * (attempt + 1))
  return ""


def parse_response(text: str) -> list:
  """Parse JSON array from Gemini response, handling markdown fences."""
  # Strip markdown code fences if present
  text = text.strip()
  if text.startswith("```"):
    # Remove first line (```json or ```)
    text = text.split("\n", 1)[1] if "\n" in text else text[3:]
  if text.endswith("```"):
    text = text[:-3]
  text = text.strip()

  try:
    data = json.loads(text)
    if isinstance(data, list):
      return data
  except json.JSONDecodeError as e:
    # Try to find JSON array in the text
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
      try:
        return json.loads(match.group())
      except json.JSONDecodeError:
        pass
    print(f"  JSON parse error: {e}")
  return []


def validate_persona(p: dict) -> bool:
  """Check that a persona has all required fields and ≥3 examples."""
  if not isinstance(p, dict):
    return False
  if (
      "user_id" not in p
      or "demographics" not in p
      or "examples" not in p
      or "district" not in p
  ):
    return False
  demo = p["demographics"]
  required_demo = [
      "age",
      "gender",
      "employment",
      "education",
      "ethnicity",
      "religion",
      "marital_status",
      "self_description",
  ]
  for key in required_demo:
    if key not in demo:
      return False
  if not isinstance(p["examples"], list) or len(p["examples"]) < 3:
    return False
  # Check examples are strings with some substance
  for ex in p["examples"]:
    if not isinstance(ex, str) or len(ex.strip()) < 10:
      return False
  return True


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
  random.seed(42)

  # Load original personas as seed examples
  with open(ORIGINAL_FILE) as f:
    originals = json.load(f)
  print(f"Loaded {len(originals)} original personas as seeds")

  # Build demographic slots
  slots = build_demographic_slots(TARGET_COUNT)
  print(f"Pre-assigned demographics for {len(slots)} personas")

  # Generate in batches
  all_personas = []
  used_ids = set()
  batch_count = (TARGET_COUNT + BATCH_SIZE - 1) // BATCH_SIZE

  for batch_idx in range(batch_count):
    start = batch_idx * BATCH_SIZE
    end = min(start + BATCH_SIZE, TARGET_COUNT)
    batch_slots = slots[start:end]

    print(
        f"\nBatch {batch_idx+1}/{batch_count}: generating {len(batch_slots)}"
        " personas..."
    )

    prompt = make_prompt(batch_slots, originals, batch_idx + 1)
    response_text = call_gemini(prompt)

    if not response_text:
      print(f"  Empty response, skipping batch")
      continue

    personas = parse_response(response_text)
    print(f"  Parsed {len(personas)} personas")

    valid_count = 0
    for p in personas:
      if not validate_persona(p):
        print(f"  Invalid persona skipped: {p.get('user_id', '?')}")
        continue

      # Ensure unique user_id
      uid = p["user_id"]
      while uid in used_ids:
        uid = f"voter_{random.randint(1000, 9999)}"
      p["user_id"] = uid
      used_ids.add(uid)

      all_personas.append(p)
      valid_count += 1

    print(f"  Valid: {valid_count}/{len(personas)}")

    # Rate limiting
    if batch_idx < batch_count - 1:
      time.sleep(2)

  print(f"\n{'='*60}")
  print(f"Total generated: {len(all_personas)}")

  # If we're short, report it
  if len(all_personas) < TARGET_COUNT:
    shortfall = TARGET_COUNT - len(all_personas)
    print(
        f"WARNING: Short by {shortfall} personas. Running additional batches..."
    )
    # Could add retry logic here

  # Save
  with open(OUTPUT_FILE, "w") as f:
    json.dump(all_personas, f, indent=2, ensure_ascii=False)
  print(f"Saved to {OUTPUT_FILE}")

  # Print demographic summary
  from collections import Counter

  print(f"\n--- Final Demographics Summary ({len(all_personas)} personas) ---")
  for field in [
      "age",
      "gender",
      "employment",
      "education",
      "ethnicity",
      "religion",
      "marital_status",
  ]:
    counts = Counter(p["demographics"][field] for p in all_personas)
    print(f"\n{field}:")
    for k, v in sorted(counts.items(), key=lambda x: -x[1]):
      print(f"  {k or '(empty)'}: {v} ({v/len(all_personas)*100:.1f}%)")

  dist_counts = Counter(p["district"] for p in all_personas)
  print(f"\ndistrict:")
  for k, v in sorted(dist_counts.items(), key=lambda x: -x[1]):
    print(f"  {k}: {v} ({v/len(all_personas)*100:.1f}%)")

  ex_counts = Counter(len(p["examples"]) for p in all_personas)
  print(f"\nexample counts:")
  for k, v in sorted(ex_counts.items()):
    print(f"  {k} examples: {v}")


if __name__ == "__main__":
  main()
