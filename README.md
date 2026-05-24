# VoteSim: Simulate Voting & Deliberation with LLMs

VoteSim is a simulation framework that models the full lifecycle of representative democracy: from voter opinion formation through electoral seat allocation, coalition formation, parliamentary deliberation, and comparative policy evaluation — all driven by large language models (LLMs) and grounded in real human personas from the [PRISM dataset](https://arxiv.org/abs/2404.16019).

The goal is to study how different electoral systems translate voter preferences into legislative outcomes, and to quantify which systems produce policies that best reflect the preferences of the electorate.

## Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                        VoteSim Pipeline                         │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Region & District Generation                                │
│     └─ LLM generates a geographically-grounded region with      │
│        N districts, each with socioeconomic attributes          │
│                                                                 │
│  2. Voter Sampling (PRISM)                                      │
│     └─ K voters sampled with demographics, values,              │
│        and past statements; assigned to districts               │
│                                                                 │
│  3. Voter Opinion Survey                                        │
│     └─ Each voter responds to a social-issue prompt             │
│        in character, conditioned on their persona               │
│                                                                 │
│  4. Party Policy Generation                                     │
│     └─ Each party produces a policy response conditioned        │
│        on its platform, voter sentiment, and region             │
│                                                                 │
│  5. Voter Ranking of Parties                                    │
│     └─ Each voter ranks parties by policy alignment             │
│        (randomised presentation order per voter)                │
│                                                                 │
│  6. Seat Allocation (per voting system)                         │
│     └─ Ballots → district-level seat allocation via             │
│        SNTV, FPTP, AV, TRS, D'Hondt, Sainte-Laguë, or STV       │
│                                                                 │
│  7. Coalition Formation & Parliamentary Deliberation            │
│     └─ Politically aligned coalition formed; coalition drafts   │
│        a bill; all members vote; re-draft on failure            │
│                                                                 │
│  8. Comparative Policy Ranking & Scoring                        │
│     └─ Voters rank AND score (1.0–5.0 Likert) the bills         │
│        produced under each voting system + baselines            │
│                                                                 │
│  9. Results Persistence (JSON)                                  │
│     └─ Policies, rankings, scores, voter data saved per model   │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Electoral Systems

VoteSim implements seven electoral systems in two families:

### Majoritarian Systems

| System | Key | Description |
|---|---|---|
| **Party Block Vote (SNTV)** | `sntv` | Winner-takes-all per district; the party with the most first-choice votes receives **all** seats in that district |
| **First Past The Post** | `fptp` | Like SNTV but each district is fixed at exactly **1 seat** regardless of population, amplifying geographic representation |
| **Instant-Runoff Voting (IRV)** | `alternative_vote` | Iterative elimination: the candidate with the fewest votes is eliminated and their ballots redistributed to next preferences until one candidate holds an absolute majority. One seat per district |
| **Two-Round System** | `trs` | If a party wins >50% in round 1, it wins outright. Otherwise, a runoff between the top two determines the winner |

### Proportional Systems

| System | Key | Description |
|---|---|---|
| **D'Hondt** | `dhondt` | Highest-averages method with divisors 1, 2, 3, … Seats allocated one at a time to the party with the highest quotient `votes / (seats_won + 1)`. Tends to favour larger parties |
| **Sainte-Laguë** | `sainte_lague` | Highest-averages method with **odd** divisors 1, 3, 5, 7, … (`2 × seats_won + 1`). Produces more proportional results; least biased toward large parties |
| **Single Transferable Vote** | `stv` | Droop-quota method with fractional surplus transfer and elimination rounds. Rewards cross-party appeal through preference transfers |

All proportional methods operate per-district. Districts are assigned 4–15 seats based on relative population via linear interpolation.

## Deliberation Phase

The parliamentary deliberation phase (Phase 7) simulates a structured legislative debate among seated members. The process works as follows:

### Coalition Formation

After seat allocation, VoteSim forms a governing coalition based on **political alignment**. The largest party seeks to join with the most ideologically compatible smaller parties (based on proximity on the spectrum: Left — Green — Social Democrat — Liberal — Conservative — Populist). The second-largest party enters opposition. Partners are added until the coalition holds >50% of seats.

### Bill Lifecycle (per round)

```
┌──────────────────────────────────────────────────────────┐
│  1. DRAFT — Coalition drafts a 5-8 point bill            │
│     • The LLM acts as a legislative advisor              │
│     • Bill reflects coalition partners proportionally     │
│     • Larger coalition partners have more influence       │
│     • Failed prior rounds inform the next draft           │
│                                                          │
│  2. VOTE — All seated members vote yes/no                │
│     • Members consider BOTH party line and constituent   │
│       interests (informed by district voter responses)   │
│     • Members may vote against their party               │
│     • Bill passes with majority (>50% of seats)          │
│                                                          │
│  3. RE-DRAFT — If bill fails, coalition re-drafts        │
│     • New draft incorporates failed bill and voting      │
│       record from previous round                         │
│     • Up to max_rounds attempts (default: 5)             │
└──────────────────────────────────────────────────────────┘
```

### Key Design Decisions

- **Constituency awareness**: Each member is assigned to a district and sees a summary of their constituents' opinions, enabling cross-party voting when the party line conflicts with local sentiment.
- **Iterative refinement**: Failed bills inform subsequent drafts, pushing the coalition toward policies that can attract broader support.
- **Coalition proportionality**: The drafting prompt explicitly instructs the LLM to weight coalition partners by their seat share, preventing the largest party from dominating.
- **Political alignment**: Coalition partners are chosen by ideological proximity, not just seat count, producing more coherent legislative programs.

## Baselines

To isolate the value of electoral mechanics and deliberation, VoteSim generates two baseline bills alongside each deliberated bill:

| Baseline | Key | What it receives | What it skips |
|----------|-----|-----------------|---------------|
| **Baseline** | `baseline` | Issue prompt only | Party policies, voter data, seat allocation, deliberation |
| **Baseline (Informed)** | `baseline_informed` | Issue prompt + all party policies | Seat allocation, coalition formation, deliberation |

### Baseline

The model acts as a "nonpartisan policy analyst" given only the issue text. This measures what an LLM produces with **zero democratic input** — no voter preferences, no party platforms, no electoral representation. It represents the "just ask the AI" approach to policy.

### Baseline (Informed)

The model receives all party policy positions but no information about which parties won or how seats were allocated. This measures what the LLM produces when it can **see the political landscape** but has no democratic mandate to weight any party's position over another. It must synthesise across all parties equally.

### Interpretation

Comparing deliberated bills against baselines answers:
- **Deliberated > Baseline**: Electoral representation and negotiation add value
- **Baseline Informed > Deliberated**: The LLM's own synthesis is preferred over the messy compromise of democratic deliberation
- **Baseline > Both**: Voters prefer a clean, unopinionated policy over both partisan and synthesised approaches

Early results show that **baselines consistently underperform deliberative systems on divisive issues** (e.g., Baseline ≈ 3.0 vs. top systems ≈ 4.2 on a 1–5 Likert scale), confirming that electoral mechanics and deliberation produce genuinely better-aligned policies when voters disagree.

## Question Datasets

Questions are framed as **open-ended deliberative prompts** (e.g., "How should firearms be regulated?") rather than prescriptive statements, allowing LLM agents to arrive at their own policy positions. The original prescriptive statements are preserved in `political-issues.csv`.

| Dataset | Key | Questions | Description |
|---------|-----|-----------|-------------|
| **Divisive 12** | `divisive-12` | 12 | Maximally polarising issues (abortion, guns, immigration, etc.) selected to ensure meaningful disagreement across the political spectrum |
| **Diverse 12** | `diverse-12` | 12 | Curated questions spanning economic, social, governance, technology, environment, health, housing, education, immigration |
| **Diverse 20** | `diverse-20` | 20 | Broader version of diverse-12 |
| **Harm 12** | `harm-12` | 12 | Contentious topics with potential for harm (surveillance, social credit, autonomous weapons, etc.) |
| **Harm 20** | `harm-20` | 20 | Broader version of harm-12 |
| **Full Dataset** | `legacy` | 2500 | Complete political-questions dataset with open-ended reformulations |

## Project Structure

```
VoteSim/
├── dataset/
│   ├── party/                      # Party platform JSON files
│   │   ├── conservative.json
│   │   ├── green.json
│   │   ├── liberal.json
│   │   ├── libertarian.json
│   │   ├── nationalist.json
│   │   ├── populist.json
│   │   └── socialist.json
│   ├── political_questions/        # Question datasets
│   │   ├── divisive-12.csv         # 12 maximally polarising issues
│   │   ├── diverse-12.csv          # 12 questions across policy axes
│   │   ├── diverse-20.csv          # 20 questions, broader coverage
│   │   ├── harm-12.csv             # 12 contentious/harm-adjacent topics
│   │   ├── harm-20.csv             # 20 contentious/harm-adjacent topics
│   │   ├── political-questions.csv # Full dataset (2500 open-ended questions)
│   │   └── political-issues.csv    # Original prescriptive statements
│   ├── personas/                   # Cached generated personas
│   ├── prism/                      # PRISM persona dataset
│   │   ├── survey.jsonl            # Demographics & self-descriptions
│   │   └── conversations.jsonl     # Past statements for persona grounding
│   └── regions/                    # Cached generated regions
├── simulation/
│   ├── main.py                     # Hydra entry point
│   ├── run.py                      # Pipeline & query dispatch
│   ├── pipeline.py                 # Orchestration (issue & platform modes)
│   ├── survey.py                   # Voter/party survey & ranking generation
│   ├── voting.py                   # 6 electoral system implementations
│   ├── deliberate.py               # Parliamentary deliberation simulation
│   ├── policy_ranking.py           # Comparative ranking & Likert scoring
│   ├── policy_generator.py         # Party platform loading & policy gen
│   ├── political_sampler.py        # Question dataset sampling
│   ├── district_generator.py       # Region & district generation
│   ├── prism_sampler.py            # PRISM voter persona sampling
│   ├── conf/
│   │   └── config.yaml             # Default configuration
│   └── launcher.sh                 # Example SLURM/HPC launch script
├── pathfinder/                     # LLM abstraction layer
├── requirements.txt
├── setup.sh                        # Environment setup script
└── README.md
```

## Configuration

All configuration is managed through a single YAML file at `simulation/conf/config.yaml` using [Hydra](https://hydra.cc/). Settings can be overridden on the command line.

### Full Configuration Reference

```yaml
# Top-level mode: "pipeline" (full simulation) or "query" (single persona query)
mode: pipeline

# LLM settings
llm:
  path: openrouter-google/gemma-4-31b-it   # Model path or API identifier
  is_api: true                              # true for API models, false for local
  backend: transformers                     # "transformers" or "vllm"
  temperature: 0.0                          # Sampling temperature

# Region generation (optional — omit to skip district grounding)
region:
  description: "Southern Ontario Canada"    # Natural-language region description
  num_districts: 5                          # Number of electoral districts
  cache_path: "dataset/regions"             # Cache generated region JSON here

# Pipeline settings
pipeline:
  voting_mode: "issue"            # "issue" (default) or "platform"
  num_voters: 100                 # Number of PRISM voters to sample
  max_workers: 5                  # Max concurrent LLM calls
  topic: "social"                 # Question topic filter
  parties:                        # Which ideologies to include
    - liberal
    - conservative
    - socialist
  voting_system: "fptp"           # Primary system (single election)
  voting_systems:                 # Systems for comparative ranking
    - fptp
    - smdp
    - alternative_vote
    - dhondt
    - hare
    - sainte_lague
  max_rank: 3                     # Number of parties each voter ranks
  results_dir: "results"          # Output directory for JSON results
  deliberation:
    enabled: true                 # Toggle parliamentary deliberation
    max_rounds: 3                 # Max bill consideration attempts

# Question dataset selection
political_questions:
  dataset: "divisive-12"          # divisive-12, diverse-12, diverse-20, harm-12, harm-20, legacy
  # question_indices: [0, 3, 7]  # Optional: select specific questions by index
```

### Key Configuration Decisions

| Parameter | Effect |
|---|---|
| `pipeline.voting_mode` | `"issue"` = parties condition on voter opinions per-issue (default); `"platform"` = voters rank parties once by platform, fixed seats across all issues |
| `pipeline.parties` | Controls which of the 7 available ideologies participate |
| `pipeline.voting_systems` | Which electoral systems to compare in the ranking phase |
| `pipeline.deliberation.enabled` | Whether parliaments actually deliberate or just pass baseline policies |
| `political_questions.dataset` | Which question set to use — see [Question Datasets](#question-datasets) |
| `region.description` | Set to any real-world region; the LLM generates plausible districts |

## Experiments

### 1. Comparative Electoral System Analysis (default)

**Research question:** *Which electoral system produces policies that best match voter preferences?*

Run the full pipeline with all 6 voting systems and compare how voters rank the resulting legislation:

```bash
python3 -m simulation.main \
  pipeline.voting_systems='[fptp,smdp,alternative_vote,dhondt,hare,sainte_lague]'
```

Results are saved to `results/<model_name>.json` with per-voter rankings and Likert scores (1.0–5.0) for each system.

### 2. Platform-Only Mode (Representative Democracy)

**Research question:** *What happens when party policies are static ideological platforms rather than being conditioned on voter feedback per issue?*

In this mode, voters rank parties **once** based on general platform summaries (like a real election). The same seat allocation is then used for deliberation across all issues.

```bash
python3 -m simulation.main pipeline.voting_mode=platform
```

This removes the dependence of party policy on per-issue voter preferences, modelling how representative democracies function — voters elect based on broad ideology, then the elected government legislates on specific issues.

### 3. Ideology Composition Experiments

**Research question:** *How do different party mixes affect policy outcomes?*

Vary which parties participate:

```bash
# Two-party system
python3 -m simulation.main pipeline.parties='[liberal,conservative]'

# Multi-party with fringe ideologies
python3 -m simulation.main \
  pipeline.parties='[liberal,conservative,socialist,green,libertarian,populist,nationalist]'
```

### 4. Regional Variation

**Research question:** *How do different regional demographics affect election outcomes?*

```bash
# Urban region
python3 -m simulation.main region.description="Greater London, United Kingdom"

# Rural region
python3 -m simulation.main region.description="Rural Saskatchewan, Canada"

# Diverse developing region
python3 -m simulation.main region.description="Western Cape, South Africa"
```

### 5. Scale Experiments

**Research question:** *How do outcomes change with electorate size and district granularity?*

```bash
python3 -m simulation.main \
  pipeline.num_voters=500 \
  region.num_districts=10
```

### 6. Deliberation Ablation

**Research question:** *Does parliamentary deliberation improve policy alignment with voters, or does the initial bill suffice?*

The pipeline automatically generates two baseline bills alongside each deliberated bill. Compare these against the deliberated outcomes in the results JSON.

To disable deliberation entirely:

```bash
python3 -m simulation.main pipeline.deliberation.enabled=false
```

### 7. Model Comparison

**Research question:** *Do different LLMs produce systematically different electoral outcomes?*

Run the same configuration with different models. Results are persisted in separate files per model:

```bash
# Run with Gemma
python3 -m simulation.main llm.path=openrouter-google/gemma-4-31b-it

# Run with Llama
python3 -m simulation.main llm.path=openrouter-meta-llama/llama-3.3-70b-instruct

# Run with a local model
python3 -m simulation.main llm.path=Qwen/Qwen3-4B-Thinking-2507 llm.is_api=false
```

### 8. Question Set Experiments

**Research question:** *Do voting system preferences vary by issue divisiveness?*

```bash
# Maximally divisive issues
python3 -m simulation.main political_questions.dataset=divisive-12

# Diverse policy topics
python3 -m simulation.main political_questions.dataset=diverse-20

# Contentious/harm-adjacent topics
python3 -m simulation.main political_questions.dataset=harm-12

# Specific questions only
python3 -m simulation.main political_questions.question_indices='[0,5,11]'
```

## How to Run

### Prerequisites

- Python 3.11+
- Access to an LLM (API-based via OpenRouter, or local via Hugging Face)
- The [PRISM dataset](https://arxiv.org/abs/2404.16019) placed in `dataset/prism/`

### Setup

```bash
# Clone and enter the project
cd VoteSim

# Run the setup script (creates venv, installs deps)
bash setup.sh

# Activate the environment
source .venv/bin/activate
```

For API-based models, set the appropriate API key:

```bash
export OPENROUTER_API_KEY="your-key-here"
# or for direct provider access:
export OPENAI_API_KEY="your-key-here"
```

### Running the Simulation

```bash
# Default pipeline (issue mode, all defaults from config.yaml)
python3 -m simulation.main

# Override any config on the command line (Hydra syntax)
python3 -m simulation.main \
  pipeline.num_voters=50 \
  pipeline.voting_mode=platform \
  llm.temperature=0.7

# Single persona query (for debugging / exploration)
python3 -m simulation.main mode=query
```

### HPC / SLURM

An example launch script is provided at `simulation/launcher.sh`. Adapt the module loads and paths to your cluster environment.

## Party Platforms

Six political ideologies are pre-configured with detailed policy positions across economics, social policy, governance, environment, and security:

| Ideology | Party Name | Brief Description |
|---|---|---|
| `left` | Left Alliance | Democratic socialism, wealth redistribution, anti-imperialist |
| `green` | Green Ecology Party | Environmental sustainability, climate justice, community governance |
| `socialist` | Social Democrat Party | Reformist, universal public services, progressive taxation |
| `liberal` | Liberal Democratic Alliance | Pragmatic centre, evidence-based policy, regulated markets |
| `conservative` | Conservative Alliance | Traditional values, fiscal restraint, strong institutions |
| `populist` | People's Patriot Movement | National-populist, nativist immigration, cultural protectionism |

## Output Format

Results are saved as JSON files in `results/` (one per model). Each file has the schema:

```json
{
  "<social issue text>": {
    "policies": {
      "<system_name>": "<adopted bill text>" 
    },
    "parties": {
      "<ideology>": {
        "position_statement": "...",
        "key_proposals": ["..."]
      }
    },
    "voters": {
      "<user_id>": {
        "response": "...",
        "demographics": { "age": 34, "gender": "Female", ... },
        "self_description": "...",
        "examples": ["...", "..."]
      }
    },
    "ballots": {
      "<user_id>": {
        "district": "Ironforge Centre",
        "ranking": ["liberal", "socialist", "conservative"]
      }
    },
    "rankings": {
      "<user_id>": ["system_a", "system_b", ...]
    },
    "scores": {
      "<user_id>": {
        "system_a": 4.2,
        "system_b": 2.0
      }
    }
  }
}
```

- **ballots**: Per-voter party rankings from phase 5, including district assignment
- **rankings**: Per-voter ordinal ranking of voting systems from phase 8 (best → worst)
- **scores**: Per-voter Likert scores from phase 8 (1.0 = no match, 5.0 = perfect match, 0.1 increments)

## Voter Personas

Voters are sampled from the PRISM dataset, which provides:

- **Demographics**: age, gender, education, employment, ethnicity, religion, location, marital status
- **Self-description**: a first-person values/worldview statement
- **Examples**: 5–10 past statements (declarative opinions, not questions) that ground the voter's "voice"

Each voter is deterministically assigned to an electoral district based on their user ID, ensuring consistent placement across simulation phases.

## Ordering Bias Mitigation

To prevent positional bias in LLM responses, the order in which parties (phase 5) and voting system policies (phase 8) are presented is **randomised per voter per context**. The same voter sees different orderings across:

- Different pipeline phases (party ranking vs. bill ranking)
- Different social issues
- Different voters always see different orderings

Ordering is deterministic (seeded by `md5(user_id + context_salt)`) for reproducibility.

------------------------------------------------
ASCII Art based on content from https://stock.adobe.com/ca/
https://www.asciiart.eu/image-to-ascii
