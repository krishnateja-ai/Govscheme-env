# GovScheme-Env 🏛️

> **OpenEnv environment — Government Scheme Eligibility Matching**
> Real-world task: Help Indian citizens identify, rank, and apply for government welfare schemes.

[![OpenEnv](https://img.shields.io/badge/OpenEnv-compatible-blue)](https://github.com/meta-pytorch/OpenEnv)

---

## What this environment does

Millions of Indians miss welfare schemes they legally qualify for — not because they don't qualify, but because the system is too complex to navigate. A government officer manually does three things for each citizen:

1. **Reads the citizen's profile** and checks which of hundreds of schemes apply
2. **Ranks those schemes** by which gives the most value
3. **Fills out the application form** accurately using the citizen's documents

This environment trains and evaluates AI agents to do all three of these tasks — deterministically, at scale, with clear scoring.

---

## The 3 Tasks

| Task | Difficulty | What the agent does | Grader |
|------|-----------|---------------------|--------|
| `scheme_identification` | Easy | List all schemes the citizen qualifies for | F-beta score (recall-weighted) |
| `scheme_ranking` | Medium | Rank eligible schemes by annual INR benefit with justification | NDCG@5 + benefit accuracy + reasoning quality |
| `form_filling` | Hard | Fill complete application form from profile, no hallucination | Field accuracy + format validity + completeness − hallucination penalty |

---

## Project structure (understand this before touching any file)

```
govscheme_env/              ← root of the project
│
├── inference.py            ← MANDATORY: hackathon baseline script (you run this)
├── openenv.yaml            ← MANDATORY: OpenEnv manifest (openenv validate reads this)
├── pyproject.toml          ← makes the package pip-installable
├── README.md               ← this file
│
├── models.py               ← shared data contracts (Action, Observation, State)
├── client.py               ← HTTP client (inference.py uses this to talk to server)
├── __init__.py             ← package exports
│
└── server/                 ← everything that runs INSIDE Docker
    ├── Dockerfile          ← builds the server container
    ├── requirements.txt    ← server-only Python deps (fastapi, uvicorn, pydantic)
    ├── app.py              ← FastAPI server (exposes /reset /step /state /health)
    ├── govscheme_environment.py  ← GovSchemeEnvironment class (reset/step/state logic)
    ├── eligibility.py      ← deterministic eligibility rule engine (18 schemes)
    ├── graders.py          ← 3 task graders (F-beta, NDCG, form validation)
    ├── schemes.json        ← 18 Indian govt schemes with eligibility rules
    └── citizens.json       ← 10 diverse citizen profiles with verified gold labels
```

### Why this split?

The `server/` folder runs in Docker and talks to nobody except the client.
Everything outside `server/` runs on your machine and talks to the server over HTTP.

```
Your machine                          Docker container
──────────────────────────            ──────────────────────────────────
inference.py                          server/app.py
  └── uses client.py           HTTP   server/govscheme_environment.py
       └── GovSchemeEnv  ──────────►  server/eligibility.py
            └── GovSchemeAction        server/graders.py
                                       server/schemes.json
                                       server/citizens.json
```

---

## How to run locally (step by step)

### Step 1: Install Python dependencies on your machine

```bash
pip install fastapi uvicorn pydantic requests openai python-multipart
```

### Step 2: Start the server

```bash
# From inside the govscheme_env/ folder:
cd server
python -m uvicorn app:app --host 0.0.0.0 --port 7860 --reload
```

You should see:
```
INFO:     Uvicorn running on http://0.0.0.0:7860
```

Open http://localhost:7860/health in your browser — you should see `{"status":"ok"}`.

### Step 3: Test the server manually

```bash
# Reset for Task 1
curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" \
  -d '{"task_name": "scheme_identification", "citizen_id": "CIT_001", "seed": 42}'

# Take an action
curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "identify_schemes", "scheme_ids": ["PM_KISAN", "MGNREGA", "AYUSHMAN_BHARAT"]}'

# Check state
curl http://localhost:7860/state
```

### Step 4: Run the inference script (with a real LLM)

```bash
export HF_TOKEN=your_huggingface_token
export MODEL_NAME=Qwen/Qwen2.5-72B-Instruct
export API_BASE_URL=https://router.huggingface.co/v1
export ENV_URL=http://localhost:7860   # or your HF Space URL

python inference.py
```

---

## How to run with Docker

```bash
# Build the Docker image
docker build -t govscheme-env -f server/Dockerfile .

# Run it
docker run -p 7860:7860 govscheme-env

# Verify it's alive
curl http://localhost:7860/health
```

---

## Observation space (what the agent sees)

```json
{
  "citizen_profile": {
    "citizen_id": "CIT_001",
    "name": "Ramesh Kumar Yadav",
    "age": 38,
    "gender": "Male",
    "state": "Uttar Pradesh",
    "district": "Varanasi",
    "village": "Rampur",
    "area_type": "rural",
    "occupation": "farmer",
    "caste": "OBC",
    "annual_income_inr": 85000,
    "annual_family_income_inr": 85000,
    "land_ownership_acres": 2.3,
    "house_type": "kachha",
    "has_lpg": false,
    "has_bank_account": true,
    "has_aadhaar": true,
    "is_govt_employee": false,
    "is_income_taxpayer": false,
    "aadhaar_number": "234567890123",
    "bank_account_number": "32145678901",
    "ifsc_code": "SBIN0001234",
    "mobile_number": "9876543210"
  },
  "task_name": "scheme_identification",
  "task_description": "...",
  "available_schemes": [...],
  "identified_schemes": null,
  "target_scheme_id": null,
  "form_template": null,
  "step_number": 1,
  "max_steps": 3,
  "cumulative_reward": 0.0,
  "reward": 0.0,
  "done": false
}
```

## Action space (what the agent sends back)

**Task 1:**
```json
{"action_type": "identify_schemes", "scheme_ids": ["PM_KISAN", "MGNREGA", "AYUSHMAN_BHARAT"]}
```

**Task 2:**
```json
{
  "action_type": "rank_schemes",
  "ranked_schemes": [
    {"scheme_id": "AYUSHMAN_BHARAT", "rank": 1, "reason": "₹5 lakh health cover — highest value for OBC farmer", "benefit_inr": 500000},
    {"scheme_id": "PM_KISAN", "rank": 2, "reason": "₹6,000/year guaranteed cash transfer for farmers", "benefit_inr": 6000}
  ]
}
```

**Task 3:**
```json
{
  "action_type": "fill_form",
  "form_data": {
    "applicant_name": "Ramesh Kumar Yadav",
    "aadhaar_number": "234567890123",
    "date_of_birth": "15/03/1986",
    "gender": "Male",
    "state": "Uttar Pradesh",
    "district": "Varanasi",
    "bank_account_number": "32145678901",
    "ifsc_code": "SBIN0001234",
    "mobile_number": "9876543210",
    "category": "OBC"
  }
}
```

---

## Reward function

### Task 1 — Scheme Identification
Recall-weighted F-beta (β=1.5). Missing an eligible scheme is penalised more than including a wrong one.

### Task 2 — Scheme Ranking
- 50%: NDCG@5 (ranking order quality)
- 30%: Benefit amount accuracy (must cite correct INR, within 10%)
- 20%: Reasoning quality (mentions citizen's caste/income/occupation)

### Task 3 — Form Filling
- 40%: Field accuracy (value matches citizen profile)
- 30%: Format validity (Aadhaar/IFSC/mobile/date regex checks)
- 20%: Completeness (all required fields present)
- −10 to −30%: Hallucination penalty (fabricated values with correct format)

---

## Scheme coverage (18 central government schemes)

| ID | Name | Annual Benefit |
|----|------|---------------|
| PM_KISAN | PM-KISAN | ₹6,000 |
| AYUSHMAN_BHARAT | Ayushman Bharat PM-JAY | ₹5,00,000 |
| MGNREGA | MGNREGA | ₹57,200 |
| PM_AWAS_GRAMIN | PMAY-G | ₹1,20,000 |
| NSP_SC_SCHOLARSHIP | SC Post-Matric Scholarship | ₹15,000 |
| NSP_OBC_SCHOLARSHIP | OBC Post-Matric Scholarship | ₹12,000 |
| SUKANYA_SAMRIDDHI | Sukanya Samriddhi Yojana | 8.2% interest |
| UJJWALA_YOJANA | PMUY (LPG) | ₹1,600 |
| KISAN_CREDIT_CARD | KCC | 4% credit |
| PMSBY | Suraksha Bima | ₹2,00,000 |
| PMJJBY | Jeevan Jyoti Bima | ₹2,00,000 |
| APY | Atal Pension Yojana | ₹60,000 |
| PMEGP | PMEGP | ₹6,25,000 subsidy |
| PMFBY | Fasal Bima Yojana | crop insurance |
| NSP_MINORITY_SCHOLARSHIP | Minority Pre-Matric | ₹10,000 |
| STAND_UP_INDIA | Stand-Up India | ₹10L–1Cr loan |
| WEAVERS_MUDRA | Weavers MUDRA | ₹10L loan |
| PMVVY | PM Vaya Vandana | ₹96,000 |

---

## Baseline scores (expected from Qwen/Qwen2.5-72B-Instruct)

| Task | Difficulty | Expected Score |
|------|-----------|---------------|
| scheme_identification | Easy | ~0.75 |
| scheme_ranking | Medium | ~0.60 |
| form_filling | Hard | ~0.45 |
| **Average** | | **~0.60** |

---

## License
MIT — built for Meta × Hugging Face OpenEnv Hackathon 2026.
