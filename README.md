---
title: DataEngEnv
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
license: mit
tags:
  - openenv
  - ml
  - data-engineering
  - debugging
---

# DataEngEnv — ML Data Pipeline Debugger

![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-success)
![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow)

> *It's 2 AM. The model ships to production in 6 hours. The pipeline is broken.*
> *Can your AI agent fix it in time?*

---

## What is DataEngEnv?

DataEngEnv is a real-world OpenEnv environment where an AI agent plays the role of a **frantic Data Engineer debugging a broken ML pipeline before a model goes to production.**

Every data team has been here: a schema change breaks the feature script, dirty data crashes the scaler, or a subtle data leakage bug silently inflates model accuracy to 98% — and nobody notices until it's too late.

DataEngEnv turns these real failure modes into a structured learning environment. Agents must **inspect data, read error logs, patch broken code, and verify fixes** — exactly the workflow a human engineer would follow. No games. No toys. Just the debugging tasks that happen every day at every ML company.

**Why this matters:** Training agents to debug data pipelines has direct real-world value. A capable pipeline-debugging agent could save ML teams hours of manual work on every model release cycle.

---

## Environment Overview

The agent receives a **broken Python ML pipeline** and must fix it before running out of steps (max 20). At each step the agent chooses one action, receives an observation, and earns a partial reward for meaningful progress.

### Observation Space

| Field | Type | Description |
|---|---|---|
| `task_id` | `int` | Which task is currently running (1, 2, or 3) |
| `step_number` | `int` | Current step (max 20) |
| `script_content` | `str` | The current state of the pipeline script |
| `last_run_output` | `str` | stdout from the last `run_script` action |
| `last_run_error` | `str \| null` | stderr / exception if the script crashed |
| `data_preview` | `str` | DataFrame shape, columns, nulls, sample rows, stats |
| `schema_info` | `dict` | Column names, dtypes, null counts |
| `done` | `bool` | True when submitted or steps exhausted |

### Action Space

| Action | Payload | What it does |
|---|---|---|
| `inspect_data` | `{}` | Reveals full dataset preview — columns, nulls, outliers, sample rows |
| `check_schema` | `{}` | Compares expected vs actual column names |
| `run_script` | `{}` | Executes current script in a 10s sandboxed subprocess |
| `edit_script` | `{"old": str, "new": str}` | Patches the script with a string replacement |
| `submit` | `{}` | Submits the fix to the grader — ends the episode |

---

## The 3 Tasks

### Task 1 — Column Rename Bug 🟢 Easy

**The scenario:** A source CSV was updated and the column `age_years` was renamed to `age`. The pipeline script still references `age_years` and crashes instantly with a `KeyError`.

**What the agent must do:** Read the error log, inspect the data schema, identify the column mismatch, patch the one broken line.

**Why it's easy:** The fix is surgical — one string replacement. A capable agent should solve this in 3–5 steps.

**Grader:** Did the script run without error and print an accuracy value?

---

### Task 2 — Dirty Data 🟡 Medium

**The scenario:** The training dataset has ~15% NaN values across 3 columns AND extreme salary outliers (`9,999,999.0`). The `StandardScaler` crashes with `ValueError: Input contains NaN` before training even starts.

**What the agent must do:** Detect the nulls and outliers via `inspect_data`, write cleaning logic (`dropna`/`fillna` for NaNs, `clip`/`quantile` for outliers), and get the training script to complete successfully.

**Why it's medium:** There's no single-line fix. The agent must reason about the data, choose a cleaning strategy, and implement it correctly.

**Grader:** Did training complete without crashing and print an accuracy score?

---

### Task 3 — Data Leakage 🔴 Hard

**The scenario:** The pipeline runs perfectly. No errors. It prints `Accuracy: 0.98`. But the model is cheating — `StandardScaler.fit_transform()` was called on the **entire dataset** before `train_test_split()`. The scaler has seen the test data. The 0.98 accuracy is a lie.

**What the agent must do:** Notice that suspiciously high accuracy, diagnose the data leakage (scaler fit before split), fix the pipeline so the scaler only sees training data, and achieve ≥ 0.75 accuracy on a **server-side held-out test set** the agent never sees.

**Why it's hard:** There's no error message. No crash. The agent must reason about *correctness*, not just syntax. This is the hardest class of ML bug to catch.

**Grader:** Does the fixed script move `scaler.fit()` after `train_test_split()`? Does it achieve ≥ 0.75 accuracy on the held-out set?

---

## Reward Function

DataEngEnv provides **partial credit throughout the episode** — not just a binary win/lose at the end.

| Task | Action | Reward |
|---|---|---|
| Task 1 | First `inspect_data` or `check_schema` | +0.1 |
| Task 1 | `edit_script` removes `age_years` | +0.3 |
| Task 1 | `run_script` with no error | +0.2 |
| Task 2 | `inspect_data` reveals null counts | +0.15 |
| Task 2 | `inspect_data` reveals outliers | +0.15 |
| Task 2 | `edit_script` adds `dropna`/`fillna` | +0.3 |
| Task 2 | `edit_script` adds `clip`/`quantile` | +0.2 |
| Task 3 | Any `inspect_data` action | +0.1 |
| Task 3 | `run_script` (sees suspicious accuracy) | +0.2 |
| Task 3 | `edit_script` moves `fit` after `split` | +0.3 |
| All | `submit` with correct fix | **grader score** |
| Task 3 | `submit` without editing anything | -0.2 |

Rewards are cumulative and capped at 1.0 before the terminal step.
The terminal reward is always the grader score (0.0–1.0).

---

## Baseline Scores

*Evaluated using `llama-3.1-8b-instant` via Groq API*

| Task | Difficulty | Score | Steps |
|------|-----------|-------|-------|
| Task 1 — Column Rename Bug | 🟢 Easy | **1.00** | 3 |
| Task 2 — Dirty Data | 🟡 Medium | **1.00** | 15 |
| Task 3 — Data Leakage | 🔴 Hard | **1.00** | 15 |
| **Average** | — | **1.00** | — |

---

## Setup & Usage

### Run locally
```bash
git clone https://huggingface.co/spaces/CoBeDigger/DataEngEnv
cd DataEngEnv
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### Run with Docker
```bash
docker build -t dataengenv .
docker run -p 7860:7860 dataengenv
```

### Environment variables
```bash
OPENAI_API_KEY=your_key    # for baseline script (OpenAI)
GROQ_API_KEY=your_key      # for baseline script (Groq alternative)
DATAENGENV_URL=http://localhost:8000  # baseline target URL
```

---

## API Reference
```bash
# Health check
curl https://cobedigger-dataengenv.hf.space/health

# List all tasks
curl https://cobedigger-dataengenv.hf.space/tasks

# Start an episode
curl -X POST https://cobedigger-dataengenv.hf.space/reset \
  -H "Content-Type: application/json" -d '{"task_id": 1}'

# Take a step
curl -X POST https://cobedigger-dataengenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_data", "payload": {}}'

# Get current state
curl https://cobedigger-dataengenv.hf.space/state

# Grade current episode
curl -X POST https://cobedigger-dataengenv.hf.space/grader \
  -H "Content-Type: application/json" -d '{"task_id": 1}'

# Run full baseline
curl https://cobedigger-dataengenv.hf.space/baseline
```

---

## Project Structure
```
dataengenv/
├── app/
│   ├── main.py              ← FastAPI app, all endpoints
│   ├── models.py            ← Pydantic v2 typed models
│   ├── environment.py       ← Core env: reset(), step(), state()
│   ├── reward.py            ← Partial reward shaping engine
│   ├── tasks/
│   │   ├── task1_easy.py    ← Column rename scenario
│   │   ├── task2_medium.py  ← Dirty data scenario
│   │   └── task3_hard.py    ← Data leakage scenario
│   └── graders/
│       ├── grader1.py       ← Easy task grader
│       ├── grader2.py       ← Medium task grader
│       └── grader3.py       ← Hard task grader
├── baseline/
│   └── run_baseline.py      ← LLM agent inference script
├── openenv.yaml             ← OpenEnv spec metadata
├── Dockerfile               ← HF Spaces deployment
└── requirements.txt
```

---

## Built By

**CobeDivers — Jey Pranav, Kavin Krish, Hari**
Meta AI × Hugging Face OpenEnv Hackathon

*Live environment: https://huggingface.co/spaces/CoBeDigger/DataEngEnv*
