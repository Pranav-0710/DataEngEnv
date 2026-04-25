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
  - reinforcement-learning
---

# 🔧 PipelineOps Arena — ML Data Pipeline Debugger

![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-blue)
![Docker](https://img.shields.io/badge/Docker-Ready-success)
![HF Spaces](https://img.shields.io/badge/HuggingFace-Spaces-yellow)
![Stages](https://img.shields.io/badge/Stages-4-purple)
![Baseline Score](https://img.shields.io/badge/Baseline-1.00-brightgreen)

> *It's 2 AM. The model ships to production in 6 hours. The pipeline is broken.*
> *Can your AI agent fix it in time?*

---

## 📦 Submission Links

| Resource | Link |
|---|---|
| 🌐 **Live Environment** | [https://huggingface.co/spaces/CoBeDigger/DataEngEnv](https://huggingface.co/spaces/CoBeDigger/DataEngEnv) |
| 🤖 **Trained Model** | [CoBeDigger/pipelineops-arena-llama3-grpo](https://huggingface.co/CoBeDigger/pipelineops-arena-llama3-grpo) |
| 📈 **Reward Curve** | [View on Model Repo](https://huggingface.co/CoBeDigger/pipelineops-arena-llama3-grpo/blob/main/reward_curve.png) |
| 📓 **Training Notebook** | *(Add your Kaggle/Colab link here)* |

---

## What is PipelineOps Arena?

PipelineOps Arena is a **4-stage cascading OpenEnv environment** where an AI agent plays the role of a **frantic Data Engineer debugging a broken ML pipeline before a model goes to production.**

Unlike simple game environments, this tests real-world engineering skills: schema debugging, feature scaling, data leakage detection, and fairness auditing — the exact workflow a human ML engineer follows every day.

### Why This Environment Is Different

- **4-stage cascade**: Each stage must be solved to unlock the next
- **Loop mechanic**: Failing Stage 4 sends the agent back to Stage 1 with a penalty — forcing genuine long-horizon planning
- **Actor bots**: Stage 4 includes a simulated MLOps reviewer that provides fairness feedback
- **Deterministic grading**: Every reward comes from objective, code-based evaluation — not LLM-as-judge

---

## 🏗️ The 4-Stage Pipeline

```
Stage 1          Stage 2          Stage 3          Stage 4
Data Repair  →  Training Fix  →  Eval Fix     →  Deploy Gate
(Schema Bug)    (Divergence)     (Leakage)       (Fairness)
   🟢 Easy       🟡 Medium       🔴 Hard         🔴 Hard
     │               │               │               │
     └─── Fix ───────┘─── Fix ───────┘─── Fix ───────┘
                                                      │
                                              ┌───────┘
                                              │ FAIL?
                                              ▼
                                         Back to Stage 1
                                         (with penalty)
```

### Stage 1 — Data Repair 🟢

**Bug:** Column `age_years` was renamed to `age` in source data. Script crashes with `KeyError`. Also has ~15% NaN values.

**Fix:** Rename column reference + `dropna()`

### Stage 2 — Training Monitor 🟡

**Bug:** `MLPClassifier` trains on unscaled features → loss diverges, model fails to converge.

**Fix:** Add `StandardScaler` before the classifier.

### Stage 3 — Eval Validation 🔴

**Bug:** `scaler.fit_transform()` runs on the **entire dataset before** `train_test_split()`. The test set has seen training statistics. 98% accuracy is a lie.

**Fix:** Move `scaler.fit()` to after the split — fit only on training data.

### Stage 4 — Deploy Gate 🔴

**Bug:** Model passes accuracy checks but has demographic bias. The MLOps bot rejects deployment.

**Fix:** Add `class_weight='balanced'` to `LogisticRegression` to ensure fairness across groups.

---

## 📊 Training Results

We fine-tuned **LLaMA 3.1 8B** using SFT warmup followed by GRPO (Group Relative Policy Optimization) via HuggingFace TRL.

| Metric | Value |
|---|---|
| First 10 episodes avg | **0.15** |
| Last 10 episodes avg | **0.52** |
| Improvement | **+0.37** |
| Peak score | **1.00** |
| Total episodes | 30 |

The reward curve shows a clear upward learning trend — the agent progresses from random exploration (near-zero reward) to consistently completing multiple pipeline stages.

### Baseline Agent Performance

| Stage | Task | Score | Steps |
|---|---|---|---|
| Stage 1 | Data Repair | **1.00** | 4 |
| Stage 2 | Training Monitor | **1.00** | 4 |
| Stage 3 | Eval Validation | **1.00** | 4 |
| Stage 4 | Deploy Gate | **1.00** | 6 |
| **Total** | **All 4 Stages** | **1.00** | **18** |

---

## Observation Space

| Field | Type | Description |
|---|---|---|
| `current_stage` | `int` | Current pipeline stage (1-4) |
| `step_number` | `int` | Current step (max 60 total, 15 per stage) |
| `script_content` | `str` | The current state of the pipeline script |
| `last_run_output` | `str` | stdout from the last `run_script` action |
| `last_run_error` | `str \| null` | stderr / exception if the script crashed |
| `data_preview` | `str` | DataFrame shape, columns, nulls, sample rows |
| `schema_info` | `dict` | Column names, dtypes, null counts |
| `done` | `bool` | True when all stages complete or steps exhausted |

## Action Space

| Action | Payload | What it does |
|---|---|---|
| `inspect_data` | `{}` | Reveals full dataset preview — columns, nulls, outliers |
| `check_schema` | `{}` | Compares expected vs actual column names |
| `run_script` | `{}` | Executes current script in a 10s sandboxed subprocess |
| `edit_script` | `{"old": str, "new": str}` | Patches the script with a string replacement |
| `query_actor` | `{}` | Consult the MLOps bot for deployment feedback |
| `submit` | `{}` | Submits the fix to the grader — advances to next stage |

---

## Reward Function

DataEngEnv provides **partial credit throughout the episode** — not just a binary win/lose at the end.

| Stage | Action | Reward |
|---|---|---|
| Stage 1 | First `inspect_data` or `check_schema` | +0.1 |
| Stage 1 | `edit_script` fixes column name | +0.3 |
| Stage 1 | `run_script` with no error | +0.2 |
| Stage 2 | `inspect_data` reveals divergence | +0.15 |
| Stage 2 | `edit_script` adds StandardScaler | +0.3 |
| Stage 3 | `run_script` (sees suspicious accuracy) | +0.2 |
| Stage 3 | `edit_script` moves fit after split | +0.3 |
| Stage 4 | `query_actor` gets fairness feedback | +0.1 |
| Stage 4 | `edit_script` adds class_weight balanced | +0.3 |
| All | `submit` with correct fix | **grader score** |

Rewards are cumulative and capped at 1.0 per stage. The terminal reward is the grader score (0.0–1.0).

---

## 🚀 Quick Start

### Use the Live API

```bash
# Health check
curl https://cobedigger-dataengenv.hf.space/health

# Reset environment (start fresh episode)
curl -X POST https://cobedigger-dataengenv.hf.space/reset \
  -H "Content-Type: application/json" -d '{}'

# Take an action
curl -X POST https://cobedigger-dataengenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_data", "payload": {}}'

# Check pipeline progress
curl https://cobedigger-dataengenv.hf.space/pipeline_status
```

### Run Locally

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

### Train Your Own Agent

```python
import requests

ENV_URL = "https://cobedigger-dataengenv.hf.space"

# Reset
obs = requests.post(f"{ENV_URL}/reset").json()

# Agent loop
for step in range(60):  # 60 total steps, 15 per stage
    action = your_agent.decide(obs)
    result = requests.post(f"{ENV_URL}/step", json=action).json()
    obs = result["observation"]
    reward = result["reward"]
    
    if obs["done"]:
        break

# Check final score
status = requests.get(f"{ENV_URL}/pipeline_status").json()
print(f"Score: {status['episode_score']}")
```

---

## Project Structure

```
PipelineOps-Arena/
├── app/
│   ├── main.py              ← FastAPI app, all endpoints
│   ├── models.py            ← Pydantic v2 typed models
│   ├── environment.py       ← Core env: reset(), step(), state()
│   ├── reward.py            ← Partial reward shaping engine
│   ├── tasks/
│   │   ├── stage1_data_repair.py     ← Column rename + NaN scenario
│   │   ├── stage2_training_monitor.py ← Feature scaling scenario
│   │   ├── stage3_eval_validation.py  ← Data leakage scenario
│   │   └── stage4_deploy_gate.py      ← Fairness gate scenario
│   ├── graders/
│   │   ├── grader_stage1.py  ← Schema fix grader
│   │   ├── grader_stage2.py  ← Training convergence grader
│   │   ├── grader_stage3.py  ← Leakage detection grader
│   │   └── grader_stage4.py  ← Fairness threshold grader
│   └── actors/
│       ├── mlops_bot.py      ← Simulated MLOps reviewer
│       └── code_reviewer.py  ← Code review actor
├── baseline/
│   └── run_baseline.py       ← Expert baseline agent
├── gradio_app.py             ← Premium Gradio dashboard UI
├── openenv.yaml              ← OpenEnv spec metadata
├── Dockerfile                ← HF Spaces deployment
├── pyproject.toml            ← Package configuration
└── requirements.txt
```

---

## Environment Variables

```bash
OPENAI_API_KEY=your_key    # for actor bots (optional)
GROQ_API_KEY=your_key      # for baseline script (optional)
```

---

## Built By

**CobeDivers — Jey Pranav, Kavin Krish, Hari**

Meta AI × Hugging Face OpenEnv Hackathon 2026

*Live environment: [https://huggingface.co/spaces/CoBeDigger/DataEngEnv](https://huggingface.co/spaces/CoBeDigger/DataEngEnv)*
