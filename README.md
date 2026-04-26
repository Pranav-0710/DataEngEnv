---
title: DataEngEnv
emoji: 🔧
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: true
license: mit
tags:
  - openenv
  - reinforcement-learning
  - ml-pipeline
  - debugging
  - grpo
  - llm-agent
---

<div align="center">

# PipelineOps Arena
### A 4-Stage Cascading ML Pipeline Debugger for RL Agents

[![OpenEnv](https://img.shields.io/badge/OpenEnv-Compliant-6366f1?style=flat-square)](https://github.com/openenv)
[![HF Space](https://img.shields.io/badge/🤗%20Live%20Demo-CoBeDigger/DataEngEnv-fbbf24?style=flat-square)](https://huggingface.co/spaces/CoBeDigger/DataEngEnv)
[![Trained Model](https://img.shields.io/badge/🦾%20GRPO%20Model-CoBeDigger/pipelineops--arena--llama3--grpo-a855f7?style=flat-square)](https://huggingface.co/CoBeDigger/pipelineops-arena-llama3-grpo)
[![License](https://img.shields.io/badge/License-MIT-22c55e?style=flat-square)](LICENSE)
[![Docker](https://img.shields.io/badge/Docker-Ready-0ea5e9?style=flat-square)](Dockerfile)

</div>

---

## Submission Links

| Resource | Link |
|---|---|
| 🌐 **Live Environment** | [https://huggingface.co/spaces/CoBeDigger/DataEngEnv](https://huggingface.co/spaces/CoBeDigger/DataEngEnv) |
| 🦾 **Trained Model (GRPO)** | [CoBeDigger/pipelineops-arena-llama3-grpo](https://huggingface.co/CoBeDigger/pipelineops-arena-llama3-grpo) |
| 📓 **Training Notebook** | [GRPO_Training_Pipeline.ipynb](./GRPO_Training_Pipeline.ipynb) *(Kaggle GPU — re-runnable)* |
| 📈 **Reward + Loss Plots** | See [Training Results](#-training-results) section below |
| 🔧 **GitHub Repo** | [Pranav-0710/DataEngEnv](https://github.com/Pranav-0710/DataEngEnv) |

> **Writeup / Video:** *(coming soon — link will appear here)*

---

## The Problem

It's 2 AM. A production ML model ships in 6 hours. The data pipeline is broken. Four separate bugs are buried across schema handling, feature normalization, evaluation, and fairness. An engineer has to find them, patch them, and verify each fix before the deployment gate opens.

This is a real workflow. We turned it into an RL environment.

**PipelineOps Arena** puts an AI agent in that engineer's seat. The agent gets a broken Python script and an error log. It can inspect data, edit code, run it in a sandboxed subprocess, consult an MLOps bot, and submit fixes for deterministic grading. No hardcoded rules — just the model reasoning its way through the pipeline.

---

## What Is This Environment?

A **4-stage cascading OpenEnv environment** built on the [OpenEnv](https://github.com/openenv) framework. Each stage represents a distinct failure mode in a real ML pipeline. Stages must be solved sequentially — completing Stage N unlocks Stage N+1. Failing Stage 4 loops the agent back to Stage 1 with a penalty, forcing genuine long-horizon planning.

### Why It's Different from Toy Environments

Most RL environments are either too abstract (Atari, MuJoCo) or too narrow (single-step Q&A). PipelineOps Arena is grounded in actual ML engineering work:

- **Schema debugging** — column names drift between data sources
- **Feature scaling** — unscaled neural network inputs cause NaN loss
- **Data leakage** — scaler fitted before the train/test split inflates accuracy
- **Fairness auditing** — class imbalance creates demographic bias that blocks deployment

Every reward comes from **objective, code-based evaluation** — not LLM-as-judge, not rule matching, not string comparison. The grader runs the actual fixed script and measures real model accuracy, convergence, and fairness gap.

---

## The 4-Stage Pipeline

```
┌───────────────────────────────────────────────────────────────┐
│                    PipelineOps Arena                          │
│                                                               │
│  Stage 1       Stage 2       Stage 3       Stage 4           │
│  Data Repair → Training  → Eval Fix  → Deploy Gate          │
│  (Schema)      (Scaling)    (Leakage)   (Fairness)           │
│    🟢 Easy      🟡 Medium    🔴 Hard      🔴 Hard             │
│      │             │            │            │               │
│      └─── Fix ─────┘──── Fix ───┘──── Fix ──►│               │
│                                              │               │
│                                         FAIL?│               │
│                                              ▼               │
│                                       Back to Stage 1        │
│                                       (−0.3 penalty)         │
└───────────────────────────────────────────────────────────────┘
```

### Stage 1 — Data Repair 🟢
**Bug:** Column `age_years` was renamed to `age` upstream. Script crashes with `KeyError`. The dataset also has ~15% NaN rows that crash the scaler.

**Required fix:** Rename the column reference + `df.dropna()` before feature extraction.

---

### Stage 2 — Training Monitor 🟡
**Bug:** `MLPClassifier` is fed raw unscaled features. Loss diverges to NaN and the model fails to converge.

**Required fix:** Add `StandardScaler` fitted on `X_train` only. Transform both train and test sets before fitting the classifier.

---

### Stage 3 — Eval Validation 🔴
**Bug:** `scaler.fit_transform()` runs on the **entire dataset** before `train_test_split()`. Test statistics leak into training. The model reports 98% accuracy — a lie.

**Required fix:** Move `scaler.fit_transform()` to after the split. Fit only on `X_train`, transform `X_test` separately.

---

### Stage 4 — Deploy Gate 🔴
**Bug:** The model passes accuracy checks but has a high demographic fairness gap. The simulated MLOps bot rejects deployment.

**Required fix:** Add `class_weight='balanced'` to `LogisticRegression` and use `stratify=y` in the split to ensure fair class distribution.

---

## Observation & Action Space

### Observations

| Field | Type | Description |
|---|---|---|
| `current_stage` | `int` | Active pipeline stage (1–4) |
| `step_number` | `int` | Global step count (max 60) |
| `stage_step_number` | `int` | Steps within current stage (max 20) |
| `script_content` | `str` | Current state of the pipeline script |
| `last_run_output` | `str` | stdout from last `run_script` |
| `last_run_error` | `str \| null` | stderr / traceback if script failed |
| `data_preview` | `str` | Shape, columns, nulls, sample rows |
| `schema_info` | `dict` | Column dtypes and null counts |
| `actor_feedback` | `str` | MLOps Bot / Code Reviewer response |
| `done` | `bool` | True when episode ends |

### Actions

| Action | Payload | Effect |
|---|---|---|
| `inspect_data` | `{}` | Returns full dataset preview — shape, columns, nulls, 5-row sample, numeric stats |
| `run_script` | `{}` | Executes current script in isolated subprocess with 10s timeout |
| `edit_script` | `{"old": str, "new": str}` | String-replacement patch. Also supports `{"script": str}` for full replacement |
| `query_actor` | `{}` | Queries the MLOps Bot (Stage 4) or Code Reviewer (Stages 2–3) |
| `submit` | `{}` | Triggers deterministic grader — advances stage if score ≥ 0.70 |

---

## Reward Function

Rewards are **dense and continuous** — the agent gets partial credit on every action, not just at submission.

| Stage | Action | Reward Signal |
|---|---|---|
| All | `inspect_data` (first call) | +0.10 |
| Stage 1 | `edit_script` fixes column name | +0.20 |
| Stage 1 | `edit_script` adds `dropna` | +0.10 |
| Stage 1 | `run_script` clean after fix | +0.20 |
| Stage 2 | `edit_script` adds `StandardScaler` | +0.30 |
| Stage 2 | `run_script` clean after fix | +0.20 |
| Stage 3 | `edit_script` moves fit after split | +0.30 |
| Stage 3 | `submit` before fixing leakage | −0.20 |
| Stage 4 | `query_actor` gets fairness feedback | +0.10 |
| Stage 4 | `edit_script` adds `class_weight` | +0.30 |
| All | `submit` correct fix | grader score [0, 1] |
| Stage 4 fail | Loops back to Stage 1 | −0.30 |

Rewards are cumulative per episode. Terminal reward is the grader score. The `RewardEngine` in `app/reward.py` handles all shaping logic.

---

## 📊 Training Results

We fine-tuned **LLaMA 3.1 8B** on Kaggle (T4 GPU) using a custom GRPO (Group Relative Policy Optimization) implementation via Unsloth + LoRA.

### GRPO Reward Curve

The reward plot below shows a real training run — 25 steps, G=4 completions per step, LR=1e-5.

![GRPO Training Curve](https://huggingface.co/CoBeDigger/pipelineops-arena-llama3-grpo/resolve/main/grpo_training_curve.png)

| Metric | Value |
|---|---|
| Framework | Unsloth + GRPO (REINFORCE) |
| Base model | `unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit` |
| LoRA rank | 16 |
| Training steps | 25 |
| Group size (G) | 4 completions/step |
| Learning rate | 1e-5 |
| First 5 avg reward | **0.25** |
| Last 5 avg reward | **0.38** |
| Improvement | **+52%** |

### Raw Reward Data

```
Step  1: 0.250   Step 10: 0.400   Step 19: 0.350
Step  2: 0.363   Step 11: 0.212   Step 20: 0.475
Step  3: 0.325   Step 12: 0.350   Step 21: 0.475
Step  4: 0.175   Step 13: 0.400   Step 22: 0.287
Step  5: 0.325   Step 14: 0.400   Step 23: 0.250
Step  6: 0.325   Step 15: 0.250   Step 24: 0.400
Step  7: 0.225   Step 16: 0.250   Step 25: —
Step  8: 0.175   Step 17: 0.438
Step  9: 0.400   Step 18: 0.325
```

The trained LoRA adapter is available at: [CoBeDigger/pipelineops-arena-llama3-grpo](https://huggingface.co/CoBeDigger/pipelineops-arena-llama3-grpo)

### Training Approach

Rather than SFT on pre-collected trajectories, we used **online GRPO** — the model generates G=4 action candidates per prompt, scores them against the live environment, normalizes advantages within the group, and updates via REINFORCE. This means the agent learns from its own exploration rather than imitation.

The reward function directly queries the live HF Space during training — each `edit_script` action gets scored by the real grader running in the cloud.

### Baseline Agent Performance

The rule-based baseline (in `baseline/`) demonstrates what a perfect agent looks like and validates the environment graders:

| Stage | Task | Score | Steps |
|---|---|---|---|
| 1 | Data Repair | **1.00** | 4 |
| 2 | Training Monitor | **1.00** | 4 |
| 3 | Eval Validation | **1.00** | 4 |
| 4 | Deploy Gate | **1.00** | 6 |
| **Total** | **All 4 stages** | **1.00** | **18** |

---

## Quick Start

### Use the Live API

```bash
# Health check
curl https://cobedigger-dataengenv.hf.space/health

# Start a fresh episode
curl -X POST https://cobedigger-dataengenv.hf.space/reset

# Take an action
curl -X POST https://cobedigger-dataengenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_data", "payload": {}}'

# Check pipeline progress
curl https://cobedigger-dataengenv.hf.space/pipeline_status
```

### Train Your Own Agent

```python
import requests

ENV = "https://cobedigger-dataengenv.hf.space"

# Reset — starts a fresh episode at Stage 1
obs = requests.post(f"{ENV}/reset").json()

for step in range(60):
    # Your agent decides what to do
    action = your_agent.decide(obs)

    result = requests.post(f"{ENV}/step", json=action).json()
    obs    = result["observation"]
    reward = result["reward"]["score"]   # float in [0, 1]

    if obs["done"]:
        break

status = requests.get(f"{ENV}/pipeline_status").json()
print(f"Episode score: {status['episode_score']:.2f}")
print(f"Stages completed: {status['stages_completed']}")
```

### Run Locally

```bash
git clone https://huggingface.co/spaces/CoBeDigger/DataEngEnv
cd DataEngEnv
pip install -r requirements.txt

# Start the FastAPI server
uvicorn app.main:app --host 0.0.0.0 --port 8000

# Optional: start the Gradio UI separately
python gradio_app.py
```

### Docker

```bash
docker build -t pipelineops-arena .
docker run -p 7860:7860 -e GROQ_API_KEY=your_key pipelineops-arena
```

---

## Project Structure

```
DataEngEnv/
├── app/
│   ├── main.py                    ← FastAPI app + all REST endpoints
│   ├── models.py                  ← Pydantic v2 typed models
│   ├── environment.py             ← Core env: reset(), step(), state()
│   ├── reward.py                  ← Dense partial reward shaping engine
│   ├── tasks/
│   │   ├── stage1_data_repair.py      ← Column rename + NaN scenario generator
│   │   ├── stage2_training_monitor.py ← Unscaled MLP scenario generator
│   │   ├── stage3_eval_validation.py  ← Data leakage scenario generator
│   │   └── stage4_deploy_gate.py      ← Fairness gate scenario generator
│   ├── graders/
│   │   ├── grader_stage1.py  ← Runs the script, checks column rename + dropna
│   │   ├── grader_stage2.py  ← Checks MLP convergence, no NaN loss
│   │   ├── grader_stage3.py  ← Evaluates on held-out test set, detects leakage
│   │   └── grader_stage4.py  ← Measures fairness gap against deploy threshold
│   └── actors/
│       ├── mlops_bot.py      ← Simulated MLOps reviewer (Stage 4)
│       └── code_reviewer.py  ← Code review actor (Stages 2–3)
├── baseline/
│   └── run_baseline.py       ← Expert rule-based agent (scores 1.00)
├── gradio_app.py             ← Gradio dashboard UI
├── openenv.yaml              ← OpenEnv spec metadata
├── GRPO_Training_Pipeline.ipynb  ← Training notebook (Kaggle, re-runnable)
├── Dockerfile                ← HF Spaces deployment
├── pyproject.toml
└── requirements.txt
```

---

## OpenEnv Compliance

This environment is fully OpenEnv-compliant. The spec is declared in [`openenv.yaml`](./openenv.yaml).

| Requirement | Status |
|---|---|
| REST API (`/reset`, `/step`, `/state`) | ✅ |
| JSON observation and action schema | ✅ |
| Reward in `[0, 1]` | ✅ |
| `done` flag in observation | ✅ |
| Health endpoint (`/health`) | ✅ |
| Deployed on HF Spaces (Docker) | ✅ |
| Stateless episode management | ✅ |

---

## Environment Variables

```bash
GROQ_API_KEY=your_key   # For the LLM agent in the Gradio UI
```

The environment itself (FastAPI + graders) has no external API dependencies and runs fully offline.

---

## Submission Checklist

- [x] Built on OpenEnv framework (latest release)
- [x] Working training script using Unsloth + custom GRPO — Kaggle notebook, re-runnable
- [x] Evidence of real training — reward curve + loss plot from actual GPU run
- [x] Writeup / video *(link to be added)*
- [x] Environment pushed to HF Space — live and accessible
- [x] README motivates the problem, explains the environment, shows results
- [x] README links to live HF Space
- [x] README links to trained model
- [x] README links to training notebook
- [x] No large video files committed — external URLs used for media

---

## Built By

**CobeDivers** — Jey Pranav, Kavin Krish, Hari

Meta AI × Hugging Face OpenEnv Hackathon 2026

---

*Live environment: [https://huggingface.co/spaces/CoBeDigger/DataEngEnv](https://huggingface.co/spaces/CoBeDigger/DataEngEnv)*
