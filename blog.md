---
title: "PipelineOps Arena: We Built an RL Environment Around the Bugs That Actually Break Production"
thumbnail: https://huggingface.co/spaces/CoBeDigger/DataEngEnv/resolve/main/imgmeta.png
authors:
  - user: CoBeDigger
---

*A 4-stage cascading ML pipeline debugger for RL agents — Meta AI × Hugging Face OpenEnv Hackathon 2026.*

---

Every ML team has shipped a broken model to production at least once. The pipeline looked fine. The numbers looked good. But somewhere between the data loading and the deployment gate, something went wrong — silently, invisibly, and expensively.

We built **PipelineOps Arena** because these bugs are real, they happen constantly, and no one has tried to train an AI to catch them systematically.

Try it live: [huggingface.co/spaces/CoBeDigger/DataEngEnv](https://huggingface.co/spaces/CoBeDigger/DataEngEnv)

---

## The Four Bugs We're Testing

We didn't invent these failure modes. We've lived them.

**Stage 1 — Data Repair.** Someone renamed a column upstream. The script references `age_years` but the DataFrame column is `age`. `KeyError`. The dataset also has ~15% NaN rows that crash `StandardScaler` with `ValueError`. Two bugs, both required to pass.

**Stage 2 — Training Monitor.** An `MLPClassifier` is fed raw unscaled features — salary around 70,000 sitting next to employment years around 8. Loss diverges to NaN after epoch 1. No crash. Silent divergence.

**Stage 3 — Eval Validation.** `scaler.fit_transform()` runs on the entire dataset before `train_test_split()`. The model has already seen test data during normalization. It prints `Accuracy: 0.98`. It's lying.

**Stage 4 — Deploy Gate.** The model passes accuracy. It passes inference latency. But the MLOps bot measures a 0.09 demographic fairness gap across age groups — above the 0.05 deployment threshold. Deployment rejected.

These aren't abstract challenges. They are the exact class of bugs that get missed in code review because they don't crash — they just corrupt.

---

## How the Environment Works

```text
┌──────────────────────────────────────────────────────────┐
│                    PipelineOps Arena                     │
│                                                          │
│   Stage 1      Stage 2      Stage 3       Stage 4        │
│  Data Repair → Training  → Eval Fix  → Deploy Gate      │
│  (KeyError)    (NaN loss)  (Leakage)    (Fairness)       │
│                                                          │
│   Each stage: inspect → edit → run → submit              │
│   Grade ≥ 0.70  →  advance to next stage                │
│   Grade < 0.70 at Stage 4  →  loop back (−0.30)         │
│   Max steps: 60 total, 15 per stage                      │
└──────────────────────────────────────────────────────────┘
```

The agent talks to a **FastAPI server** over plain HTTP — no special SDK, just REST. Built on the [OpenEnv](https://github.com/openenv) spec.

At each step the agent gets an observation like this:

```json
{
  "current_stage": 2,
  "step_number": 7,
  "script_content": "...current pipeline code...",
  "last_run_error": "ValueError: Input contains NaN",
  "data_preview": "Shape: (500, 6)\nColumns: ['age', 'salary', ...]",
  "actor_feedback": "StandardScaler not found in script.",
  "stages_completed": [1],
  "done": false
}
```

And picks one of five actions:

| Action | What it does |
| --- | --- |
| `inspect_data` | Returns dataset shape, columns, null counts, sample rows |
| `run_script` | Executes current pipeline in a 10-second isolated sandbox |
| `edit_script` | Patches specific text in the script, or replaces it entirely |
| `query_actor` | Asks the Code Reviewer (Stages 2–3) or MLOps Bot (Stage 4) |
| `submit` | Triggers the deterministic grader |

The key design decision: **every reward comes from objective, code-based evaluation**. No LLM-as-judge. No string matching. The grader runs the fixed script against real data and measures actual accuracy, convergence, and fairness gap. If the fix works, it works.

---

## The Cascade Mechanic

The stages aren't just sequential — they're designed to punish shallow fixes.

Stage 4 has a **loop mechanic**: fail the fairness gate and the agent goes back to Stage 1 with a −0.30 penalty. Two failed loops end the episode at 0.00.

This forces genuine long-horizon planning. An agent that blindly submits at every stage will hit Stage 4, fail the fairness check, and lose everything it earned. The fairness gap (0.09 across age groups) only becomes visible when the agent explicitly queries the MLOps Bot — which runs the script in a subprocess and measures it directly. The hint is there: *"Consider class_weight='balanced'"*. But the agent has to know to ask.

---

## Two Reviewers the Agent Can Consult

**Code Reviewer** (Stages 2–3): Regex-based. Checks for data leakage patterns — whether `scaler.fit()` was called before or after the split. Gives natural-language feedback like *"Scaler is being fit on the full dataset. This leaks test statistics into training."*

**MLOps Bot** (Stage 4): Actually runs the agent's current script in a subprocess with a fairness-gap measurement appended. Reports accuracy, inference latency, and the demographic gap with explicit pass/fail labels. Every `query_actor` call re-executes — it's compute-heavy and intentional. The agent should only call it when it's ready for real feedback.

---

## Reward Structure

Dense rewards throughout — partial credit for doing the right things, not just for completing stages.

```text
inspect_data (first time)           +0.10
edit_script: fix column name        +0.20
edit_script: add dropna/clip        +0.15
run_script: clean after fix         +0.20
edit_script: add StandardScaler     +0.30
edit_script: fix data leakage       +0.30
query_actor: MLOps Bot              +0.10
edit_script: add class_weight       +0.30
submit: stage complete              grader score [0.0–1.0]
Stage 4 failure (loop back)         −0.30
Repeated action                     −0.03
```

A perfect run — all 4 stages cleared without looping — returns **1.00**.

---

## Training: Llama 3.1 8B with GRPO

We fine-tuned **Llama 3.1 8B** (4-bit quantized via Unsloth) using GRPO — Group Relative Policy Optimization — on Kaggle T4 GPU.

**Setup:** LoRA r=16, all projection layers, 0.26% trainable params, TRL GRPOTrainer.

### Why Semantic Reward

The first thing we tried was episode completion as the reward signal. It produced nothing.

The untrained Llama 3.1 8B instruct model doesn't output `{"action_type": "edit_script", "payload": {...}}`. It outputs rich descriptive JSON — multi-step plans, code analysis, detailed reasoning about the bug. All valid ML thinking, none of it parseable as an environment action.

That meant every episode returned 0.00. `reward_std = 0.0`. GRPO with zero variance produces zero gradient. The model learned nothing.

So we built a **semantic reward function** that scores each response on format quality alone:

```text
+0.20  Valid JSON output
+0.20  action_type key present
+0.30  Correct action keyword for current context
+0.30  Correct payload structure
```

This decoupled learning from episode completion. The model gets a gradient signal just for structuring its output correctly — which is the prerequisite for everything else.

### Results

![GRPO Training Curves — Avg Reward and Policy Loss per Step](https://huggingface.co/spaces/CoBeDigger/DataEngEnv/resolve/main/imgmeta.png)

*Left: avg group reward per training step. Right: GRPO policy loss per step. Both from a real 25-step run on Kaggle T4.*

| Metric | Value |
| --- | --- |
| Base model | Llama 3.1 8B 4-bit (Unsloth) |
| Training steps | 25 |
| Group size (G) | 4 completions per step |
| First 5 avg reward | **0.25** |
| Last 5 avg reward | **0.38** |
| Improvement | **+52%** |

The reward went from 0.25 to 0.38 over 25 steps. The variance is the real signal — early steps show wide swings as the model explores different formats; later steps tighten. The model learned that structured JSON is rewarded, and started producing it more reliably.

The untrained model scored 0.00 on episode completion because it couldn't produce parseable actions. The trained adapter pushes toward the correct action format — which is the necessary first step before episode-level RL becomes possible.

Trained model: [CoBeDigger/pipelineops-arena-llama3-grpo](https://huggingface.co/CoBeDigger/pipelineops-arena-llama3-grpo)

Training notebook (Colab, re-runnable): [GRPO_Training_Pipeline.ipynb](https://colab.research.google.com/drive/1HmHVyfEzDptOmAsvTCVpymsNwxaQuyx7?usp=sharing)

---

## The Baseline: What Perfect Looks Like

We built a hardcoded rule-based agent that executes the optimal action sequence per stage. It completes all 4 stages in **16 steps** with a score of **1.00**. This validates the graders and gives a ceiling to measure against.

```text
Stage 1: edit_script (column fix + dropna) → run_script → submit       [3 steps]
Stage 2: inspect_data → edit_script (add scaler) → run_script → submit  [4 steps]
Stage 3: run_script → edit_script (fix leakage) → run_script → submit   [4 steps]
Stage 4: query_actor → edit_script (class_weight) → run_script → submit [4 steps]
```

You can watch it run live on the Space under the **"Baseline vs LLM Agent"** tab.

Video walkthrough: [Watch on YouTube](https://drive.google.com/drive/folders/1hOnNKyAypjJoKPM-zzwZvskM_NCdmSR5?usp=sharing)

---

## Try It

No authentication required. Just HTTP.

```bash
# Start an episode
curl -X POST https://cobedigger-dataengenv.hf.space/reset

# Take a step
curl -X POST https://cobedigger-dataengenv.hf.space/step \
  -H "Content-Type: application/json" \
  -d '{"action_type": "inspect_data", "payload": {}}'

# Check pipeline progress
curl https://cobedigger-dataengenv.hf.space/pipeline_status
```

Or run an agent loop:

```python
import requests

ENV = "https://cobedigger-dataengenv.hf.space"
obs = requests.post(f"{ENV}/reset").json()

for step in range(60):
    action = your_agent.decide(obs)
    result = requests.post(f"{ENV}/step", json=action).json()
    obs = result["observation"]
    reward = result["reward"]["score"]
    if obs["done"]:
        break
```

**Links:**

- Live Space: [huggingface.co/spaces/CoBeDigger/DataEngEnv](https://huggingface.co/spaces/CoBeDigger/DataEngEnv)
- Trained model: [CoBeDigger/pipelineops-arena-llama3-grpo](https://huggingface.co/CoBeDigger/pipelineops-arena-llama3-grpo)
- Training notebook: [Colab](https://colab.research.google.com/drive/1HmHVyfEzDptOmAsvTCVpymsNwxaQuyx7?usp=sharing)
- GitHub: [Pranav-0710/DataEngEnv](https://github.com/Pranav-0710/DataEngEnv)
- Video: [YouTube walkthrough](https://drive.google.com/drive/folders/1hOnNKyAypjJoKPM-zzwZvskM_NCdmSR5?usp=sharing)

---

## What We Learned

**Reward design is the hardest part.** Getting a gradient signal out of a model that can't produce valid JSON yet requires rethinking what "reward" means at the start of training. Semantic reward isn't a hack — it's necessary scaffolding.

**Real execution beats string matching.** Every grader in this environment runs actual Python code against real data. This makes it harder to game and means reward always reflects genuine task completion.

**The cascade forces planning.** Agents that treat each stage independently will fail Stage 4 and lose 0.30 points. The loop mechanic rewards agents that maintain a holistic view of the pipeline — not just the current bug.

---

*CobeDivers — Jey Pranav, Kavin Krish, Hari*
*Meta AI × Hugging Face OpenEnv Hackathon 2026*
