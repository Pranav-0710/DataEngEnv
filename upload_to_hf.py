from huggingface_hub import HfApi, login
from kaggle_secrets import UserSecretsClient

# 1. Login
login(token=UserSecretsClient().get_secret("HF_TOKEN"))
api = HfApi()
REPO_ID = "CoBeDigger/pipelineops-arena-llama3-grpo"

# 2. Define the Model Card (README.md)
model_card = """---
license: apache-2.0
tags:
- openenv
- rl
- grpo
- unsloth
- data-engineering
- mlops
datasets:
- custom
base_model: unsloth/Meta-Llama-3.1-8B
---

# 🚀 PipelineOps Arena: Llama 3.1 8B (GRPO Trained)

This model was trained using **GRPO (Generative Reinforcement Learning)** to autonomously debug and fix broken Machine Learning pipelines. It interacts with the **[PipelineOps Arena OpenEnv](https://huggingface.co/spaces/CoBeDigger/DataEngEnv)** benchmark.

![Training Progress](reward_curve.png)

## 🏆 Hackathon Submission
This model and environment were built for the **OpenEnv Community Hackathon**. 
- **Live Environment / API:** [PipelineOps Arena Space](https://huggingface.co/spaces/CoBeDigger/DataEngEnv)
- **Base Model:** Llama 3.1 8B (Quantized via Unsloth)
- **RL Algorithm:** GRPO (Generative RLPO)

## 🧠 The Environment (4-Stage Cascade)
Standard RL benchmarks focus on QA or math. We built a continuous, stateful, 4-stage pipeline that tests real-world ML engineering reasoning. The agent must fix one stage to unlock the next. Failure at Stage 4 triggers a loop-back penalty.

1. **Stage 1 (Data Repair):** Diagnose KeyError, rename mismatched columns, and handle injected NaNs/outliers.
2. **Stage 2 (Training Monitor):** Observe loss divergence during `MLPClassifier` training and correctly insert a `StandardScaler`.
3. **Stage 3 (Eval Validation):** Detect suspiciously high accuracy (0.98+) indicating Data Leakage, and restructure the script to move `scaler.fit()` *after* `train_test_split()`.
4. **Stage 4 (Deploy Gate):** Query the MLOps Actor for fairness reviews, detect demographic parity violations, and add `class_weight='balanced'`.

## 📈 Training Details & Dense Semantic Rewards
Standard GRPO trainers are designed for single-turn Q&A, making credit assignment impossible over a 60-step cascade. 

To solve this, we implemented **Dense Semantic Rewards**. Instead of simple text-matching or sparse end-of-episode rewards, the environment parses the AST/script state dynamically after every `edit_script` action.
- *Example:* Fixing a column name yields `+0.2`.
- *Example:* Moving fit after split yields `+0.3`.

As seen in the training curve above, the model successfully explores action spaces in the early episodes and rapidly converges to a flawless optimal policy by Episode 60, achieving a sustained `1.0` average reward.

## 🛠️ How to use
The model is designed to interact with our OpenEnv REST API. It expects the following observation JSON and outputs a structured action JSON:

**Input Observation:**
```json
{
  "last_run_error": "KeyError: 'age_years'",
  "script_content": "...",
  "data_preview": "..."
}
```

**Output Action:**
```json
{
  "action_type": "edit_script",
  "payload": {
    "old": "X = df[['age_years']]",
    "new": "X = df[['age']]"
  }
}
```
"""

# 3. Save and Upload
with open("README.md", "w", encoding="utf-8") as f:
    f.write(model_card)

api.upload_file(
    path_or_fileobj="README.md",
    path_in_repo="README.md",
    repo_id=REPO_ID
)

print(f"✅ Model card successfully updated at: https://huggingface.co/{REPO_ID}")
