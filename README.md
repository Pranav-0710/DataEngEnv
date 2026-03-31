---
title: DataEngEnv
emoji: 🔧
colorFrom: blue
colorTo: green
sdk: docker
pinned: true
tags:
  - openenv
  - ml
  - data-engineering
  - debugging
---

## 1. What is DataEngEnv
DataEngEnv is an interactive ML Data Pipeline Debugger designed to test AI agents against realistic, broken machine learning pipelines. Agents act as automated data engineers, tasked with identifying and fixing issues like critical logical data leakage, dirty missing values, and misaligned schemas before the code hits production.

## 2. Observation Space
| Field | Type | Description |
|---|---|---|
| `task_id` | `int` | The identifier for the current task sequence. |
| `step_number` | `int` | Current sequence iteration limit bound to 20 maximum. |
| `script_content` | `str` | The actual python string content of the ML pipeline being worked on. |
| `last_run_output` | `str` | Terminal STDOUT resulting from the last `run_script` action. |
| `last_run_error` | `str \| null` | Terminal STDERR if the pipeline threw an exception or timeout. |
| `data_preview` | `str` | A printed visualization dump of `.describe()` and `.head()`. |
| `schema_info` | `dict` | A mapping dictionary of column names and respective `dtypes`. |
| `done` | `bool` | True if the agent has submitted its fix or ran out of steps. |

## 3. Action Space
| Action Type | Payload Format | Description |
|---|---|---|
| `inspect_data` | `{}` | Requests `data_preview` of current dataset statistics. |
| `check_schema` | `{}` | Requests current pandas dataframe schema properties. |
| `run_script` | `{}` | Executes the script in a 10s timeout protected sandbox. |
| `edit_script` | `{"old": str, "new": str}` | Simple string patching to correct pipeline code. |
| `submit` | `{}` | Submits script to grading suite finalizing the episode. |

## 4. The 3 Tasks
1. **column_rename_bug (Easy)**: The agent must fix a CSV loading script schema bug by identifying and renaming the string target `age_years` back to `age`.
2. **dirty_data (Medium)**: The agent has to handle real-world missing values and extreme numeric outliers by properly leveraging `dropna`/`fillna` and `clip`/`quantile` constraints.
3. **data_leakage (Hard)**: The agent must fix critical logical target leakage where the predictive `scaler.fit()` was erroneously processed *before* `train_test_split()`, resulting in test set snooping. 

## 5. Reward Function
DataEngEnv operates on a tiered `0.0 to 1.0` reward curve designed to encourage strong Data Engineering practices.
A maximum `1.0` score is only achieved on sequence completion, evaluated against hidden test suites, provided the final script compiles perfectly. However, partial step rewards (e.g. `+0.1` or `+0.2`) are iteratively granted via the `RewardEngine` across each step if the agent dynamically uses `inspect_data` or correctly patches missing `nan` strings before submitting, incentivizing exploratory debugging habits. Submissions with zero exploration or logic modifications receive explicit score deductions (`-0.2`).

## 6. Setup
**Standard Local Setup:**
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app
```
**Docker Build Deployment:**
```bash
docker build -t dataengenv .
docker run -p 7860:7860 dataengenv
```

## 7. API Reference
### Current Endpoint Specs
- **POST `/reset`**
  ```bash
  curl -X POST http://localhost:7860/reset \
  -H "Content-Type: application/json" -d '{"task_id": 1}'
  ```
- **POST `/step`**
  ```bash
  curl -X POST http://localhost:7860/step \
  -H "Content-Type: application/json" -d '{"action_type": "inspect_data", "payload": {}}'
  ```
- **GET `/state`**
  ```bash
  curl http://localhost:7860/state
  ```
- **GET `/tasks`**
  ```bash
  curl http://localhost:7860/tasks
  ```
- **POST `/grader`**
  ```bash
  curl -X POST http://localhost:7860/grader \
  -H "Content-Type: application/json" -d '{"task_id": 1}'
  ```
- **GET `/baseline`**
  ```bash
  curl http://localhost:7860/baseline
  ```

## 8. Baseline Scores
*Evaluated using `llama-3.1-8b-instant` via Groq*

| Task | Run Status | Score | Steps |
|---|---|---|---|
| Task 1 (Easy) | Exhausted Steps | 0.00 | 15 |
| Task 2 (Medium) | Exhausted Steps | 0.00 | 15 |
| Task 3 (Hard) | Partial Credit | 0.75 | 15 |
| **Global Average** | | **0.25** | |

## 9. Team
Ctrl Alt Hacks — Pranav | Meta AI × Hugging Face OpenEnv Challenge
