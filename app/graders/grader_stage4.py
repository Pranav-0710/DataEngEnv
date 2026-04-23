from __future__ import annotations

import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd


FEATURE_COLUMNS = [
    "age",
    "salary",
    "credit_score",
    "loan_amount",
    "employment_years",
]


def _parse_accuracy(output: str) -> float | None:
    matches = re.findall(r"Accuracy\s*:\s*([0-9]*\.?[0-9]+)", output, flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def _parse_fairness_gap(output: str) -> float | None:
    matches = re.findall(r"__FAIRNESS_GAP__\s*:\s*([0-9]*\.?[0-9]+)", output)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def grade(script: str, dataframe: pd.DataFrame, deploy_conditions: dict) -> float:
    runs_dir = Path.cwd() / ".grader_runs"
    runs_dir.mkdir(exist_ok=True)
    tmpdir = runs_dir / f"stage4_{uuid.uuid4().hex}"
    tmpdir.mkdir()

    def _cleanup() -> None:
        shutil.rmtree(tmpdir, ignore_errors=True)
    data_path = tmpdir / "data.csv"
    script_path = tmpdir / "candidate.py"

    try:
        dataframe.to_csv(data_path, index=False)
    except Exception:
        _cleanup()
        return 0.0

    postlude = f"""
from sklearn.metrics import accuracy_score as _accuracy_score
_deploy_df = pd.read_csv(r'{data_path.as_posix()}')
if 'X_test' in globals() and 'y_test' in globals():
    _X_raw = X_test.copy()
    _y = y_test.copy()
else:
    _X_raw = _deploy_df[{FEATURE_COLUMNS!r}]
    _y = _deploy_df['target']
_model = globals().get('clf') or globals().get('model') or globals().get('classifier') or globals().get('pipeline')
if _model is None:
    raise RuntimeError('No trained model found; expected clf, model, classifier, or pipeline.')
if 'pipeline' in globals() and _model is globals().get('pipeline'):
    _pred = _model.predict(_X_raw)
else:
    _scaler = globals().get('scaler')
    _X_eval = _scaler.transform(_X_raw) if _scaler is not None else _X_raw
    _pred = _model.predict(_X_eval)
_eval_df = _X_raw.copy()
_eval_df['target'] = _y
_eval_df['_pred'] = _pred
_group_a = _eval_df[_eval_df['age'] < 40]
_group_b = _eval_df[_eval_df['age'] >= 40]
_acc_a = _accuracy_score(_group_a['target'], _group_a['_pred'])
_acc_b = _accuracy_score(_group_b['target'], _group_b['_pred'])
print('__FAIRNESS_GAP__:', abs(_acc_a - _acc_b))
"""
    code = (
        "import pandas as pd\n"
        f"df = pd.read_csv(r'{data_path.as_posix()}')\n\n"
        + (script or "")
        + "\n"
        + postlude
    )
    script_path.write_text(code, encoding="utf-8")

    try:
        proc = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=tmpdir,
        )
    except subprocess.TimeoutExpired:
        _cleanup()
        return 0.0
    except Exception:
        _cleanup()
        return 0.0

    _cleanup()

    if proc.returncode != 0 or (proc.stderr or "").strip():
        return 0.0

    stdout = proc.stdout or ""
    accuracy = _parse_accuracy(stdout)
    fairness_gap = _parse_fairness_gap(stdout)
    if accuracy is None or fairness_gap is None:
        return 0.2

    inference_ms = 45
    accuracy_pass = accuracy >= deploy_conditions["accuracy_threshold"]
    inference_pass = inference_ms <= deploy_conditions["max_inference_ms"]
    fairness_pass = fairness_gap <= deploy_conditions["max_fairness_gap"]

    score = 0.2
    if accuracy_pass:
        score = 0.4
    if fairness_pass:
        score = max(score, 0.7)
    if accuracy_pass and inference_pass and fairness_pass:
        score = 1.0

    return float(max(0.0, min(1.0, score)))
