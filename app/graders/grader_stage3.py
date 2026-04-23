from __future__ import annotations

import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "age",
    "salary",
    "credit_score",
    "loan_amount",
    "employment_years",
]


def _detect_fit_after_split(script: str) -> bool:
    lines = script.split('\n')
    # Strip import lines and comments entirely
    code_lines = [
        l for l in lines 
        if not l.strip().startswith('import') 
        and not l.strip().startswith('from')
        and not l.strip().startswith('#')
        and l.strip() != ''
    ]
    code_only = '\n'.join(code_lines)
    
    fit_pos = code_only.find('scaler.fit')
    split_pos = code_only.find('train_test_split(')
    
    # Fix is correct only if train_test_split appears BEFORE scaler.fit in code
    if fit_pos == -1 or split_pos == -1:
        return False
    return split_pos < fit_pos


def _parse_marker(output: str) -> float | None:
    matches = re.findall(r"__HELD_OUT_ACCURACY__\s*:\s*([0-9]*\.?[0-9]+)", output)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def grade(script: str, full_df: pd.DataFrame, held_out_X, held_out_y) -> float:
    fit_after_split = _detect_fit_after_split(script or "")

    runs_dir = Path.cwd() / ".grader_runs"
    runs_dir.mkdir(exist_ok=True)
    tmpdir = runs_dir / f"stage3_{uuid.uuid4().hex}"
    tmpdir.mkdir()

    def _cleanup() -> None:
        shutil.rmtree(tmpdir, ignore_errors=True)
    data_path = tmpdir / "data.csv"
    heldout_path = tmpdir / "held_out.csv"
    script_path = tmpdir / "candidate.py"

    try:
        full_df.to_csv(data_path, index=False)
        heldout_df = pd.DataFrame(np.asarray(held_out_X), columns=FEATURE_COLUMNS)
        heldout_df["target"] = np.asarray(held_out_y)
        heldout_df.to_csv(heldout_path, index=False)
    except Exception:
        _cleanup()
        return 0.0

    postlude = f"""
from sklearn.metrics import accuracy_score as _accuracy_score
_held_df = pd.read_csv(r'{heldout_path.as_posix()}')
_X_held_raw = _held_df[{FEATURE_COLUMNS!r}]
_y_held = _held_df['target']
_model = globals().get('clf') or globals().get('model') or globals().get('classifier') or globals().get('pipeline')
if _model is None:
    raise RuntimeError('No trained model found; expected clf, model, classifier, or pipeline.')
if 'pipeline' in globals() and _model is globals().get('pipeline'):
    _pred = _model.predict(_X_held_raw)
else:
    _scaler = globals().get('scaler')
    _X_held = _scaler.transform(_X_held_raw) if _scaler is not None else _X_held_raw
    _pred = _model.predict(_X_held)
print('__HELD_OUT_ACCURACY__:', _accuracy_score(_y_held, _pred))
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

    held_out_accuracy = _parse_marker(proc.stdout or "")
    if held_out_accuracy is None or held_out_accuracy < 0.60:
        return 0.0

    score = 0.5
    if fit_after_split:
        score = max(score, 0.3)
        if held_out_accuracy >= 0.75:
            score = 1.0
        elif held_out_accuracy >= 0.70:
            score = 0.75

    return float(max(0.0, min(1.0, score)))
