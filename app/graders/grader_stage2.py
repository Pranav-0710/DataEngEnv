from __future__ import annotations

import re
from typing import Optional


def grade(script: str, run_output: str, run_error: Optional[str]) -> float:
    """
    Stage 2 Grader — Training Monitor.

    Scoring ladder (deterministic string-based):
      0.0 → script crashes
      0.3 → script contains StandardScaler
      0.5 → scaler fitted correctly (after train_test_split)
      0.7 → script runs without error
      0.85 → script runs clean AND loss is not NaN
      1.0 → script runs clean AND prints final accuracy >= 0.60
    """
    run_error = run_error or ""
    run_output = run_output or ""
    script = script or ""

    # Hard crash
    if run_error.strip() != "":
        # We can still award partial credit if they were working on scaling
        if "StandardScaler" in script:
            # Check if fitted after split (rough regex)
            if re.search(r"train_test_split.*fit_transform", script, flags=re.DOTALL):
                return 0.5
            return 0.3
        return 0.0

    # No crash: baseline 0.7
    score = 0.7

    # Check for NaN loss
    if "nan" not in run_output.lower():
        score = 0.85

    # Check for accuracy >= 0.60
    acc_match = re.search(r"Accuracy\s*:\s*(0\.[6-9]\d*|1\.0+)", run_output, flags=re.IGNORECASE)
    if acc_match and score == 0.85:
        score = 1.0

    return score
