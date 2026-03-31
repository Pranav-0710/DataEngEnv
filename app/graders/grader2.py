from __future__ import annotations

import re
from typing import Optional


def _contains_accuracy(output: str) -> bool:
    return bool(re.search(r"Accuracy\s*:\s*\d*\.?\d+", output, flags=re.IGNORECASE))


def _has_nan_handling(script: str) -> bool:
    s = script.lower()
    return any(k in s for k in ["dropna", "fillna", "simpleimputer", "isna()", "notna()"])


def _has_outlier_handling(script: str) -> bool:
    s = script.lower()
    # Common outlier treatments: clip, quantile capping, IQR-based filtering, winsorize
    return any(k in s for k in ["clip(", "quantile(", "iqr", "winsor", "robustscaler"]) 


def grade(script: str, run_output: str, run_error: Optional[str]) -> float:
    """
    Task 2 Grader

    Scoring (pick highest satisfied level, deterministic):
    - 1.0 = accuracy value printed
    - 0.7 = no crash (run_error empty)
    - 0.5 = outlier handling present in script
    - 0.3 = NaN handling present in script
    - 0.0 = crashes
    """
    run_error = run_error or ""
    run_output = run_output or ""
    script = script or ""

    # Crash => 0.0
    if run_error.strip():
        return 0.0

    # Accuracy printed => 1.0
    if _contains_accuracy(run_output):
        return 1.0

    score = 0.0

    # No crash
    score = max(score, 0.7)

    # Evidence of outlier and NaN handling
    if _has_outlier_handling(script):
        score = max(score, 0.5)
    if _has_nan_handling(script):
        score = max(score, 0.3)

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, float(score)))
