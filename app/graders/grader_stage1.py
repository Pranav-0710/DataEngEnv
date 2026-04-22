from __future__ import annotations

import re
from typing import Optional


def _contains_accuracy(output: str) -> bool:
    """Check if output contains a printed accuracy value."""
    return bool(re.search(r"Accuracy\s*:\s*\d*\.?\d+", output, flags=re.IGNORECASE))


def _has_nan_handling(script: str) -> bool:
    """Detect common NaN/missing-value handling patterns in script."""
    s = script.lower()
    return any(
        k in s for k in ["dropna", "fillna", "simpleimputer", "isna()", "notna()"]
    )


def _has_outlier_handling(script: str) -> bool:
    """Detect common outlier handling patterns in script."""
    s = script.lower()
    return any(
        k in s for k in ["clip(", "quantile(", "iqr", "winsor", "robustscaler"]
    )


def grade(script: str, run_output: str, run_error: Optional[str]) -> float:
    """
    Stage 1 Grader — Data Repair.

    Scoring ladder (deterministic, string-based only, no script execution):
      0.0 → KeyError still present (age_years in script)
      0.2 → age_years fixed but NaN error still occurs
      0.4 → script contains NaN handling (dropna/fillna/etc.)
      0.6 → script contains outlier handling (clip/quantile/etc.)
      0.8 → script runs without any error
      1.0 → script runs clean AND prints accuracy value

    Checks from top (1.0) downward, returns first match.
    """
    run_error = run_error or ""
    run_output = run_output or ""
    script = script or ""

    # ── Hard failure: column rename bug still present ──
    if "age_years" in script:
        return 0.0

    # ── Top tier: clean run with accuracy printed ──
    if run_error.strip() == "" and _contains_accuracy(run_output):
        return 1.0

    # ── Clean run but no accuracy printed ──
    if run_error.strip() == "":
        return 0.8

    # ── Script still has errors, but check for fix progress ──

    # Outlier handling present (implies NaN handling also likely attempted)
    if _has_outlier_handling(script) and _has_nan_handling(script):
        return 0.6

    # NaN handling present
    if _has_nan_handling(script):
        return 0.4

    # Column rename fixed but script still crashes (NaN error)
    return 0.2
