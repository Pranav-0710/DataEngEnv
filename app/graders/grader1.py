from __future__ import annotations

import re
from typing import Optional


def _contains_accuracy(output: str) -> bool:
    return bool(re.search(r"Accuracy\s*:\s*\d*\.?\d+", output, flags=re.IGNORECASE))


def grade(script: str, run_output: str, run_error: Optional[str]) -> float:
    """
    Task 1 Grader

    Scoring (deterministic, pick highest satisfied level):
    - 1.0 = accuracy printed
    - 0.7 = no error (run_error is empty/None)
    - 0.4 = check_schema was used (in script or output)
    - 0.0 = KeyError still present
    """
    run_error = run_error or ""
    run_output = run_output or ""
    script = script or ""

    # Hard failure: KeyError still present
    if "KeyError" in run_error or "age_years" in (run_error + run_output):
        return 0.0

    # Top level: printed accuracy
    if _contains_accuracy(run_output):
        return 1.0

    # No error
    if run_error.strip() == "":
        # Award 0.7 for clean run
        base = 0.7
    else:
        base = 0.0

    # Evidence of schema checking in the code or output
    if "check_schema" in script or "check_schema" in run_output:
        base = max(base, 0.4)

    # Clamp to [0.0, 1.0]
    return max(0.0, min(1.0, float(base)))
