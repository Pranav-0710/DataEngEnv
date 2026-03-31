from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Optional

import pandas as pd

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


def _parse_accuracy(output: str) -> Optional[float]:
    # Parse last occurrence of Accuracy: <number>
    matches = re.findall(r"Accuracy\s*:\s*([0-9]*\.?[0-9]+)", output, flags=re.IGNORECASE)
    if not matches:
        return None
    try:
        return float(matches[-1])
    except ValueError:
        return None


def grade(script: str, full_df: pd.DataFrame) -> float:
    """
    Task 3 Grader (executes candidate script in a subprocess sandbox with the full dataset).
    """
    fit_after_split = _detect_fit_after_split(script or "")
    
    with TemporaryDirectory() as td:
        tmpdir = Path(td)
        csv_path = tmpdir / "data.csv"
        
        try:
            full_df.to_csv(csv_path, index=False)
        except Exception:
            return 0.0
            
        prelude = (
            "import pandas as pd\n"
            f"df = pd.read_csv(r'{csv_path.as_posix()}')\n"
        )
        
        code = prelude + "\n" + (script or "") + "\n"
        script_path = tmpdir / "candidate.py"
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
            return 0.0
        except Exception:
            return 0.0
            
    stdout = proc.stdout or ""
    stderr = proc.stderr or ""
    
    # Check execution success
    if proc.returncode != 0 or stderr.strip():
        # Score 0.0 on error OR if fit_after_split is also missing
        return max(0.0, 0.3 if fit_after_split else 0.0)
        
    # cumulative scoring
    score = 0.5 # starts at 0.5 because it ran cleanly (0.3 + 0.2 cumulative as requested)
    if not fit_after_split:
        return 0.3 # Max 0.3 if fit_after_split is false
        
    acc = _parse_accuracy(stdout)
    if acc is None or acc < 0.60:
        return 0.0 
        
    if acc >= 0.75:
        score = 1.0
    elif acc >= 0.70:
        score = max(score, 0.75)
        
    return float(score)
