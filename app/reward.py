from __future__ import annotations

from typing import Dict, Any

from app.models import Reward


class RewardEngine:
    """Dense semantic reward shaping for 4-stage cascade.

    Key principle: reward is based on RESULTING SCRIPT STATE,
    not exact payload text. This lets any valid fix earn reward.
    """

    def __init__(self) -> None:
        self._state: Dict[int, Dict[str, Any]] = {}

    def _get_task_state(self, task_id: int) -> Dict[str, Any]:
        if task_id not in self._state:
            self._state[task_id] = {
                "awarded": {},
                "action_history": [],
            }
        return self._state[task_id]

    def _add_partial(self, st: Dict[str, Any], key: str, value: float) -> bool:
        if key in st["awarded"]:
            return False
        st["awarded"][key] = float(value)
        return True

    def _compute_score(self, st: Dict[str, Any]) -> float:
        total = float(sum(st["awarded"].values()))
        return max(0.0, min(0.99, total))

    def _repeat_penalty(self, st: Dict[str, Any], action_type: str) -> float:
        """Small penalty for repeating the same action consecutively."""
        history = st["action_history"]
        if len(history) >= 2 and history[-1] == action_type and history[-2] == action_type:
            return -0.03
        return 0.0

    def compute_reward(
        self,
        task_id: int,
        action_type: str,
        action_payload: dict,
        observation: dict,
        grader_score: float,
        step_number: int,
        current_script: str = "",
    ) -> Reward:
        st = self._get_task_state(task_id)
        msg_parts = []
        action_type = (action_type or "").strip()

        # Track action history for repeat penalty
        st["action_history"].append(action_type)

        # Check for repeat penalty
        penalty = self._repeat_penalty(st, action_type)
        if penalty < 0:
            self._add_partial(st, f"repeat_penalty_{step_number}", penalty)
            msg_parts.append(f"{penalty:.2f} repeated action")

        script_lower = current_script.lower()

        # ── query_actor — all stages ─────────────────────────────
        if action_type == "query_actor":
            if self._add_partial(st, "actor_query", 0.1):
                msg_parts.append("+0.1 actor consulted")
            return Reward(
                score=self._compute_score(st),
                partial_rewards=dict(st["awarded"]),
                message=", ".join(msg_parts) or "actor already consulted",
                is_terminal=False,
            )

        # ── STAGE 1 — Data Repair ────────────────────────────────
        if task_id == 1:
            if action_type in {"inspect_data", "check_schema"}:
                if self._add_partial(st, "s1_inspect", 0.1):
                    msg_parts.append("S1: +0.1 inspect/schema")

            # Semantic check on RESULTING SCRIPT after edit
            if action_type == "edit_script":
                # Column fix: age_years no longer in script
                if "age_years" not in script_lower and "age" in script_lower:
                    if self._add_partial(st, "s1_column_fix", 0.2):
                        msg_parts.append("S1: +0.2 column bug fixed")
                # NaN handling present in script
                if any(kw in script_lower for kw in ["dropna", "fillna", "simpleimputer", "isna", "isnull"]):
                    if self._add_partial(st, "s1_nan_fix", 0.15):
                        msg_parts.append("S1: +0.15 NaN handling added")
                # Outlier handling present in script
                if any(kw in script_lower for kw in ["clip", "quantile", "robustscaler", "winsor"]):
                    if self._add_partial(st, "s1_outlier_fix", 0.15):
                        msg_parts.append("S1: +0.15 outlier handling added")

            # Clean run
            no_error = observation.get("last_run_error") is None or str(observation.get("last_run_error", "")).strip() == ""
            if action_type == "run_script" and no_error:
                if self._add_partial(st, "s1_clean_run", 0.2):
                    msg_parts.append("S1: +0.2 clean run")

        # ── STAGE 2 — Training Monitor (StandardScaler) ─────────
        elif task_id == 2:
            if action_type in {"inspect_data", "check_schema"}:
                if self._add_partial(st, "s2_inspect", 0.1):
                    msg_parts.append("S2: +0.1 inspected")

            if action_type == "run_script":
                if self._add_partial(st, "s2_run_diagnose", 0.1):
                    msg_parts.append("S2: +0.1 divergence observed")

            # Semantic: does script now contain a scaler?
            if action_type == "edit_script":
                if any(kw in script_lower for kw in ["standardscaler", "minmaxscaler", "robustscaler", "normalize"]):
                    if self._add_partial(st, "s2_scaler_fix", 0.3):
                        msg_parts.append("S2: +0.3 scaler added")

            # Clean run after edit
            no_error = observation.get("last_run_error") is None or str(observation.get("last_run_error", "")).strip() == ""
            if action_type == "run_script" and "s2_scaler_fix" in st["awarded"] and no_error:
                if self._add_partial(st, "s2_clean_run", 0.2):
                    msg_parts.append("S2: +0.2 clean run after fix")

        # ── STAGE 3 — Eval Validation (Data Leakage) ────────────
        elif task_id == 3:
            if action_type in {"inspect_data", "check_schema"}:
                if self._add_partial(st, "s3_inspect", 0.1):
                    msg_parts.append("S3: +0.1 inspected")

            if action_type == "run_script":
                if self._add_partial(st, "s3_run_suspicious", 0.1):
                    msg_parts.append("S3: +0.1 suspicious accuracy observed")

            # Semantic: is fit/fit_transform now AFTER train_test_split?
            if action_type == "edit_script":
                idx_split = script_lower.find("train_test_split")
                fit_positions = [script_lower.find("scaler.fit("), script_lower.find("fit_transform(")]
                fit_positions = [p for p in fit_positions if p != -1]
                idx_fit = min(fit_positions) if fit_positions else -1

                if idx_split != -1 and idx_fit != -1 and idx_split < idx_fit:
                    if self._add_partial(st, "s3_leakage_fix", 0.3):
                        msg_parts.append("S3: +0.3 leakage fixed (fit after split)")
                elif idx_split != -1:
                    if self._add_partial(st, "s3_partial_edit", 0.05):
                        msg_parts.append("S3: +0.05 relevant edit attempt")

            no_error = observation.get("last_run_error") is None or str(observation.get("last_run_error", "")).strip() == ""
            if action_type == "run_script" and "s3_leakage_fix" in st["awarded"] and no_error:
                if self._add_partial(st, "s3_clean_run", 0.2):
                    msg_parts.append("S3: +0.2 clean run after fix")

            if action_type == "submit" and "s3_leakage_fix" not in st["awarded"]:
                self._add_partial(st, "s3_no_edit_penalty", -0.2)
                msg_parts.append("S3: -0.2 submitted without fixing leakage")

        # ── STAGE 4 — Deploy Gate (Fairness) ─────────────────────
        elif task_id == 4:
            if action_type == "inspect_data":
                if self._add_partial(st, "s4_inspect", 0.1):
                    msg_parts.append("S4: +0.1 inspected")

            # Semantic: does script now contain class_weight?
            if action_type == "edit_script":
                if "class_weight" in script_lower:
                    if self._add_partial(st, "s4_fairness_fix", 0.3):
                        msg_parts.append("S4: +0.3 fairness fix applied")
                elif "stratify" in script_lower:
                    if self._add_partial(st, "s4_stratify", 0.1):
                        msg_parts.append("S4: +0.1 stratified sampling")
                else:
                    if self._add_partial(st, "s4_generic_edit", 0.05):
                        msg_parts.append("S4: +0.05 script edited")

            if action_type == "run_script":
                if self._add_partial(st, "s4_run", 0.1):
                    msg_parts.append("S4: +0.1 script executed")

        # ── Score ─────────────────────────────────────────────────
        is_terminal = action_type == "submit"
        total_score = float(grader_score) if is_terminal else self._compute_score(st)
        total_score = max(0.0, min(1.0, total_score))

        message = ", ".join(msg_parts) if msg_parts else "no partial reward"

        return Reward(
            score=float(total_score),
            partial_rewards=dict(st["awarded"]),
            message=message,
            is_terminal=is_terminal,
        )
