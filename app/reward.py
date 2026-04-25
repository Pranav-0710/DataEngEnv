from __future__ import annotations

from typing import Dict, Any

from app.models import Reward


class RewardEngine:
    """Dense partial reward shaping aligned to the 4-stage cascade.

    Stage 1 — Data Repair:    rename age_years→age, dropna
    Stage 2 — Training Monitor: add StandardScaler before MLPClassifier
    Stage 3 — Eval Validation:  move scaler.fit after train_test_split
    Stage 4 — Deploy Gate:      add class_weight='balanced'
    """

    def __init__(self) -> None:
        self._state: Dict[int, Dict[str, Any]] = {}

    def _get_task_state(self, task_id: int) -> Dict[str, Any]:
        if task_id not in self._state:
            self._state[task_id] = {
                "awarded": {},
                "has_edited": False,
            }
        return self._state[task_id]

    def _add_partial(self, st: Dict[str, Any], key: str, value: float) -> bool:
        if key in st["awarded"]:
            return False
        st["awarded"][key] = float(value)
        return True

    def _payload_text(self, payload: dict) -> str:
        parts = []
        for v in (payload or {}).values():
            if isinstance(v, str):
                parts.append(v)
        return "\n".join(parts).lower()

    def _no_error(self, observation: dict) -> bool:
        err = (observation or {}).get("last_run_error")
        return err is None or str(err).strip() == ""

    def _compute_score(self, st: Dict[str, Any]) -> float:
        total = float(sum(st["awarded"].values()))
        return max(0.01, min(0.99, total))

    def compute_reward(
        self,
        task_id: int,
        action_type: str,
        action_payload: dict,
        observation: dict,
        grader_score: float,
        step_number: int,
    ) -> Reward:
        st = self._get_task_state(task_id)
        msg_parts = []
        action_type = (action_type or "").strip()
        payload_text = self._payload_text(action_payload)

        # ── query_actor — applies to ALL stages ──────────────────────
        if action_type == "query_actor":
            if self._add_partial(st, "actor_query", 0.1):
                msg_parts.append("+0.1 actor consulted")
            return Reward(
                score=self._compute_score(st),
                partial_rewards=dict(st["awarded"]),
                message=", ".join(msg_parts) if msg_parts else "actor already consulted",
                is_terminal=False,
            )

        # ── STAGE 1 — Data Repair ────────────────────────────────────
        if task_id == 1:
            if action_type in {"inspect_data", "check_schema"}:
                if self._add_partial(st, "s1_inspect", 0.1):
                    msg_parts.append("S1: +0.1 inspect/schema")

            if action_type == "edit_script":
                st["has_edited"] = True
                # Check for column rename fix (age_years → age)
                if "age" in payload_text and "age_years" not in payload_text:
                    if self._add_partial(st, "s1_rename_fix", 0.2):
                        msg_parts.append("S1: +0.2 column rename fix")
                # Check for NaN handling
                if "dropna" in payload_text or "fillna" in payload_text:
                    if self._add_partial(st, "s1_nan_fix", 0.1):
                        msg_parts.append("S1: +0.1 NaN handling added")

            if action_type == "run_script" and self._no_error(observation):
                if self._add_partial(st, "s1_clean_run", 0.2):
                    msg_parts.append("S1: +0.2 clean run")

        # ── STAGE 2 — Training Monitor (StandardScaler) ─────────────
        elif task_id == 2:
            if action_type == "inspect_data":
                if self._add_partial(st, "s2_inspect", 0.1):
                    msg_parts.append("S2: +0.1 data inspected")

            if action_type == "run_script":
                # Running reveals NaN loss / divergence
                if self._add_partial(st, "s2_run_diagnose", 0.1):
                    msg_parts.append("S2: +0.1 divergence observed")

            if action_type == "edit_script":
                st["has_edited"] = True
                # Check for StandardScaler addition
                if "standardscaler" in payload_text or "standard_scaler" in payload_text:
                    if self._add_partial(st, "s2_scaler_fix", 0.3):
                        msg_parts.append("S2: +0.3 StandardScaler added")
                # Also accept MinMaxScaler or normalize
                elif "minmaxscaler" in payload_text or "normalize" in payload_text:
                    if self._add_partial(st, "s2_scaler_fix", 0.2):
                        msg_parts.append("S2: +0.2 alternative scaler added")

            if action_type == "run_script" and st.get("has_edited") and self._no_error(observation):
                if self._add_partial(st, "s2_clean_run", 0.2):
                    msg_parts.append("S2: +0.2 clean run after fix")

        # ── STAGE 3 — Eval Validation (Data Leakage) ────────────────
        elif task_id == 3:
            if action_type == "inspect_data":
                if self._add_partial(st, "s3_inspect", 0.1):
                    msg_parts.append("S3: +0.1 data inspected")

            if action_type == "run_script":
                # Running reveals suspicious 98% accuracy
                if self._add_partial(st, "s3_run_suspicious", 0.1):
                    msg_parts.append("S3: +0.1 suspicious accuracy observed")

            if action_type == "edit_script":
                st["has_edited"] = True
                txt = payload_text
                # Detect: train_test_split appears before scaler.fit
                idx_split = txt.find("train_test_split")
                idx_fit = min(
                    [i for i in [txt.find("scaler.fit("), txt.find("fit_transform(")] if i != -1]
                    or [-1]
                )
                if idx_split != -1 and idx_fit != -1 and idx_split < idx_fit:
                    if self._add_partial(st, "s3_leakage_fix", 0.3):
                        msg_parts.append("S3: +0.3 fit moved after split")
                elif idx_split != -1 or idx_fit != -1:
                    if self._add_partial(st, "s3_partial_edit", 0.1):
                        msg_parts.append("S3: +0.1 relevant edit attempt")

            if action_type == "run_script" and st.get("has_edited") and self._no_error(observation):
                if self._add_partial(st, "s3_clean_run", 0.2):
                    msg_parts.append("S3: +0.2 clean run after fix")

            if action_type == "submit" and not st.get("has_edited"):
                self._add_partial(st, "s3_submit_no_edit", -0.2)
                msg_parts.append("S3: -0.2 submitted without editing")

        # ── STAGE 4 — Deploy Gate (Fairness) ─────────────────────────
        elif task_id == 4:
            if action_type == "inspect_data":
                if self._add_partial(st, "s4_inspect", 0.1):
                    msg_parts.append("S4: +0.1 data inspected")

            if action_type == "edit_script":
                st["has_edited"] = True
                payload_str = str(action_payload)
                if "class_weight" in payload_str:
                    if self._add_partial(st, "s4_fairness_fix", 0.3):
                        msg_parts.append("S4: +0.3 class_weight balanced added")
                elif "stratify" in payload_str:
                    if self._add_partial(st, "s4_stratify", 0.1):
                        msg_parts.append("S4: +0.1 stratified sampling")
                else:
                    if self._add_partial(st, "s4_generic_edit", 0.05):
                        msg_parts.append("S4: +0.05 script edited")

            if action_type == "run_script":
                if self._add_partial(st, "s4_run", 0.1):
                    msg_parts.append("S4: +0.1 script executed")

        # ── Terminal / Score ──────────────────────────────────────────
        is_terminal = action_type == "submit"
        if is_terminal:
            total_score = float(grader_score)
        else:
            total_score = self._compute_score(st)

        total_score = max(0.0, min(1.0, total_score))

        message = ", ".join(msg_parts) if msg_parts else "no partial reward"
        cumulative_partials = {k: float(v) for k, v in st["awarded"].items()}

        return Reward(
            score=float(total_score),
            partial_rewards=cumulative_partials,
            message=message,
            is_terminal=is_terminal,
        )
