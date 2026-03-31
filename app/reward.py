from __future__ import annotations

from typing import Dict, Any

from app.models import Reward


class RewardEngine:
    def __init__(self) -> None:
        # Episode-scoped memory per task_id
        self._state: Dict[int, Dict[str, Any]] = {}

    def _get_task_state(self, task_id: int) -> Dict[str, Any]:
        if task_id not in self._state:
            self._state[task_id] = {
                "awarded": {},  # partial_id -> float
                "has_edited": False,
                "last_edit_step": None,
            }
        return self._state[task_id]

    def _add_partial(self, st: Dict[str, Any], key: str, value: float) -> bool:
        # Only add if not already awarded
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

    def _preview_text(self, observation: dict) -> str:
        return str((observation or {}).get("data_preview", ""))

    def _compute_score(self, st: Dict[str, Any]) -> float:
        total = float(sum(st["awarded"].values()))
        # Cap cumulative to [0.0, 1.0] pre-terminal
        if total < 0.0:
            total = 0.0
        if total > 1.0:
            total = 1.0
        return total

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
        awarded_now: Dict[str, float] = {}
        action_type = (action_type or "").strip()
        payload_text = self._payload_text(action_payload)
        preview = self._preview_text(observation)
        msg_parts = []

        # TASK 1
        if task_id == 1:
            if action_type in {"inspect_data", "check_schema"}:
                if self._add_partial(st, "t1_inspect_or_schema", 0.1):
                    awarded_now["t1_inspect_or_schema"] = 0.1
                    msg_parts.append("T1: +0.1 inspect/check_schema")

            if action_type == "edit_script":
                if ("age_years" not in payload_text) and ("age" in payload_text):
                    if self._add_partial(st, "t1_edit_fix", 0.3):
                        awarded_now["t1_edit_fix"] = 0.3
                        msg_parts.append("T1: +0.3 edit replaced age_years -> age")

            if action_type == "run_script" and self._no_error(observation):
                if self._add_partial(st, "t1_run_clean", 0.2):
                    awarded_now["t1_run_clean"] = 0.2
                    msg_parts.append("T1: +0.2 run without error")

        # TASK 2
        elif task_id == 2:
            if action_type == "inspect_data":
                if ("nan" in preview.lower()) and self._add_partial(st, "t2_inspect_nan", 0.15):
                    awarded_now["t2_inspect_nan"] = 0.15
                    msg_parts.append("T2: +0.15 NaN counts revealed")
                if ("9999999" in preview or "outlier" in preview.lower()) and self._add_partial(
                    st, "t2_inspect_outlier", 0.15
                ):
                    awarded_now["t2_inspect_outlier"] = 0.15
                    msg_parts.append("T2: +0.15 outlier revealed")

            if action_type == "edit_script":
                if ("dropna" in payload_text) or ("fillna" in payload_text):
                    if self._add_partial(st, "t2_edit_nan", 0.3):
                        awarded_now["t2_edit_nan"] = 0.3
                        msg_parts.append("T2: +0.3 dropna/fillna present")
                if ("clip(" in payload_text) or ("quantile(" in payload_text):
                    if self._add_partial(st, "t2_edit_outlier", 0.2):
                        awarded_now["t2_edit_outlier"] = 0.2
                        msg_parts.append("T2: +0.2 clip/quantile present")

            if action_type == "run_script" and self._no_error(observation):
                if self._add_partial(st, "t2_run_clean", 0.2):
                    awarded_now["t2_run_clean"] = 0.2
                    msg_parts.append("T2: +0.2 run without error")

        # TASK 3
        elif task_id == 3:
            if action_type == "inspect_data":
                if self._add_partial(st, "t3_inspect", 0.1):
                    awarded_now["t3_inspect"] = 0.1
                    msg_parts.append("T3: +0.1 inspect")

            if action_type == "run_script":
                if self._add_partial(st, "t3_run", 0.2):
                    awarded_now["t3_run"] = 0.2
                    msg_parts.append("T3: +0.2 run_script")
                if st.get("has_edited") and self._no_error(observation):
                    if self._add_partial(st, "t3_run_clean_post_edit", 0.2):
                        awarded_now["t3_run_clean_post_edit"] = 0.2
                        msg_parts.append("T3: +0.2 clean run after edit")

            if action_type == "edit_script":
                st["has_edited"] = True
                st["last_edit_step"] = step_number
                txt = payload_text
                # Detect move: train_test_split occurs before scaler.fit or fit_transform (or general fit() fallback)
                idx_split = txt.find("train_test_split")
                idx_fit_scaler = min([i for i in [txt.find("scaler.fit("), txt.find("scaler.fit_transform(")] if i != -1] or [-1])
                idx_fit_any = txt.find("fit(")
                moved = False
                if idx_split != -1 and idx_fit_scaler != -1:
                    moved = idx_split < idx_fit_scaler
                elif idx_split != -1 and idx_fit_any != -1:
                    moved = idx_split < idx_fit_any
                if moved and self._add_partial(st, "t3_edit_fix", 0.3):
                    awarded_now["t3_edit_fix"] = 0.3
                    msg_parts.append("T3: +0.3 fit after split detected")

            if action_type == "submit":
                if not st.get("has_edited"):
                    # Record the penalty in partials, but terminal score is grader_score
                    self._add_partial(st, "t3_submit_without_edit", -0.2)
                    if "t3_submit_without_edit" not in awarded_now:
                        awarded_now["t3_submit_without_edit"] = -0.2
                        msg_parts.append("T3: -0.2 submit without edit")

        # Determine terminal and score
        is_terminal = action_type == "submit"
        if is_terminal:
            total_score = float(grader_score)
        else:
            total_score = self._compute_score(st)

        # Clamp final score to [0.0, 1.0]
        if total_score < 0.0:
            total_score = 0.0
        if total_score > 1.0:
            total_score = 1.0

        # Prepare message and cumulative partials
        message = ", ".join(msg_parts) if msg_parts else "no reward"
        cumulative_partials = {k: float(v) for k, v in st["awarded"].items()}

        return Reward(
            score=float(total_score),
            partial_rewards=cumulative_partials,
            message=message,
            is_terminal=is_terminal,
        )

