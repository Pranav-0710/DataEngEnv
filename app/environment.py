from __future__ import annotations

import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pandas as pd

from app.actors.code_reviewer import CodeReviewer
from app.actors.mlops_bot import MLOpsBot
from app.graders import grader_stage1, grader_stage2, grader_stage3, grader_stage4
from app.models import Action, Observation, Reward, StateResponse, StepResponse
from app.reward import RewardEngine
from app.tasks import (
    stage1_data_repair,
    stage2_training_monitor,
    stage3_eval_validation,
    stage4_deploy_gate,
)


class DataEngEnvironment:
    def __init__(self):
        self.current_task_id: int = 1
        self.step_number: int = 0
        self.current_stage: int = 1
        self.stage_step_number: int = 0
        self.loop_count: int = 0
        self.stages_completed: list[int] = []
        self.episode_score: float = 0.0
        self.actor_feedback: str = ""

        self.current_script: str = ""
        self.dataframe: pd.DataFrame | None = None
        self.task_description: str = ""
        self.done: bool = False
        self.last_run_output: str = ""
        self.last_run_error: str | None = None

        self._held_out_X = None
        self._held_out_y = None
        self._full_df = None
        self._deploy_conditions = None

        self._code_reviewer = CodeReviewer()
        self._mlops_bot = MLOpsBot()
        self._reward_engine = RewardEngine()

    def reset(self, task_id: int | None = None) -> Observation:
        start_stage = task_id if task_id in {1, 2, 3, 4} else 1
        self.current_task_id = start_stage
        self.step_number = 0
        self.current_stage = start_stage
        self.stage_step_number = 0
        self.loop_count = 0
        self.stages_completed = []
        self.episode_score = 0.0
        self.actor_feedback = ""
        self._reward_engine = RewardEngine()
        self.done = False
        self.last_run_output = ""
        self.last_run_error = None
        self._load_stage(start_stage)
        return self._get_observation()

    def _load_stage(self, stage: int) -> None:
        self.current_stage = stage
        self.current_task_id = stage
        self.stage_step_number = 0
        self.actor_feedback = ""
        self.last_run_output = ""

        self._held_out_X = None
        self._held_out_y = None
        self._full_df = None
        self._deploy_conditions = None

        if stage == 1:
            scenario = stage1_data_repair.generate_scenario()
        elif stage == 2:
            scenario = stage2_training_monitor.generate_scenario()
        elif stage == 3:
            scenario = stage3_eval_validation.generate_scenario()
            self._held_out_X = scenario.get("held_out_X")
            self._held_out_y = scenario.get("held_out_y")
            self._full_df = scenario.get("full_df")
        elif stage == 4:
            scenario = stage4_deploy_gate.generate_scenario()
            self._deploy_conditions = scenario.get("deploy_conditions")
        else:
            raise ValueError(f"Unknown stage: {stage}")

        self.current_script = scenario.get("broken_script", "")
        self.dataframe = scenario.get("dataframe")
        self.task_description = scenario.get("task_description", "")
        self.last_run_error = scenario.get("initial_error_log")

    def _get_observation(
        self,
        partial_schema: dict | None = None,
        partial_preview: str = "",
    ) -> Observation:
        return Observation(
            task_id=self.current_task_id,
            step_number=self.step_number,
            script_content=self.current_script,
            last_run_output=self.last_run_output,
            last_run_error=self.last_run_error,
            data_preview=partial_preview,
            schema_info=partial_schema or {},
            done=self.done,
            current_stage=self.current_stage,
            stage_step_number=self.stage_step_number,
            actor_feedback=self.actor_feedback,
            loop_count=self.loop_count,
            stages_completed=list(self.stages_completed),
        )

    def _make_reward(self, score: float, message: str, is_terminal: bool) -> Reward:
        return Reward(
            score=float(max(0.0, min(1.0, score))),
            partial_rewards={
                "episode_score": float(max(0.0, min(1.0, self.episode_score))),
                "current_stage": float(self.current_stage),
            },
            message=message,
            is_terminal=is_terminal,
        )

    def _run_current_script(self) -> None:
        runs_dir = Path.cwd() / ".env_runs"
        runs_dir.mkdir(exist_ok=True)
        tmpdir = runs_dir / f"stage_{self.current_stage}_{uuid.uuid4().hex}"
        tmpdir.mkdir()

        def cleanup() -> None:
            shutil.rmtree(tmpdir, ignore_errors=True)

        try:
            if self.dataframe is not None:
                self.dataframe.to_csv(tmpdir / "data.csv", index=False)

            script_path = tmpdir / "script.py"
            sandbox_script = (
                "import pandas as pd\n"
                "df = pd.read_csv('data.csv')\n"
                + self.current_script
            )
            script_path.write_text(sandbox_script, encoding="utf-8")

            result = subprocess.run(
                [sys.executable, str(script_path)],
                capture_output=True,
                text=True,
                timeout=10,
                cwd=tmpdir,
                check=False,
            )
            self.last_run_output = result.stdout
            self.last_run_error = result.stderr if result.returncode != 0 else None
        except subprocess.TimeoutExpired:
            self.last_run_error = "TimeoutExpired: script ran longer than 10s."
            self.last_run_output = ""
        finally:
            cleanup()

    def _grade_current_stage(self) -> float:
        if self.current_stage == 1:
            return grader_stage1.grade(
                self.current_script,
                self.last_run_output,
                self.last_run_error,
            )
        if self.current_stage == 2:
            return grader_stage2.grade(
                self.current_script,
                self.last_run_output,
                self.last_run_error,
            )
        if self.current_stage == 3:
            return grader_stage3.grade(
                self.current_script,
                self._full_df,
                self._held_out_X,
                self._held_out_y,
            )
        if self.current_stage == 4:
            return grader_stage4.grade(
                self.current_script,
                self.dataframe,
                self._deploy_conditions,
            )
        return 0.0

    def _check_stage_transition(self, score: float) -> str:
        stage = self.current_stage

        if score >= 0.7:
            if stage not in self.stages_completed:
                self.stages_completed.append(stage)
            self.episode_score = max(0.0, min(1.0, self.episode_score + score * 0.25))

            if stage == 4:
                self.done = True
                return "Stage 4 complete. Deployment approved."

            self._load_stage(stage + 1)
            return f"Stage {stage} complete. Advanced to Stage {self.current_stage}."

        if stage == 4:
            self.loop_count += 1
            self.episode_score = max(0.0, self.episode_score - 0.3)
            if self.loop_count >= 2:
                self.done = True
                return "Stage 4 failed twice. Episode terminated."

            self._load_stage(1)
            return "Stage 4 failed. Looping back to Stage 1."

        return f"Stage {stage} score below threshold. Continue fixing this stage."

    def _handle_query_actor(self) -> None:
        if self.current_stage == 1:
            self.actor_feedback = "No reviewer assigned to this stage."
        elif self.current_stage in {2, 3}:
            self.actor_feedback = self._code_reviewer.review(
                self.current_script,
                stage=self.current_stage,
            )
        elif self.current_stage == 4:
            self.actor_feedback = self._mlops_bot.check_deployment(
                self.current_script,
                self.dataframe,
                self._deploy_conditions,
            )

    def step(self, action: Action) -> StepResponse:
        if self.done:
            obs = self._get_observation()
            return StepResponse(
                observation=obs,
                reward=self._make_reward(self.episode_score, "episode already complete", True),
            )

        self.step_number += 1
        self.stage_step_number += 1

        preview = ""
        schema = {}
        score = 0.0
        message = "no reward"

        try:
            if action.action_type == "inspect_data":
                df = self.dataframe
                if df is not None:
                    preview = (
                        f"Shape: {df.shape}\n\n"
                        f"Columns: {list(df.columns)}\n\n"
                        f"Dtypes:\n{df.dtypes.to_string()}\n\n"
                        f"Null counts:\n{df.isnull().sum().to_string()}\n\n"
                        f"Sample (5 rows):\n{df.head().to_string()}\n\n"
                        f"Numeric stats:\n{df.describe().to_string()}"
                    )
                    schema = {
                        "columns": list(df.columns),
                        "dtypes": {col: str(df[col].dtype) for col in df.columns},
                        "null_counts": df.isnull().sum().to_dict(),
                        "shape": list(df.shape),
                    }
                message = "data inspected"

            elif action.action_type == "check_schema":
                if self.dataframe is not None:
                    schema = self.dataframe.dtypes.astype(str).to_dict()
                message = "schema checked"

            elif action.action_type == "edit_script":
                old_text = action.payload.get("old", "")
                new_text = action.payload.get("new", "")
                replacement_script = action.payload.get("script")
                if isinstance(replacement_script, str):
                    self.current_script = replacement_script
                    self.last_run_error = None
                elif old_text:
                    if old_text not in self.current_script:
                        self.last_run_error = "edit_script failed: the text to replace was not found in the script"
                    else:
                        self.current_script = self.current_script.replace(old_text, new_text)
                        self.last_run_error = None
                message = "script edited"

            elif action.action_type == "run_script":
                self._run_current_script()
                message = "script ran" if not self.last_run_error else "script failed"

            elif action.action_type == "query_actor":
                self._handle_query_actor()
                message = "actor queried"

            elif action.action_type == "submit":
                score = self._grade_current_stage()
                message = self._check_stage_transition(score)

        except Exception as exc:
            self.last_run_error = f"Action Handler Error: {exc}"
            message = self.last_run_error

        if self.step_number >= 60 or self.stage_step_number >= 15:
            self.done = True
            message = f"{message} Stage or episode step limit reached."

        obs = self._get_observation(partial_schema=schema, partial_preview=preview)

        # Use RewardEngine for dense partial rewards on every action
        shaped_reward = self._reward_engine.compute_reward(
            task_id=self.current_stage,
            action_type=action.action_type,
            action_payload=action.payload or {},
            observation={
                "last_run_error": self.last_run_error,
                "data_preview": preview,
            },
            grader_score=score,
            step_number=self.step_number,
        )

        # For submit, override score but keep partial breakdown
        if action.action_type == "submit":
            final_score = self.episode_score if self.done else score
            shaped_reward = Reward(
                score=float(max(0.0, min(1.0, final_score))),
                partial_rewards={
                    **shaped_reward.partial_rewards,
                    "episode_score": float(max(0.0, min(1.0, self.episode_score))),
                    "current_stage": float(self.current_stage),
                },
                message=message,
                is_terminal=self.done,
            )

        return StepResponse(
            observation=obs,
            reward=shaped_reward,
        )

    def submit_grader(self, task_id: int | None = None) -> Reward:
        score = self._grade_current_stage()
        message = self._check_stage_transition(score)
        return self._make_reward(self.episode_score if self.done else score, message, self.done)

    def state(self) -> StateResponse:
        return StateResponse(
            task_id=self.current_task_id,
            step_number=self.step_number,
            max_steps=60,
            current_script=self.current_script,
            episode_done=self.done,
            task_description=self.task_description,
            current_stage=self.current_stage,
            stage_step_number=self.stage_step_number,
            loop_count=self.loop_count,
            stages_completed=list(self.stages_completed),
            episode_score=float(max(0.0, min(1.0, self.episode_score))),
            actor_feedback=self.actor_feedback,
        )
