import sys
import subprocess
import tempfile
import shutil
from pathlib import Path

import pandas as pd

from app.models import Action, Observation, Reward, StepResponse, StateResponse
from app.tasks import task1_easy, task2_medium, task3_hard
from app.graders import grader1, grader2, grader3
from app.reward import RewardEngine

class DataEngEnvironment:
    def __init__(self):
        self.current_task_id: int = 0
        self.step_number: int = 0
        self.current_script: str = ""
        self.dataframe: pd.DataFrame | None = None
        self.task_description: str = ""
        self.done: bool = False
        self.last_run_output: str = ""
        self.last_run_error: str | None = None
        
        # Privacy variables for Task 3 held-out sets
        self._held_out_X = None
        self._held_out_y = None
        self._full_df = None
        
        self._reward_engine = RewardEngine()

    def reset(self, task_id: int) -> Observation:
        self.current_task_id = task_id
        self.step_number = 0
        self.done = False
        self.last_run_output = ""
        self.last_run_error = None
        
        self._held_out_X = None
        self._held_out_y = None
        self._full_df = None
        
        self._reward_engine = RewardEngine()
        
        if task_id == 1:
            scenario = task1_easy.generate_scenario()
        elif task_id == 2:
            scenario = task2_medium.generate_scenario()
        elif task_id == 3:
            scenario = task3_hard.generate_scenario()
            # Strict assignment isolating testing datasets
            self._held_out_X = scenario.get("held_out_X")
            self._held_out_y = scenario.get("held_out_y")
            self._full_df = scenario.get("full_df")
        else:
            raise ValueError(f"Unknown task_id: {task_id}")
            
        self.current_script = scenario.get("broken_script", "")
        self.dataframe = scenario.get("dataframe")
        self.task_description = scenario.get("task_description", "")
        self.last_run_error = scenario.get("initial_error_log")

        return self._get_observation()
        
    def _get_observation(self, partial_schema: dict | None = None, partial_preview: str = "") -> Observation:
        return Observation(
            task_id=self.current_task_id,
            step_number=self.step_number,
            script_content=self.current_script,
            last_run_output=self.last_run_output,
            last_run_error=self.last_run_error,
            data_preview=partial_preview,
            schema_info=partial_schema or {},
            done=self.done
        )

    def step(self, action: Action) -> StepResponse:
        self.step_number += 1
        
        preview = ""
        schema = {}
        score = 0.0
        
        try:
            if action.action_type == "inspect_data":
                df = self.dataframe
                if df is not None:
                    # Build data_preview
                    data_preview = f"Shape: {df.shape}\n\n"
                    data_preview += f"Columns: {list(df.columns)}\n\n"
                    data_preview += f"Dtypes:\n{df.dtypes.to_string()}\n\n"
                    data_preview += f"Null counts:\n{df.isnull().sum().to_string()}\n\n"
                    data_preview += f"Sample (5 rows):\n{df.head().to_string()}\n\n"
                    data_preview += f"Numeric stats:\n{df.describe().to_string()}"
                    preview = data_preview
                    
                    # Build schema_info
                    schema_info = {
                        "columns": list(df.columns),
                        "dtypes": {col: str(df[col].dtype) for col in df.columns},
                        "null_counts": df.isnull().sum().to_dict(),
                        "shape": list(df.shape)
                    }
                    schema = schema_info
            
            elif action.action_type == "check_schema":
                df = self.dataframe
                if df is not None:
                    schema = df.dtypes.astype(str).to_dict()
                    
            elif action.action_type == "edit_script":
                old_text = action.payload.get("old", "")
                new_text = action.payload.get("new", "")
                if old_text:
                    if old_text not in self.current_script:
                        self.last_run_error = "edit_script failed: the text to replace was not found in the script"
                    else:
                        self.current_script = self.current_script.replace(old_text, new_text)
                
            elif action.action_type == "run_script":
                # Creating sandbox subprocess per contract requirements
                with tempfile.TemporaryDirectory() as tmpdir:
                    tmpdir_path = Path(tmpdir)
                    try:
                        if self.dataframe is not None:
                            self.dataframe.to_csv(tmpdir_path / "data.csv", index=False)
                            
                        script_path = tmpdir_path / "script.py"
                        sandbox_script = "import pandas as pd\ndf = pd.read_csv('data.csv')\n" + self.current_script
                        script_path.write_text(sandbox_script, encoding="utf-8")
                        
                        result = subprocess.run(
                            [sys.executable, str(script_path)],
                            capture_output=True,
                            text=True,
                            timeout=10,
                            cwd=tmpdir
                        )
                        self.last_run_output = result.stdout
                        self.last_run_error = result.stderr if result.returncode != 0 else None
                    except subprocess.TimeoutExpired as e:
                        self.last_run_error = f"TimeoutExpired: script ran longer than 10s."
                        self.last_run_output = ""
                    
            elif action.action_type == "submit":
                if self.current_task_id == 1:
                    score = grader1.grade(self.current_script, self.last_run_output, self.last_run_error)
                elif self.current_task_id == 2:
                    score = grader2.grade(self.current_script, self.last_run_output, self.last_run_error)
                elif self.current_task_id == 3:
                    score = grader3.grade(self.current_script, self._full_df)
                
                self.done = True
                
        except Exception as e:
            self.last_run_error = f"Action Handler Error: {str(e)}"
            
        # Maximum step bound limit restriction per project contract requirements
        if self.step_number >= 20:
            self.done = True
            
        obs = self._get_observation(partial_schema=schema, partial_preview=preview)
        
        # Obtain reward from Reward engine
        try:
            reward = self._reward_engine.compute_reward(
                task_id=self.current_task_id,
                action_type=action.action_type,
                action_payload=action.payload,
                observation=obs.model_dump(),
                grader_score=score,
                step_number=self.step_number
            )
        except Exception as e:
            # Fallback if RewardEngine throws error based on parameter changes
            reward = Reward(
                score=score, 
                partial_rewards={}, 
                message=f"Reward calculation fallback. {e}", 
                is_terminal=self.done
            )
            
        return StepResponse(observation=obs, reward=reward)

    def submit_grader(self, task_id: int) -> Reward:
        score = 0.0
        if task_id == 1:
            score = grader1.grade(self.current_script, self.last_run_output, self.last_run_error)
        elif task_id == 2:
            score = grader2.grade(self.current_script, self.last_run_output, self.last_run_error)
        elif task_id == 3:
            score = grader3.grade(self.current_script, self._full_df)
            
        obs = self._get_observation()
        
        try:
            reward = self._reward_engine.compute_reward(
                task_id=task_id,
                action_type="submit",
                action_payload={},
                observation=obs.model_dump(),
                grader_score=score,
                step_number=self.step_number
            )
        except Exception as e:
            reward = Reward(
                score=score, 
                partial_rewards={}, 
                message=f"Reward calculation fallback. {e}", 
                is_terminal=True
            )
        return reward

    def state(self) -> StateResponse:
        return StateResponse(
            task_id=self.current_task_id,
            step_number=self.step_number,
            max_steps=20,
            current_script=self.current_script,
            episode_done=self.done,
            task_description=self.task_description
        )
