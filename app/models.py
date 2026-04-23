from pydantic import BaseModel, Field, field_validator

class Action(BaseModel):
    action_type: str = Field(description="One of: inspect_data | run_script | edit_script | check_schema | query_actor | submit")
    payload: dict = Field(description="Action payload details like patch content etc.")

    @field_validator("action_type")
    @classmethod
    def validate_action_type(cls, v: str) -> str:
        allowed = {"inspect_data", "run_script", "edit_script", "check_schema", "query_actor", "submit"}
        if v not in allowed:
            raise ValueError(f"action_type must be one of {allowed}")
        return v

class Observation(BaseModel):
    task_id: int = Field(description="Current task ID")
    step_number: int = Field(description="Current step number")
    script_content: str = Field(description="Content of the current script")
    last_run_output: str = Field(description="Standard output from the last script run")
    last_run_error: str | None = Field(description="Standard error from the last script run execution, or None if successful")
    data_preview: str = Field(description="Preview of the current Pandas dataframe")
    schema_info: dict = Field(description="Column types and info")
    done: bool = Field(description="Whether the environment episode is complete")
    current_stage: int = Field(default=1, description="Current pipeline stage")
    stage_step_number: int = Field(default=0, description="Step count within current stage")
    actor_feedback: str = Field(default="", description="Most recent actor bot feedback")
    loop_count: int = Field(default=0, description="Number of deployment failure loops")
    stages_completed: list[int] = Field(default_factory=list, description="Completed pipeline stages")

class Reward(BaseModel):
    score: float = Field(description="Reward score from 0.0 to 1.0")
    partial_rewards: dict[str, float] = Field(description="Dictionary breaking down components of the score")
    message: str = Field(description="Feedback message strings")
    is_terminal: bool = Field(description="Whether the step resulting in this reward is final")

    @field_validator("score")
    @classmethod
    def validate_score(cls, v: float) -> float:
        if not (0.0 <= v <= 1.0):
            raise ValueError("Score must be between 0.0 and 1.0 inclusive")
        return v

class StepResponse(BaseModel):
    observation: Observation = Field(description="The observation after applying the action")
    reward: Reward = Field(description="The reward for applying the action")

class StateResponse(BaseModel):
    task_id: int = Field(description="ID of the task")
    step_number: int = Field(description="Current step number")
    max_steps: int = Field(description="Max steps allowed")
    current_script: str = Field(description="Content of the active python script")
    episode_done: bool = Field(description="Whether the episode is done")
    task_description: str = Field(description="Task objective description")
    current_stage: int = Field(default=1, description="Current pipeline stage")
    stage_step_number: int = Field(default=0, description="Step count within current stage")
    loop_count: int = Field(default=0, description="Number of deployment failure loops")
    stages_completed: list[int] = Field(default_factory=list, description="Completed pipeline stages")
    episode_score: float = Field(default=0.0, description="Cumulative 4-stage episode score")
    actor_feedback: str = Field(default="", description="Most recent actor bot feedback")

class TaskInfo(BaseModel):
    task_id: int = Field(description="ID of the task")
    name: str = Field(description="Task Name")
    difficulty: str = Field(description="Task difficulty: easy, medium, or hard")
    description: str = Field(description="Detailed description of the dataset and goal")
    action_schema: dict = Field(description="Schema specifying allowable actions and payloads")

    @field_validator("difficulty")
    @classmethod
    def validate_difficulty(cls, v: str) -> str:
        allowed = {"easy", "medium", "hard"}
        if v not in allowed:
            raise ValueError(f"difficulty must be one of {allowed}")
        return v
