import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.environment import DataEngEnvironment
from app.models import Action, Observation, Reward, StepResponse, StateResponse, TaskInfo
from app.tasks import (
    stage1_data_repair,
    stage2_training_monitor,
    stage3_eval_validation,
    stage4_deploy_gate,
)

app = FastAPI(title="DataEngEnv API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Singleton Environment Instance
env = DataEngEnvironment()

class TaskIdRequest(BaseModel):
    task_id: int | None = 1

class ErrorResponse(BaseModel):
    error: str

class HealthResponse(BaseModel):
    status: str
    env: str
    version: str

class BaselineResponse(BaseModel):
    success: bool
    output: str

class PipelineStatusResponse(BaseModel):
    current_stage: int
    loop_count: int
    stages_completed: list[int]
    total_steps: int
    episode_score: float
    actor_feedback: str

@app.on_event("startup")
async def startup_event():
    print("DataEngEnv ready")
    # Call generate_scenario() on all 4 stages to validate they load properly at startup
    try:
        stage1_data_repair.generate_scenario()
        stage2_training_monitor.generate_scenario()
        stage3_eval_validation.generate_scenario()
        stage4_deploy_gate.generate_scenario()
    except Exception as e:
        print(f"Warning: A stage failed to load during startup validation: {e}")


@app.post("/reset", response_model=Observation | ErrorResponse)
async def api_reset(req: TaskIdRequest | None = None):
    try:
        tid = req.task_id if req is not None and req.task_id is not None else 1
        return env.reset(tid)
    except Exception as e:
        return ErrorResponse(error=str(e))

@app.post("/step", response_model=StepResponse | ErrorResponse)
async def api_step(action: Action):
    try:
        return env.step(action)
    except Exception as e:
        return ErrorResponse(error=str(e))

@app.get("/state", response_model=StateResponse | ErrorResponse)
async def api_state():
    try:
        return env.state()
    except Exception as e:
        return ErrorResponse(error=str(e))

@app.get("/tasks", response_model=list[TaskInfo] | ErrorResponse)
async def api_tasks():
    try:
        scenarios = {
            1: stage1_data_repair.generate_scenario(),
            2: stage2_training_monitor.generate_scenario(),
            3: stage3_eval_validation.generate_scenario(),
            4: stage4_deploy_gate.generate_scenario(),
        }
        return [
            TaskInfo(
                task_id=1,
                name="Stage 1: Data Repair",
                difficulty="easy",
                description=scenarios[1]["task_description"],
                action_schema={}
            ),
            TaskInfo(
                task_id=2,
                name="Stage 2: Training Monitor",
                difficulty="medium",
                description=scenarios[2]["task_description"],
                action_schema={}
            ),
            TaskInfo(
                task_id=3,
                name="Stage 3: Eval Validation",
                difficulty="hard",
                description=scenarios[3]["task_description"],
                action_schema={}
            ),
            TaskInfo(
                task_id=4,
                name="Stage 4: Deployment Gate",
                difficulty="hard",
                description=scenarios[4]["task_description"],
                action_schema={}
            )
        ]
    except Exception as e:
        return ErrorResponse(error=str(e))

@app.post("/grader", response_model=Reward | ErrorResponse)
async def api_grader(req: TaskIdRequest | None = None):
    try:
        tid = req.task_id if req is not None and req.task_id is not None else env.current_task_id
        if tid != env.current_task_id:
            return ErrorResponse(error="task_id mismatch - call /reset first; the pipeline controls stage advancement")
        return env.submit_grader(tid)
    except Exception as e:
        return ErrorResponse(error=str(e))

@app.get("/pipeline_status", response_model=PipelineStatusResponse | ErrorResponse)
async def api_pipeline_status():
    try:
        return PipelineStatusResponse(
            current_stage=env.current_stage,
            loop_count=env.loop_count,
            stages_completed=list(env.stages_completed),
            total_steps=env.step_number,
            episode_score=float(env.episode_score),
            actor_feedback=env.actor_feedback,
        )
    except Exception as e:
        return ErrorResponse(error=str(e))

@app.get("/baseline", response_model=BaselineResponse | ErrorResponse)
async def api_baseline():
    try:
        # Max timeout set to 300s safely bounds execution
        result = subprocess.run(["python", "baseline/run_baseline.py"], capture_output=True, text=True, timeout=300)
        return BaselineResponse(success=(result.returncode == 0), output=result.stdout + result.stderr)
    except Exception as e:
        return ErrorResponse(error=str(e))

@app.get("/health", response_model=HealthResponse | ErrorResponse)
async def api_health():
    try:
        return HealthResponse(status="ok", env="DataEngEnv", version="1.0.0")
    except Exception as e:
        return ErrorResponse(error=str(e))

import gradio as gr
import urllib.request
import time

try:
    from gradio_app import demo
    app = gr.mount_gradio_app(app, demo, path="/")
except ImportError as e:
    print(f"Warning: Could not import gradio dashboard: {e}")

def main():
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=7860)
