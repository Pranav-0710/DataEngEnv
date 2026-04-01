import subprocess
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.environment import DataEngEnvironment
from app.models import Action, Observation, Reward, StepResponse, StateResponse, TaskInfo
from app.tasks import task1_easy, task2_medium, task3_hard

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

@app.on_event("startup")
async def startup_event():
    print("DataEngEnv ready")
    # Call generate_scenario() on all 3 tasks to validate they load properly at startup
    try:
        task1_easy.generate_scenario()
        task2_medium.generate_scenario()
        task3_hard.generate_scenario()
    except Exception as e:
        print(f"Warning: A task failed to load during startup validation: {e}")

@app.get("/")
def root():
    return {"status": "ok", "env": "DataEngEnv", "version": "1.0.0", 
            "docs": "/docs", "health": "/health", "tasks": "/tasks"}
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
        return [
            TaskInfo(
                task_id=1,
                name="Easy Data Cleaning",
                difficulty="easy",
                description="Fix the broken script by identifying and replacing the incorrect column name ('age_years' to 'age').",
                action_schema={}
            ),
            TaskInfo(
                task_id=2,
                name="Medium Data Imputation",
                difficulty="medium",
                description="Detect missing values and outliers in the numeric dataset. Apply dropna/fillna for NaNs and clip/quantile for outliers.",
                action_schema={}
            ),
            TaskInfo(
                task_id=3,
                name="Hard Data Leakage",
                difficulty="hard",
                description="Fix the critical logical data leakage issue by moving the scaler.fit() call AFTER train_test_split() to avoid cheating on the validation set.",
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
            return ErrorResponse(error="task_id mismatch — call /reset with this task_id first")
        return env.submit_grader(tid)
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
