import gradio as gr
import httpx
import json

BASE_URL = "http://localhost:7860"

TASK_DESCRIPTIONS = {
    1: "Easy — Column Rename Bug: Fix the pipeline where 'age_years' should be 'age'.",
    2: "Medium — Dirty Data: Handle NaN values and salary outliers before modeling.",
    3: "Hard — Data Leakage: Move scaler.fit() to after train_test_split() to fix leakage.",
}

# ─── Playground helpers ───────────────────────────────────────────────────────

def reset_env(task_id: int):
    try:
        r = httpx.post(f"{BASE_URL}/reset", json={"task_id": task_id}, timeout=10)
        data = r.json()
        script = data.get("script_content", "")
        error  = data.get("last_run_error") or "None"
        obs_pretty = json.dumps(data, indent=2)
        status = f"✅ Reset to Task {task_id}"
        return status, script, error, obs_pretty
    except Exception as e:
        return f"❌ Error: {e}", "", "", ""

def step_env(action_type: str, old_text: str, new_text: str):
    payload: dict = {}
    if action_type == "edit_script":
        payload = {"old": old_text, "new": new_text}
    try:
        body = {"action_type": action_type, "payload": payload}
        r = httpx.post(f"{BASE_URL}/step", json=body, timeout=15)
        data = r.json()
        obs  = data.get("observation", {})
        rwd  = data.get("reward", {})
        script = obs.get("script_content", "")
        error  = obs.get("last_run_error") or "None"
        score  = rwd.get("score", "—")
        msg    = rwd.get("message", "")
        obs_pretty = json.dumps(data, indent=2)
        status = f"✅ Step done | Score: {score} | {msg}"
        return status, script, error, obs_pretty
    except Exception as e:
        return f"❌ Error: {e}", "", "", ""

def get_state():
    try:
        r = httpx.get(f"{BASE_URL}/state", timeout=10)
        return json.dumps(r.json(), indent=2)
    except Exception as e:
        return f"Error: {e}"

def submit_grader_ui(task_id: int):
    try:
        r = httpx.post(f"{BASE_URL}/grader", json={"task_id": task_id}, timeout=15)
        data = r.json()
        score = data.get("score", "—")
        msg   = data.get("message", "")
        pretty = json.dumps(data, indent=2)
        return f"🏆 Final Score: {score} | {msg}", pretty
    except Exception as e:
        return f"❌ Error: {e}", ""

# ─── CSS ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
/* ── Global ─────────────────────────────────────────── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
* { font-family: 'Inter', sans-serif !important; }

/* ── Hero section (Custom tab) ───────────────────────── */
.hero-section {
    background: linear-gradient(135deg, #0f172a 0%, #1e1b4b 50%, #0f172a 100%);
    border-radius: 16px;
    padding: 40px 48px;
    margin-bottom: 24px;
    border: 1px solid rgba(99,102,241,0.3);
    position: relative;
    overflow: hidden;
}
.hero-section::before {
    content: '';
    position: absolute;
    top: -60px; right: -60px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.hero-title {
    color: #f8fafc;
    font-size: 2.2rem;
    font-weight: 700;
    margin: 0 0 8px 0;
    display: flex;
    align-items: center;
    gap: 12px;
}
.hero-subtitle {
    color: #94a3b8;
    font-size: 1rem;
    margin: 0 0 24px 0;
    line-height: 1.6;
    max-width: 600px;
}
.hero-badges {
    display: flex;
    gap: 10px;
    flex-wrap: wrap;
}
.badge {
    background: rgba(255,255,255,0.08);
    border: 1px solid rgba(255,255,255,0.12);
    color: #e2e8f0;
    padding: 6px 14px;
    border-radius: 20px;
    font-size: 0.82rem;
    font-weight: 500;
}
.badge-blue  { border-color: rgba(59,130,246,0.5);  color: #93c5fd; background: rgba(59,130,246,0.1); }
.badge-green { border-color: rgba(34,197,94,0.5);   color: #86efac; background: rgba(34,197,94,0.1); }
.badge-amber { border-color: rgba(245,158,11,0.5);  color: #fcd34d; background: rgba(245,158,11,0.1); }
.badge-purple{ border-color: rgba(139,92,246,0.5);  color: #c4b5fd; background: rgba(139,92,246,0.1); }

/* ── Status bar ──────────────────────────────────────── */
.status-bar { font-weight: 600 !important; }

/* ── Terminal output ─────────────────────────────────── */
.terminal-box textarea {
    background: #0d1117 !important;
    color: #4ade80 !important;
    font-family: 'Courier New', monospace !important;
    font-size: 0.82rem !important;
    border: 1px solid #21262d !important;
    border-radius: 8px !important;
}

/* ── Run button ───────────────────────────────────────── */
.run-btn {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    font-size: 1rem !important;
    font-weight: 600 !important;
    border-radius: 10px !important;
    border: none !important;
    padding: 14px !important;
}
.run-btn:hover { opacity: 0.9 !important; }

/* ── Step button ──────────────────────────────────────── */
.step-btn {
    background: #15803d !important;
    color: white !important;
    font-weight: 600 !important;
    border-radius: 8px !important;
}

/* ── Section label ────────────────────────────────────── */
.section-label {
    color: #64748b;
    font-size: 0.72rem;
    font-weight: 600;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 8px;
}
"""

HERO_HTML = """
<div class="hero-section">
  <div class="hero-title">🛠️ DataEngEnv</div>
  <p class="hero-subtitle">
    An OpenEnv agentic environment where an AI agent debugs broken ML data pipelines —
    graded <strong style="color:#818cf8">deterministically</strong>, no LLM judge required.
  </p>
  <div class="hero-badges">
    <span class="badge badge-blue">⚡ 3 Tasks</span>
    <span class="badge badge-green">✅ Deterministic Grader</span>
    <span class="badge badge-amber">📊 Reward ∈ (0, 1)</span>
    <span class="badge badge-purple">🤖 OpenEnv Compliant</span>
  </div>
</div>
"""

# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(css=CUSTOM_CSS, title="DataEngEnv — OpenEnv") as demo:

    gr.Markdown("# OpenEnv Agentic Environment: DataEngEnv")

    with gr.Tabs():

        # ── PLAYGROUND TAB ────────────────────────────────────────────────────
        with gr.Tab("🎮 Playground"):
            with gr.Row():
                # Left sidebar
                with gr.Column(scale=1):
                    with gr.Accordion("📖 Quick Start", open=True):
                        gr.Markdown("""
**Connect from Python:**
```python
import httpx
base = "https://CoBeDigger-DataEngEnv.hf.space"

# Reset to task 1
obs = httpx.post(f"{base}/reset", json={"task_id": 1}).json()

# Take a step
result = httpx.post(f"{base}/step", json={
    "action_type": "inspect_data",
    "payload": {}
}).json()
```

**Available actions:**
- `inspect_data` — view dataset
- `check_schema` — compare columns
- `run_script` — execute pipeline
- `edit_script` — patch the code
- `submit` — grade your fix
                        """)

                # Main playground area
                with gr.Column(scale=2):
                    gr.Markdown("## Playground")
                    gr.Markdown("Select a task, hit **Reset**, then interact step by step.")

                    with gr.Row():
                        task_dd = gr.Dropdown(
                            choices=[1, 2, 3], value=1, label="Task ID",
                            info="1=Easy · 2=Medium · 3=Hard"
                        )
                        action_dd = gr.Dropdown(
                            choices=["inspect_data", "check_schema", "run_script", "edit_script", "submit"],
                            value="inspect_data", label="Action Type"
                        )

                    with gr.Row():
                        old_text = gr.Textbox(label="Old Code (edit_script only)", lines=2, placeholder="Text to replace…")
                        new_text = gr.Textbox(label="New Code (edit_script only)", lines=2, placeholder="Replacement text…")

                    with gr.Row():
                        step_btn   = gr.Button("▶ Step",   elem_classes="step-btn")
                        reset_btn  = gr.Button("🔄 Reset", variant="secondary")
                        state_btn  = gr.Button("📋 State", variant="secondary")
                        submit_btn = gr.Button("🏆 Submit / Grade", variant="secondary")

                    status_out = gr.Textbox(label="Status", elem_classes="status-bar", interactive=False)

                    with gr.Row():
                        script_out = gr.Code(label="Current Script", language="python", lines=12)
                        error_out  = gr.Textbox(label="Last Run Error", lines=12, interactive=False)

                    json_out = gr.Code(label="Raw JSON Response", language="json", lines=10)

            # ── Wire up buttons ──
            reset_btn.click(
                reset_env,
                inputs=[task_dd],
                outputs=[status_out, script_out, error_out, json_out]
            )
            step_btn.click(
                step_env,
                inputs=[action_dd, old_text, new_text],
                outputs=[status_out, script_out, error_out, json_out]
            )
            state_btn.click(
                lambda: get_state(),
                inputs=[],
                outputs=[json_out]
            )
            submit_btn.click(
                submit_grader_ui,
                inputs=[task_dd],
                outputs=[status_out, json_out]
            )

        # ── CUSTOM / BASELINE TAB ─────────────────────────────────────────────
        with gr.Tab("🚀 Baseline"):
            gr.HTML(HERO_HTML)

            gr.Markdown('<p class="section-label">Configuration</p>')

            with gr.Row():
                with gr.Column():
                    model_input = gr.Textbox(
                        label="Model",
                        value="llama-3.1-8b-instant",
                        info="Model name passed to the OpenAI-compatible proxy (MODEL_NAME env var).",
                        placeholder="e.g. llama-3.1-8b-instant"
                    )
                with gr.Column():
                    tasks_check = gr.CheckboxGroup(
                        choices=[
                            "Task 1 — Easy (Column Rename Bug)",
                            "Task 2 — Medium (Dirty Data)",
                            "Task 3 — Hard (Data Leakage)"
                        ],
                        value=[
                            "Task 1 — Easy (Column Rename Bug)",
                            "Task 2 — Medium (Dirty Data)",
                            "Task 3 — Hard (Data Leakage)"
                        ],
                        label="Tasks to Evaluate"
                    )

            run_btn = gr.Button("▶  Run Baseline Evaluation", elem_classes="run-btn")

            gr.Markdown('<p class="section-label">Live Output</p>')
            live_output = gr.Textbox(
                label="",
                lines=18,
                interactive=False,
                placeholder="Baseline output will appear here…",
                elem_classes="terminal-box"
            )

            def run_baseline_ui(model, tasks):
                import subprocess, os, sys
                env = os.environ.copy()
                env["MODEL_NAME"] = model
                env["DATAENGENV_URL"] = BASE_URL
                try:
                    result = subprocess.run(
                        [sys.executable, "baseline/run_baseline.py"],
                        capture_output=True, text=True, timeout=300, env=env
                    )
                    out = result.stdout + result.stderr
                    return out if out.strip() else "✅ Baseline completed with no output."
                except subprocess.TimeoutExpired:
                    return "⏱ Baseline timed out after 5 minutes."
                except Exception as e:
                    return f"❌ Error running baseline: {e}"

            run_btn.click(run_baseline_ui, inputs=[model_input, tasks_check], outputs=[live_output])

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)
