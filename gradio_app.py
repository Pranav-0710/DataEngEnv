import gradio as gr
import httpx
import json

BASE_URL = "http://localhost:7860"

# ─── API Helpers ──────────────────────────────────────────────────────────────

def get_pipeline_status():
    try:
        r = httpx.get(f"{BASE_URL}/pipeline_status", timeout=10)
        return r.json()
    except:
        return {}

def reset_env():
    try:
        r = httpx.post(f"{BASE_URL}/reset", json={}, timeout=10)
        data = r.json()
        obs = data if "current_stage" in data else data.get("observation", data)
        script = obs.get("script_content", "")
        error = obs.get("last_run_error") or "None"
        status = get_pipeline_status()
        stage_html = build_stage_html(status)
        return "✅ Environment reset! Starting Stage 1.", script, error, json.dumps(data, indent=2), stage_html
    except Exception as e:
        return f"❌ Error: {e}", "", "", "", build_stage_html({})

def step_env(action_type: str, old_text: str, new_text: str):
    payload = {}
    if action_type == "edit_script":
        payload = {"old": old_text, "new": new_text}
    try:
        body = {"action_type": action_type, "payload": payload}
        r = httpx.post(f"{BASE_URL}/step", json=body, timeout=15)
        data = r.json()
        obs = data.get("observation", data)
        rwd = data.get("reward", {})
        script = obs.get("script_content", "")
        error = obs.get("last_run_error") or "None"
        score = rwd.get("score", "—")
        msg = rwd.get("message", "")
        status = get_pipeline_status()
        stage_html = build_stage_html(status)
        current_stage = obs.get("current_stage", "—")
        status_msg = f"✅ Stage {current_stage} | Action: {action_type} | Score: {score} | {msg}"
        return status_msg, script, error, json.dumps(data, indent=2), stage_html
    except Exception as e:
        return f"❌ Error: {e}", "", "", "", build_stage_html({})

def query_actor_ui():
    try:
        body = {"action_type": "query_actor", "payload": {}}
        r = httpx.post(f"{BASE_URL}/step", json=body, timeout=15)
        data = r.json()
        obs = data.get("observation", data)
        feedback = obs.get("actor_feedback", "No feedback available.")
        rwd = data.get("reward", {})
        status = get_pipeline_status()
        stage_html = build_stage_html(status)
        return f"🤖 Actor Feedback: {feedback}", "", "", json.dumps(data, indent=2), stage_html
    except Exception as e:
        return f"❌ Error: {e}", "", "", "", build_stage_html({})

def refresh_status():
    status = get_pipeline_status()
    return build_stage_html(status)

def build_stage_html(status: dict) -> str:
    completed = status.get("stages_completed", [])
    current = status.get("current_stage", 1)
    episode_score = status.get("episode_score", 0.0)
    total_steps = status.get("total_steps", 0)

    stages = [
        ("1", "Data Repair", "Fix column bugs & NaN values", "🔧"),
        ("2", "Training Monitor", "Fix divergence with StandardScaler", "📈"),
        ("3", "Eval Validation", "Fix data leakage bug", "🔍"),
        ("4", "Deploy Gate", "Fix model fairness constraint", "🚀"),
    ]

    cards = ""
    for sid, name, desc, icon in stages:
        sid_int = int(sid)
        if sid_int in completed:
            state_class = "stage-done"
            badge = '<span class="stage-badge badge-done">✅ Complete</span>'
        elif sid_int == current:
            state_class = "stage-active"
            badge = '<span class="stage-badge badge-active">⚡ Active</span>'
        else:
            state_class = "stage-pending"
            badge = '<span class="stage-badge badge-pending">⏳ Pending</span>'

        cards += f"""
        <div class="stage-card {state_class}">
            <div class="stage-icon">{icon}</div>
            <div class="stage-info">
                <div class="stage-name">Stage {sid}: {name}</div>
                <div class="stage-desc">{desc}</div>
            </div>
            {badge}
        </div>"""

    score_pct = int(episode_score * 100)
    bar_color = "#22c55e" if score_pct >= 75 else "#f59e0b" if score_pct >= 25 else "#6366f1"

    return f"""
    <div class="pipeline-panel">
      <div class="pipeline-header">
        <div class="pipeline-title">🏗️ PipelineOps Arena</div>
        <div class="pipeline-meta">
          <span class="meta-chip">Stage {current}/4</span>
          <span class="meta-chip">Steps: {total_steps}</span>
          <span class="meta-chip" style="color:#86efac">Score: {score_pct}%</span>
        </div>
      </div>
      <div class="progress-bar-wrap">
        <div class="progress-bar-inner" style="width:{score_pct}%; background:{bar_color};"></div>
      </div>
      <div class="stage-list">{cards}</div>
    </div>
    """

# ─── CSS ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=JetBrains+Mono:wght@400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }
body, .gradio-container { 
    font-family: 'Inter', sans-serif !important; 
    background: #060910 !important;
}

/* ── Tab styling ─────────────────────────────────────────── */
.tabs > .tab-nav { border-bottom: 1px solid rgba(99,102,241,0.2) !important; }
.tabs > .tab-nav button {
    color: #64748b !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    padding: 10px 20px !important;
    border-radius: 8px 8px 0 0 !important;
    transition: all 0.2s !important;
}
.tabs > .tab-nav button.selected { 
    color: #818cf8 !important; 
    border-bottom: 2px solid #6366f1 !important;
    background: rgba(99,102,241,0.08) !important;
}

/* ── Hero Banner ─────────────────────────────────────────── */
.hero-banner {
    background: linear-gradient(135deg, #0f172a 0%, #1a1040 40%, #0f1729 100%);
    border: 1px solid rgba(99,102,241,0.25);
    border-radius: 16px;
    padding: 36px 44px;
    margin-bottom: 20px;
    position: relative;
    overflow: hidden;
}
.hero-banner::before {
    content: '';
    position: absolute;
    top: -80px; right: -80px;
    width: 350px; height: 350px;
    background: radial-gradient(circle, rgba(99,102,241,0.12) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-banner::after {
    content: '';
    position: absolute;
    bottom: -60px; left: 30%;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(139,92,246,0.08) 0%, transparent 70%);
    border-radius: 50%;
    pointer-events: none;
}
.hero-title {
    font-size: 2rem;
    font-weight: 800;
    color: #f1f5f9;
    margin: 0 0 8px 0;
    letter-spacing: -0.5px;
}
.hero-title span { 
    background: linear-gradient(90deg, #818cf8, #c084fc);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
.hero-sub {
    color: #94a3b8;
    font-size: 0.95rem;
    line-height: 1.6;
    margin: 0 0 20px 0;
    max-width: 580px;
}
.hero-badges { display: flex; gap: 8px; flex-wrap: wrap; }
.hbadge {
    padding: 5px 13px;
    border-radius: 999px;
    font-size: 0.78rem;
    font-weight: 600;
    border: 1px solid;
}
.hb-blue  { color: #93c5fd; border-color: rgba(59,130,246,0.4);  background: rgba(59,130,246,0.1); }
.hb-green { color: #86efac; border-color: rgba(34,197,94,0.4);   background: rgba(34,197,94,0.1); }
.hb-amber { color: #fcd34d; border-color: rgba(245,158,11,0.4);  background: rgba(245,158,11,0.1); }
.hb-purple{ color: #d8b4fe; border-color: rgba(139,92,246,0.4);  background: rgba(139,92,246,0.1); }
.hb-rose  { color: #fda4af; border-color: rgba(244,63,94,0.4);   background: rgba(244,63,94,0.1); }

/* ── Pipeline Panel ──────────────────────────────────────── */
.pipeline-panel {
    background: #0d1117;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 20px;
    height: 100%;
}
.pipeline-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 14px;
}
.pipeline-title {
    font-size: 1rem;
    font-weight: 700;
    color: #e2e8f0;
}
.pipeline-meta { display: flex; gap: 6px; }
.meta-chip {
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.1);
    color: #94a3b8;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 999px;
}
.progress-bar-wrap {
    background: rgba(255,255,255,0.06);
    border-radius: 999px;
    height: 6px;
    margin-bottom: 16px;
    overflow: hidden;
}
.progress-bar-inner {
    height: 100%;
    border-radius: 999px;
    transition: width 0.5s ease;
}
.stage-list { display: flex; flex-direction: column; gap: 10px; }
.stage-card {
    display: flex;
    align-items: center;
    gap: 12px;
    padding: 12px 14px;
    border-radius: 10px;
    border: 1px solid;
    transition: all 0.2s;
}
.stage-done    { background: rgba(34,197,94,0.06);   border-color: rgba(34,197,94,0.2); }
.stage-active  { background: rgba(99,102,241,0.1);   border-color: rgba(99,102,241,0.4);
                  box-shadow: 0 0 12px rgba(99,102,241,0.15); }
.stage-pending { background: rgba(255,255,255,0.02); border-color: rgba(255,255,255,0.06); }
.stage-icon { font-size: 1.4rem; flex-shrink: 0; }
.stage-info { flex: 1; }
.stage-name { font-size: 0.85rem; font-weight: 600; color: #e2e8f0; }
.stage-desc { font-size: 0.75rem; color: #64748b; margin-top: 2px; }
.stage-badge {
    font-size: 0.7rem;
    font-weight: 700;
    padding: 3px 10px;
    border-radius: 999px;
    white-space: nowrap;
    border: 1px solid;
}
.badge-done    { color: #86efac; border-color: rgba(34,197,94,0.4);   background: rgba(34,197,94,0.1); }
.badge-active  { color: #a5b4fc; border-color: rgba(99,102,241,0.5);  background: rgba(99,102,241,0.15); }
.badge-pending { color: #475569; border-color: rgba(71,85,105,0.3);   background: transparent; }

/* ── Action buttons ──────────────────────────────────────── */
.btn-primary {
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px !important;
    transition: opacity 0.2s !important;
    box-shadow: 0 4px 15px rgba(99,102,241,0.3) !important;
}
.btn-primary:hover { opacity: 0.88 !important; }
.btn-reset {
    background: rgba(255,255,255,0.06) !important;
    color: #94a3b8 !important;
    border: 1px solid rgba(255,255,255,0.1) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    transition: all 0.2s !important;
}
.btn-reset:hover { background: rgba(255,255,255,0.1) !important; color: #e2e8f0 !important; }
.btn-actor {
    background: rgba(245,158,11,0.12) !important;
    color: #fcd34d !important;
    border: 1px solid rgba(245,158,11,0.3) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
}
.btn-actor:hover { background: rgba(245,158,11,0.2) !important; }

/* ── Code / terminal boxes ───────────────────────────────── */
.code-box textarea, .code-box pre {
    background: #0d1117 !important;
    color: #e2e8f0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    border: 1px solid #1e293b !important;
    border-radius: 10px !important;
}
.error-box textarea {
    background: #130a0a !important;
    color: #fca5a5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    border: 1px solid #3b1a1a !important;
    border-radius: 10px !important;
}
.status-box textarea {
    background: #0a130a !important;
    color: #86efac !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    border: 1px solid #1a3b1a !important;
    border-radius: 10px !important;
}

/* ── Inputs ──────────────────────────────────────────────── */
.gr-input, textarea, input, select, .gr-dropdown {
    background: #0d1117 !important;
    border: 1px solid #1e293b !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
label span { color: #64748b !important; font-size: 0.8rem !important; font-weight: 600 !important; }

/* ── Docs tab ────────────────────────────────────────────── */
.doc-box {
    background: #0d1117;
    border: 1px solid #1e293b;
    border-radius: 12px;
    padding: 20px 24px;
    color: #cbd5e1;
    line-height: 1.7;
    font-size: 0.88rem;
}
.doc-box code {
    background: #1e293b;
    color: #a5b4fc;
    padding: 2px 6px;
    border-radius: 4px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.8rem;
}
.doc-box pre {
    background: #060910 !important;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 14px 16px;
    overflow-x: auto;
}
"""

HERO_HTML = """
<div class="hero-banner">
  <div class="hero-title">🛠️ <span>PipelineOps Arena</span></div>
  <p class="hero-sub">
    A 4-stage cascading RL environment where AI agents debug broken ML pipelines —
    graded <strong style="color:#818cf8">deterministically</strong>, no LLM judge required.
  </p>
  <div class="hero-badges">
    <span class="hbadge hb-blue">⚡ 4-Stage Cascade</span>
    <span class="hbadge hb-green">✅ Deterministic Grading</span>
    <span class="hbadge hb-amber">📊 Reward ∈ [0, 1]</span>
    <span class="hbadge hb-purple">🤖 OpenEnv Compliant</span>
    <span class="hbadge hb-rose">🦾 GRPO Fine-tuned</span>
  </div>
</div>
"""

DOCS_HTML = """
<div class="doc-box">
<h3 style="color:#e2e8f0; margin-top:0">🔌 API Quick Start</h3>

<b style="color:#94a3b8">Base URL:</b> <code>https://CoBeDigger-DataEngEnv.hf.space</code>

<pre><code style="color:#86efac">import requests

BASE = "https://CoBeDigger-DataEngEnv.hf.space"

# 1. Reset (starts at Stage 1)
obs = requests.post(f"{BASE}/reset").json()

# 2. Take a step
result = requests.post(f"{BASE}/step", json={
    "action_type": "edit_script",
    "payload": {"old": "age_years", "new": "age"}
}).json()

# 3. Check pipeline progress
status = requests.get(f"{BASE}/pipeline_status").json()
</code></pre>

<h3 style="color:#e2e8f0">🎮 Available Actions</h3>
<table style="width:100%; border-collapse:collapse;">
  <tr style="border-bottom:1px solid #1e293b">
    <td style="padding:8px; color:#a5b4fc; font-family:monospace">inspect_data</td>
    <td style="padding:8px; color:#94a3b8">View dataset shape, columns, nulls, sample rows</td>
  </tr>
  <tr style="border-bottom:1px solid #1e293b">
    <td style="padding:8px; color:#a5b4fc; font-family:monospace">check_schema</td>
    <td style="padding:8px; color:#94a3b8">Compare expected vs actual column names</td>
  </tr>
  <tr style="border-bottom:1px solid #1e293b">
    <td style="padding:8px; color:#a5b4fc; font-family:monospace">run_script</td>
    <td style="padding:8px; color:#94a3b8">Execute the current pipeline script</td>
  </tr>
  <tr style="border-bottom:1px solid #1e293b">
    <td style="padding:8px; color:#a5b4fc; font-family:monospace">edit_script</td>
    <td style="padding:8px; color:#94a3b8">Patch the script: payload = {"old": "...", "new": "..."}</td>
  </tr>
  <tr style="border-bottom:1px solid #1e293b">
    <td style="padding:8px; color:#a5b4fc; font-family:monospace">query_actor</td>
    <td style="padding:8px; color:#94a3b8">Ask the MLOps Bot for feedback (Stage 4)</td>
  </tr>
  <tr>
    <td style="padding:8px; color:#a5b4fc; font-family:monospace">submit</td>
    <td style="padding:8px; color:#94a3b8">Submit the current fix for grading</td>
  </tr>
</table>

<h3 style="color:#e2e8f0">🏗️ The 4 Stages</h3>
<ol style="color:#94a3b8; padding-left:20px; line-height:2">
  <li><b style="color:#e2e8f0">Data Repair</b> — Fix <code>age_years → age</code> column bug + handle NaN values</li>
  <li><b style="color:#e2e8f0">Training Monitor</b> — Add <code>StandardScaler</code> to fix MLP divergence</li>
  <li><b style="color:#e2e8f0">Eval Validation</b> — Move <code>scaler.fit()</code> after train/test split to fix data leakage</li>
  <li><b style="color:#e2e8f0">Deploy Gate</b> — Add <code>class_weight='balanced'</code> to fix model fairness</li>
</ol>
</div>
"""

# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="PipelineOps Arena — OpenEnv") as demo:

    gr.HTML(HERO_HTML)

    with gr.Tabs():

        # ── PLAYGROUND TAB ────────────────────────────────────────────────────
        with gr.Tab("🎮 Playground"):
            with gr.Row():

                # Left: Pipeline Status Panel
                with gr.Column(scale=1, min_width=280):
                    stage_display = gr.HTML(
                        value=build_stage_html({}),
                        label=""
                    )
                    refresh_btn = gr.Button("🔄 Refresh Status", elem_classes="btn-reset", size="sm")

                # Right: Interaction Panel
                with gr.Column(scale=2):
                    with gr.Row():
                        action_dd = gr.Dropdown(
                            choices=["inspect_data", "check_schema", "run_script",
                                     "edit_script", "query_actor", "submit"],
                            value="inspect_data",
                            label="Action",
                            scale=2
                        )
                        reset_btn = gr.Button("⟳ Reset Episode", elem_classes="btn-reset", scale=1)

                    with gr.Row():
                        old_text = gr.Textbox(
                            label="Old Code  (edit_script only)",
                            lines=2,
                            placeholder="Exact text to replace in the script…",
                            elem_classes="code-box"
                        )
                        new_text = gr.Textbox(
                            label="New Code  (edit_script only)",
                            lines=2,
                            placeholder="Replacement text…",
                            elem_classes="code-box"
                        )

                    with gr.Row():
                        step_btn  = gr.Button("▶  Execute Action", elem_classes="btn-primary", scale=3)
                        actor_btn = gr.Button("🤖 Query MLOps Bot", elem_classes="btn-actor", scale=1)

                    status_out = gr.Textbox(
                        label="Status",
                        interactive=False,
                        elem_classes="status-box",
                        lines=1
                    )

                    with gr.Row():
                        script_out = gr.Code(
                            label="📄 Current Pipeline Script",
                            language="python",
                            lines=14,
                            elem_classes="code-box"
                        )
                        error_out = gr.Textbox(
                            label="🔴 Last Run Error",
                            lines=14,
                            interactive=False,
                            elem_classes="error-box"
                        )

                    json_out = gr.Code(
                        label="📦 Raw API Response",
                        language="json",
                        lines=8,
                        elem_classes="code-box"
                    )

            # Wire up buttons
            step_btn.click(
                step_env,
                inputs=[action_dd, old_text, new_text],
                outputs=[status_out, script_out, error_out, json_out, stage_display]
            )
            actor_btn.click(
                query_actor_ui,
                inputs=[],
                outputs=[status_out, script_out, error_out, json_out, stage_display]
            )
            reset_btn.click(
                reset_env,
                inputs=[],
                outputs=[status_out, script_out, error_out, json_out, stage_display]
            )
            refresh_btn.click(
                refresh_status,
                inputs=[],
                outputs=[stage_display]
            )

        # ── BASELINE TAB ──────────────────────────────────────────────────────
        with gr.Tab("🚀 Run Baseline"):
            gr.Markdown("### Run the live LLM agent — Llama 3.1 reasons through each stage in real time")

            run_btn = gr.Button("▶  Run LLM Agent (All 4 Stages)", elem_classes="btn-primary")

            live_output = gr.Textbox(
                label="⚡ Live Action Log",
                lines=28,
                interactive=False,
                placeholder="Click Run to start the LLM agent…\n\nThe agent reads the error logs and script, reasons about the bug,\nand generates fix actions entirely on its own.",
                elem_classes="code-box"
            )

            ACTION_ICONS = {
                "inspect_data":  "🔍",
                "check_schema":  "📋",
                "run_script":    "▶️ ",
                "edit_script":   "✏️ ",
                "query_actor":   "🤖",
                "submit":        "📤",
            }
            STAGE_LABELS = {
                1: "Data Repair",
                2: "Training Monitor",
                3: "Eval Validation",
                4: "Deploy Gate",
            }

            def run_baseline_streaming():
                import time as _time
                import os as _os
                import re as _re
                log = []

                def emit(line):
                    log.append(line)
                    return "\n".join(log)

                yield emit("╔══════════════════════════════════════════════════════╗")
                yield emit("║      PipelineOps Arena — Live LLM Agent Run          ║")
                yield emit("╚══════════════════════════════════════════════════════╝\n")

                groq_key = _os.environ.get("GROQ_API_KEY", "")
                if not groq_key:
                    yield emit("❌ GROQ_API_KEY not set. Add it to the HF Space secrets.")
                    return

                try:
                    from groq import Groq
                    client = Groq(api_key=groq_key)
                except ImportError:
                    yield emit("❌ groq package not installed. Add 'groq' to requirements.txt.")
                    return

                SYSTEM_PROMPT = """You are debugging a broken ML pipeline. Fix it fast.

RESPOND WITH ONLY A JSON OBJECT. No explanation. No markdown.

Actions:
1. {"action_type": "inspect_data", "payload": {}}
2. {"action_type": "run_script", "payload": {}}
3. {"action_type": "edit_script", "payload": {"old": "exact line from script", "new": "replacement"}}
4. {"action_type": "edit_script", "payload": {"script": "FULL corrected script here"}}
5. {"action_type": "query_actor", "payload": {}}
6. {"action_type": "submit", "payload": {}}

STRATEGY:
1. Inspect data OR run script ONCE to see the error
2. Use edit_script to fix the bug — prefer option 4 (full script) if unsure about exact text match
3. Run script ONCE to verify fix
4. Submit immediately if no error

RULES:
- Maximum 5 actions per stage
- NEVER inspect more than once, NEVER run more than twice
- Submit as soon as output looks correct
- If you see KeyError 'age_years': rename to 'age' and add df.dropna(inplace=True) before feature selection
- If loss is NaN: add StandardScaler before classifier
- If accuracy suspiciously high (>0.95): move scaler.fit() to AFTER train_test_split()
- If fairness issue: add class_weight='balanced' to classifier"""

                def parse_action(raw_text):
                    match = _re.search(r'\{[^{}]*("payload"\s*:\s*\{[^{}]*\})?[^{}]*\}', raw_text, _re.DOTALL)
                    if match:
                        try:
                            action = json.loads(match.group().replace("'", '"'))
                            if "action_type" in action:
                                action.setdefault("payload", {})
                                return action
                        except json.JSONDecodeError:
                            pass
                    for atype in ["edit_script", "run_script", "submit", "inspect_data", "query_actor"]:
                        if atype in raw_text:
                            return {"action_type": atype, "payload": {}}
                    return {"action_type": "run_script", "payload": {}}

                def format_obs(obs, reward=None):
                    parts = [f"Stage {obs.get('current_stage','?')} | Step {obs.get('stage_step_number','?')}"]
                    if obs.get("last_run_error"):
                        parts.append(f"\nERROR:\n{str(obs['last_run_error'])[:400]}")
                    if obs.get("last_run_output"):
                        parts.append(f"\nOUTPUT:\n{str(obs['last_run_output'])[:300]}")
                    if obs.get("data_preview"):
                        parts.append(f"\nDATA:\n{str(obs['data_preview'])[:400]}")
                    if obs.get("actor_feedback"):
                        parts.append(f"\nREVIEWER:\n{str(obs['actor_feedback'])[:300]}")
                    if obs.get("script_content"):
                        parts.append(f"\nSCRIPT:\n{obs['script_content'][:800]}")
                    if reward:
                        parts.append(f"\nREWARD: {reward.get('score',0):.2f} | {reward.get('message','')}")
                    return "\n".join(parts)

                try:
                    r = httpx.post(f"{BASE_URL}/reset", timeout=15)
                    obs = r.json()
                    if "observation" in obs:
                        obs = obs["observation"]
                    yield emit(f"✅ Environment reset → Stage 1: {STAGE_LABELS[1]}\n")
                    _time.sleep(0.3)

                    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
                    initial = format_obs(obs)
                    initial += "\n\nStart by inspecting the data, then fix the bug. What is your first action?"
                    messages.append({"role": "user", "content": initial})

                    current_stage = 1
                    step_num = 0
                    action_history = []

                    for _ in range(60):
                        try:
                            trimmed = [messages[0], messages[1]] + messages[-4:] if len(messages) > 8 else messages
                            response = client.chat.completions.create(
                                model="llama-3.1-8b-instant",
                                messages=trimmed,
                                max_tokens=800,
                                temperature=0.2,
                            )
                            raw = response.choices[0].message.content.strip()
                            action = parse_action(raw)
                        except Exception as e:
                            yield emit(f"  LLM error: {e}")
                            action = {"action_type": "run_script", "payload": {}}

                        # Anti-loop guards
                        action_history.append(action["action_type"])
                        if len(action_history) >= 2 and action_history[-2] == "submit" and action["action_type"] == "submit":
                            action = {"action_type": "edit_script", "payload": {"script": obs.get("script_content", "")}}
                        if len(action_history) >= 4:
                            last4 = action_history[-4:]
                            if last4 in [["edit_script", "run_script", "edit_script", "run_script"],
                                         ["run_script", "edit_script", "run_script", "edit_script"]]:
                                action = {"action_type": "submit", "payload": {}}
                        if len(action_history) >= 3 and len(set(action_history[-3:])) == 1:
                            if action_history[-1] == "inspect_data":
                                action = {"action_type": "edit_script", "payload": {"script": obs.get("script_content", "")}}
                            elif action_history[-1] == "submit":
                                action = {"action_type": "run_script", "payload": {}}
                            else:
                                action = {"action_type": "submit", "payload": {}}

                        step_num += 1
                        atype = action["action_type"]
                        icon = ACTION_ICONS.get(atype, "▸")

                        yield emit(f"  Step {step_num:02d} │ {icon} {atype.upper():<18} │ running…")
                        _time.sleep(0.2)

                        resp = httpx.post(f"{BASE_URL}/step", json=action, timeout=30)
                        data = resp.json()
                        obs = data.get("observation", {})
                        reward = data.get("reward", {})
                        score = float(reward.get("score", 0.0))
                        msg = reward.get("message", "")[:55]
                        new_stage = obs.get("current_stage", current_stage)
                        done = obs.get("done", False)

                        log[-1] = f"  Step {step_num:02d} │ {icon} {atype.upper():<18} │ score={score:.2f}  {msg}"
                        yield "\n".join(log)

                        if new_stage != current_stage:
                            current_stage = new_stage
                            action_history = []
                            label = STAGE_LABELS.get(current_stage, f"Stage {current_stage}")
                            yield emit(f"\n{'─'*54}")
                            yield emit(f"  ➜ Advancing to Stage {current_stage}: {label}")
                            yield emit(f"{'─'*54}\n")

                        messages.append({"role": "assistant", "content": json.dumps(action)})
                        next_prompt = format_obs(obs, reward)
                        if atype == "inspect_data":
                            next_prompt += "\n\nData inspected. Now use edit_script to fix the bug."
                        elif atype == "run_script" and obs.get("last_run_error"):
                            next_prompt += '\n\nScript has error. Use edit_script with full script rewrite: {"action_type":"edit_script","payload":{"script":"..."}}'
                        elif atype == "run_script" and not obs.get("last_run_error"):
                            next_prompt += '\n\nScript ran with no error! Submit now: {"action_type":"submit","payload":{}}'
                        elif atype == "edit_script":
                            next_prompt += '\n\nEdited. Run script to verify: {"action_type":"run_script","payload":{}}'
                        messages.append({"role": "user", "content": next_prompt})

                        if done:
                            break

                        _time.sleep(0.3)

                    status = httpx.get(f"{BASE_URL}/pipeline_status", timeout=10).json()
                    completed = status.get("stages_completed", [])
                    final_score = status.get("episode_score", 0.0)
                    total_steps = status.get("total_steps", step_num)

                    yield emit(f"\n╔══════════════════════════════════════════════════════╗")
                    yield emit(f"║  EPISODE COMPLETE                                    ║")
                    yield emit(f"║  Stages completed : {str(completed):<34}║")
                    yield emit(f"║  Total steps      : {str(total_steps):<34}║")
                    yield emit(f"║  Final score      : {final_score:.2f}{'':<33}║")
                    yield emit(f"╚══════════════════════════════════════════════════════╝")

                except Exception as e:
                    yield emit(f"\n❌ Error during agent run: {e}")

            run_btn.click(run_baseline_streaming, inputs=[], outputs=[live_output])

        # ── DOCS TAB ─────────────────────────────────────────────────────────
        with gr.Tab("📖 API Docs"):
            gr.HTML(DOCS_HTML)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        css=CUSTOM_CSS
    )
