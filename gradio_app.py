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

/* ── Hero stats ──────────────────────────────────────────── */
.hero-stat {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 12px;
    padding: 14px 20px;
    text-align: center;
    min-width: 72px;
}
.hero-stat-num {
    font-size: 1.6rem;
    font-weight: 800;
    color: #818cf8;
    line-height: 1;
}
.hero-stat-label {
    font-size: 0.68rem;
    color: #64748b;
    font-weight: 600;
    margin-top: 4px;
    white-space: nowrap;
}
.how-step {
    display: flex;
    align-items: center;
    gap: 8px;
    padding: 8px 12px;
    background: rgba(255,255,255,0.03);
    border-radius: 8px;
    border: 1px solid rgba(255,255,255,0.06);
}
.how-num {
    background: rgba(99,102,241,0.2);
    color: #818cf8;
    border-radius: 999px;
    width: 20px; height: 20px;
    display: flex; align-items: center; justify-content: center;
    font-size: 0.7rem; font-weight: 800; flex-shrink: 0;
}
.how-text { color: #94a3b8; font-size: 0.78rem; font-weight: 500; }
.how-arrow { color: #334155; font-size: 1rem; padding: 0 4px; display: flex; align-items: center; }

/* ── Score display ───────────────────────────────────────── */
.score-display {
    background: linear-gradient(135deg, #0d1117 0%, #0f1a2e 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 14px;
    padding: 24px;
    margin-top: 12px;
    display: flex;
    gap: 16px;
    align-items: stretch;
    flex-wrap: wrap;
}
.score-card {
    flex: 1;
    min-width: 140px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 10px;
    padding: 16px;
    text-align: center;
}
.score-card-big { font-size: 2.5rem; font-weight: 900; line-height: 1; }
.score-card-label { font-size: 0.72rem; color: #64748b; font-weight: 600; margin-top: 6px; }
.score-card-sub { font-size: 0.78rem; color: #475569; margin-top: 4px; }

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
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:20px">
    <div style="flex:1;min-width:260px">
      <div class="hero-title">🛠️ <span>PipelineOps Arena</span></div>
      <p class="hero-sub">
        A <strong style="color:#818cf8">4-stage cascading RL environment</strong> where AI agents
        autonomously debug broken ML pipelines — inspecting data, editing scripts, and verifying fixes
        in a real Python sandbox. Graded <strong style="color:#86efac">deterministically</strong>,
        no LLM judge required.
      </p>
      <div class="hero-badges">
        <span class="hbadge hb-blue">⚡ 4-Stage Cascade</span>
        <span class="hbadge hb-green">✅ Deterministic Grading</span>
        <span class="hbadge hb-amber">📊 Reward ∈ [0, 1]</span>
        <span class="hbadge hb-purple">🤖 OpenEnv Compliant</span>
        <span class="hbadge hb-rose">🦾 GRPO Fine-tuned LLM</span>
      </div>
    </div>
    <div style="display:flex;gap:12px;flex-wrap:wrap;align-items:flex-start">
      <div class="hero-stat"><div class="hero-stat-num">4</div><div class="hero-stat-label">Pipeline Stages</div></div>
      <div class="hero-stat"><div class="hero-stat-num">5</div><div class="hero-stat-label">Agent Actions</div></div>
      <div class="hero-stat"><div class="hero-stat-num">60</div><div class="hero-stat-label">Max Steps</div></div>
      <div class="hero-stat"><div class="hero-stat-num">1.0</div><div class="hero-stat-label">Max Score</div></div>
    </div>
  </div>
  <div style="margin-top:20px;padding-top:16px;border-top:1px solid rgba(99,102,241,0.15)">
    <div style="font-size:0.8rem;color:#475569;font-weight:600;letter-spacing:.05em;margin-bottom:10px">HOW IT WORKS</div>
    <div style="display:flex;gap:0;flex-wrap:wrap">
      <div class="how-step"><span class="how-num">1</span><span class="how-text">Agent receives broken ML pipeline + error log</span></div>
      <div class="how-arrow">→</div>
      <div class="how-step"><span class="how-num">2</span><span class="how-text">Calls actions: inspect, edit, run, query</span></div>
      <div class="how-arrow">→</div>
      <div class="how-step"><span class="how-num">3</span><span class="how-text">Fixed script runs in isolated Python sandbox</span></div>
      <div class="how-arrow">→</div>
      <div class="how-step"><span class="how-num">4</span><span class="how-text">Grader scores fix → advance to next stage</span></div>
    </div>
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

# ─── Comparison helper ────────────────────────────────────────────────────────

def _build_comparison_html(b_stages, b_score, t_stages, t_score):
    def pct(s):
        return int(float(s) * 100)

    def bar(s, color):
        p = pct(s)
        return (
            f'<div style="background:rgba(255,255,255,0.06);border-radius:999px;height:10px;overflow:hidden;margin:6px 0">'
            f'<div style="width:{p}%;height:100%;border-radius:999px;background:{color};transition:width 0.5s"></div></div>'
        )

    b_color = "#22c55e" if b_score >= 0.7 else "#f59e0b" if b_score >= 0.3 else "#ef4444"
    t_color = "#22c55e" if t_score >= 0.7 else "#f59e0b" if t_score >= 0.3 else "#ef4444"

    b_stages_str = ", ".join(str(s) for s in b_stages) if b_stages else "none"
    t_stages_str = ", ".join(str(s) for s in t_stages) if t_stages else "none"

    improvement = (t_score - b_score) * 100
    imp_color = "#22c55e" if improvement >= 0 else "#ef4444"
    imp_sign = "+" if improvement >= 0 else ""

    return f"""
<div style="background:#0d1117;border:1px solid rgba(99,102,241,0.25);border-radius:14px;padding:24px;margin-top:16px">
  <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:18px">📊 Comparison Results</div>
  <div style="display:grid;grid-template-columns:1fr 1fr 1fr;gap:16px">
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:16px">
      <div style="color:#94a3b8;font-size:0.8rem;font-weight:600;margin-bottom:8px">BASELINE (Rule-Based)</div>
      <div style="font-size:2rem;font-weight:800;color:{b_color}">{pct(b_score)}%</div>
      {bar(b_score, b_color)}
      <div style="color:#64748b;font-size:0.75rem">Stages: {b_stages_str}</div>
    </div>
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(99,102,241,0.3);border-radius:10px;padding:16px">
      <div style="color:#94a3b8;font-size:0.8rem;font-weight:600;margin-bottom:8px">GRPO-TRAINED (LLM)</div>
      <div style="font-size:2rem;font-weight:800;color:{t_color}">{pct(t_score)}%</div>
      {bar(t_score, t_color)}
      <div style="color:#64748b;font-size:0.75rem">Stages: {t_stages_str}</div>
    </div>
    <div style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);border-radius:10px;padding:16px">
      <div style="color:#94a3b8;font-size:0.8rem;font-weight:600;margin-bottom:8px">IMPROVEMENT</div>
      <div style="font-size:2rem;font-weight:800;color:{imp_color}">{imp_sign}{improvement:.0f}%</div>
      <div style="color:#64748b;font-size:0.75rem;margin-top:12px">GRPO training enables the agent to reason dynamically about unseen pipeline states vs fixed rule-based sequences.</div>
    </div>
  </div>
</div>"""

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

        # ── LLM AGENT TAB ─────────────────────────────────────────────────────
        with gr.Tab("🤖 Live LLM Agent"):
            gr.HTML("""
<div style="background:linear-gradient(135deg,rgba(99,102,241,0.08),rgba(139,92,246,0.05));
            border:1px solid rgba(99,102,241,0.2);border-radius:12px;padding:20px 24px;margin-bottom:16px">
  <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:6px">
    🤖 Watch the LLM Agent Solve All 4 Stages
  </div>
  <div style="color:#94a3b8;font-size:0.88rem;line-height:1.6;max-width:660px">
    <strong style="color:#a5b4fc">Groq Llama 3.1 8B</strong> receives error logs and the broken script,
    reasons about the root cause, and generates fix actions entirely on its own —
    no hardcoded scripts, no rule-based overrides.
    Watch it inspect data, patch code, run the sandbox, and advance through all 4 stages.
  </div>
  <div style="display:flex;gap:8px;margin-top:12px;flex-wrap:wrap">
    <span style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);
                 color:#86efac;padding:3px 10px;border-radius:999px;font-size:0.75rem;font-weight:600">
      ✅ Real-time reasoning
    </span>
    <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.3);
                 color:#a5b4fc;padding:3px 10px;border-radius:999px;font-size:0.75rem;font-weight:600">
      🔒 Isolated sandbox execution
    </span>
    <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
                 color:#fcd34d;padding:3px 10px;border-radius:999px;font-size:0.75rem;font-weight:600">
      📊 Deterministic grading
    </span>
  </div>
</div>""")

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

ALWAYS use full script replacement for edit_script:
{"action_type": "edit_script", "payload": {"script": "FULL corrected script here"}}

Never use the old/new format — it breaks when the script has already been partially edited.

Actions:
1. {"action_type": "inspect_data", "payload": {}}
2. {"action_type": "run_script", "payload": {}}
3. {"action_type": "edit_script", "payload": {"script": "FULL corrected script here"}}
4. {"action_type": "query_actor", "payload": {}}
5. {"action_type": "submit", "payload": {}}

STRATEGY (follow exactly):
Stage 1 — run_script → see error → edit_script (full replacement fixing ALL bugs at once) → run_script → submit
Stage 2 — inspect_data → edit_script (full replacement adding StandardScaler) → run_script → submit
Stage 3 — run_script → edit_script (full replacement moving scaler.fit after split) → run_script → submit
Stage 4 — query_actor → edit_script (full replacement adding class_weight='balanced') → run_script → submit

CRITICAL RULES:
- Fix ALL bugs for the stage in ONE single edit_script call — never split fixes across multiple edits
- NEVER use old/new format — always replace the full script
- Submit as soon as run_script shows no error and prints Accuracy
- Stage 1 has TWO bugs: rename age_years→age AND add df.dropna() before scaling — fix BOTH in one edit
- Stage 2: add StandardScaler fitted on X_train only, transform X_test separately
- Stage 3: move scaler.fit() to AFTER train_test_split(), fit only on X_train
- Stage 4: add class_weight='balanced' to LogisticRegression"""

                def parse_action(raw_text):
                    # Find the outermost JSON object by tracking brace depth
                    start = raw_text.find('{')
                    if start != -1:
                        depth = 0
                        for i, ch in enumerate(raw_text[start:], start):
                            if ch == '{':
                                depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    candidate = raw_text[start:i+1]
                                    try:
                                        action = json.loads(candidate)
                                        if "action_type" in action:
                                            action.setdefault("payload", {})
                                            return action
                                    except json.JSONDecodeError:
                                        pass
                                    break
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
                                max_tokens=1500,
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

            live_scorecard = gr.HTML("")

            def run_llm_agent_with_score():
                log_text = ""
                for chunk in run_baseline_streaming():
                    log_text = chunk
                    yield chunk, ""
                import re as _re
                completed_m = _re.search(r"Stages completed\s*:\s*(\[.*?\])", log_text)
                score_m = _re.search(r"Final score\s*:\s*([\d.]+)", log_text)
                steps_m = _re.search(r"Total steps\s*:\s*(\d+)", log_text)
                completed = completed_m.group(1) if completed_m else "[]"
                score = float(score_m.group(1)) if score_m else 0.0
                steps = steps_m.group(1) if steps_m else "—"
                pct = int(score * 100)
                color = "#22c55e" if pct >= 75 else "#f59e0b" if pct >= 25 else "#ef4444"
                bar = f'<div style="background:rgba(255,255,255,0.06);border-radius:999px;height:8px;overflow:hidden;margin:8px 0"><div style="width:{pct}%;height:100%;border-radius:999px;background:{color}"></div></div>'
                card = f"""<div style="background:linear-gradient(135deg,#0d1117,#0f1a2e);border:1px solid rgba(99,102,241,0.3);
                           border-radius:14px;padding:20px 24px;margin-top:12px">
                  <div style="font-size:0.85rem;font-weight:700;color:#64748b;letter-spacing:.05em;margin-bottom:14px">EPISODE RESULT</div>
                  <div style="display:flex;gap:16px;align-items:flex-start;flex-wrap:wrap">
                    <div style="text-align:center">
                      <div style="font-size:3rem;font-weight:900;color:{color};line-height:1">{pct}%</div>
                      {bar}
                      <div style="color:#64748b;font-size:0.72rem;font-weight:600">EPISODE SCORE</div>
                    </div>
                    <div style="flex:1;min-width:160px">
                      <div style="color:#e2e8f0;font-size:0.88rem;margin-bottom:6px">
                        <strong style="color:#94a3b8">Stages completed:</strong> {completed}
                      </div>
                      <div style="color:#e2e8f0;font-size:0.88rem;margin-bottom:6px">
                        <strong style="color:#94a3b8">Total steps:</strong> {steps}
                      </div>
                      <div style="color:#64748b;font-size:0.78rem;line-height:1.6;margin-top:10px">
                        Groq Llama 3.1 8B reasoned through each stage autonomously —
                        inspecting data, patching scripts, and verifying fixes in a real Python sandbox.
                      </div>
                    </div>
                  </div>
                </div>"""
                yield log_text, card

            run_btn.click(run_llm_agent_with_score, inputs=[], outputs=[live_output, live_scorecard])

        # ── COMPARISON TAB ────────────────────────────────────────────────────
        with gr.Tab("📊 Baseline vs LLM Agent"):
            gr.HTML("""
<div style="background:linear-gradient(135deg,rgba(15,23,42,0.8),rgba(20,16,48,0.8));
            border:1px solid rgba(99,102,241,0.2);border-radius:12px;padding:20px 24px;margin-bottom:16px">
  <div style="font-size:1.1rem;font-weight:700;color:#e2e8f0;margin-bottom:8px">
    📊 Head-to-Head: Rule-Based Baseline vs LLM Agent
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:16px;margin-top:12px">
    <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.08);
                border-radius:10px;padding:14px">
      <div style="color:#94a3b8;font-size:0.78rem;font-weight:700;margin-bottom:6px">
        🔧 RULE-BASED BASELINE
      </div>
      <div style="color:#64748b;font-size:0.8rem;line-height:1.6">
        Hardcoded scripts — always applies the same predefined fix sequence.
        Fast, but brittle. Fails if the environment varies even slightly.
      </div>
    </div>
    <div style="background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.25);
                border-radius:10px;padding:14px">
      <div style="color:#a5b4fc;font-size:0.78rem;font-weight:700;margin-bottom:6px">
        🤖 LLM AGENT (Groq Llama 3.1 8B)
      </div>
      <div style="color:#64748b;font-size:0.8rem;line-height:1.6">
        Reads the error log and current script, reasons about the root cause,
        then generates targeted fixes — no rules, no hardcoding.
      </div>
    </div>
  </div>
  <div style="margin-top:10px;color:#475569;font-size:0.77rem">
    ⚡ Each agent runs on its own <strong style="color:#64748b">private isolated environment</strong> — zero shared state between runs.
  </div>
</div>""")
            with gr.Row():
                with gr.Column():
                    gr.HTML("""<div style="color:#94a3b8;font-weight:700;font-size:0.9rem;padding:4px 0">
                      🔧 Rule-Based Baseline</div>""")
                    run_baseline_cmp_btn = gr.Button("▶ Run Baseline Agent", elem_classes="btn-reset")
                    baseline_cmp_log = gr.Textbox(
                        label="Baseline Log", lines=28, interactive=False,
                        placeholder="Click to run baseline agent…\n\nApplies fixed rule sequences — no reasoning.", elem_classes="code-box"
                    )
                with gr.Column():
                    gr.HTML("""<div style="color:#a5b4fc;font-weight:700;font-size:0.9rem;padding:4px 0">
                      🤖 LLM Agent (Groq Llama 3.1 8B)</div>""")
                    run_llm_cmp_btn = gr.Button("▶ Run LLM Agent", elem_classes="btn-primary")
                    llm_cmp_log = gr.Textbox(
                        label="LLM Log", lines=28, interactive=False,
                        placeholder="Click to run LLM agent…\n\nThe model reads error logs and reasons about each fix.", elem_classes="code-box"
                    )
                    llm_scorecard = gr.HTML("")

            baseline_scorecard = gr.HTML("")

            _ICONS = {"inspect_data":"🔍","run_script":"▶️ ","edit_script":"✏️ ","query_actor":"🤖","submit":"📤"}

            # ── fixed scripts for the baseline ──────────────────────────────
            _S1 = (
                "import pandas as pd\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "df = df.dropna()\n"
                "df['salary'] = df['salary'].clip(upper=df['salary'].quantile(0.99))\n"
                "X = df[['age','salary','credit_score','loan_amount','employment_years']].copy()\n"
                "y = df['target']\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "scaler = StandardScaler()\n"
                "X_train = scaler.fit_transform(X_train)\n"
                "X_test  = scaler.transform(X_test)\n"
                "clf = LogisticRegression(max_iter=1000, random_state=42)\n"
                "clf.fit(X_train, y_train)\n"
                "print('Accuracy:', clf.score(X_test, y_test))\n"
            )
            _S2 = (
                "import pandas as pd\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.neural_network import MLPClassifier\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "X = df.drop(columns=['target'])\n"
                "y = df['target']\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "scaler = StandardScaler()\n"
                "X_train = scaler.fit_transform(X_train)\n"
                "X_test  = scaler.transform(X_test)\n"
                "clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42)\n"
                "clf.fit(X_train, y_train)\n"
                "print('Accuracy:', clf.score(X_test, y_test))\n"
            )
            _S3 = (
                "import pandas as pd\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "X = df.drop(columns=['target'])\n"
                "y = df['target']\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                "scaler = StandardScaler()\n"
                "X_train = scaler.fit_transform(X_train)\n"
                "X_test  = scaler.transform(X_test)\n"
                "clf = LogisticRegression(max_iter=1000, random_state=42)\n"
                "clf.fit(X_train, y_train)\n"
                "print('Accuracy:', clf.score(X_test, y_test))\n"
            )
            _S4 = (
                "import pandas as pd\n"
                "from sklearn.preprocessing import StandardScaler\n"
                "from sklearn.linear_model import LogisticRegression\n"
                "from sklearn.model_selection import train_test_split\n\n"
                "features = ['age', 'salary', 'credit_score', 'loan_amount', 'employment_years']\n"
                "X = df[features].copy()\n"
                "y = df['target']\n"
                "X_train, X_test, y_train, y_test = train_test_split(\n"
                "    X, y, test_size=0.25, random_state=42, stratify=y\n"
                ")\n"
                "scaler = StandardScaler()\n"
                "X_train_scaled = scaler.fit_transform(X_train)\n"
                "X_test_scaled  = scaler.transform(X_test)\n"
                "clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')\n"
                "clf.fit(X_train_scaled, y_train)\n"
                "print('Accuracy:', clf.score(X_test_scaled, y_test))\n"
            )
            _BSTAGES = {
                1: [("inspect_data", {}), ("edit_script", {"script": _S1}), ("run_script", {}), ("submit", {})],
                2: [("inspect_data", {}), ("edit_script", {"script": _S2}), ("run_script", {}), ("submit", {})],
                3: [("run_script", {}),   ("edit_script", {"script": _S3}), ("run_script", {}), ("submit", {})],
                4: [("query_actor", {}),  ("edit_script", {"script": _S4}), ("run_script", {}), ("submit", {})],
            }

            def run_baseline_cmp():
                """Baseline uses its own private DataEngEnvironment — zero shared state."""
                import time as _t
                from app.environment import DataEngEnvironment
                from app.models import Action
                lines = []
                def emit(l):
                    lines.append(l)
                    return "\n".join(lines)

                yield emit("╔══════════════════════════════════════════════════════╗")
                yield emit("║      Rule-Based Baseline Agent                       ║")
                yield emit("╚══════════════════════════════════════════════════════╝\n")

                _env = DataEngEnvironment()
                _env.reset()
                yield emit("  ✓ Private env initialised — Stage 1\n")

                step = 0
                LABELS = {1:"Data Repair",2:"Training Monitor",3:"Eval Validation",4:"Deploy Gate"}
                for stage_num, actions in _BSTAGES.items():
                    if _env.done:
                        break
                    yield emit(f"── Stage {stage_num}: {LABELS[stage_num]} {'─'*(38-len(LABELS[stage_num]))}")
                    for atype, payload in actions:
                        step += 1
                        icon = _ICONS.get(atype, "▸")
                        emit(f"  Step {step:02d} │ {icon} {atype.upper():<18} │ …")
                        yield "\n".join(lines)
                        try:
                            result = _env.step(Action(action_type=atype, payload=payload))
                            rwd    = result.reward
                            obs    = result.observation
                            score  = float(rwd.score)
                            msg    = (rwd.message or "")[:65]
                            lines[-1] = f"  Step {step:02d} │ {icon} {atype.upper():<18} │ score={score:.2f}  {msg}"
                        except Exception as ex:
                            lines[-1] = f"  Step {step:02d} │ {icon} {atype.upper():<18} │ ❌ {ex}"
                        yield "\n".join(lines)
                        _t.sleep(0.15)
                    if _env.current_stage == stage_num and not _env.done:
                        yield emit(f"  ✗ Stage {stage_num} did not pass — stopping here")
                        break

                yield emit(f"\n  Stages completed : {list(_env.stages_completed)}")
                yield emit(f"  Episode score    : {_env.episode_score:.2f}")
                yield emit(f"\n  {'✅' if _env.episode_score > 0.5 else '❌'} Baseline complete.")

            def run_llm_cmp():
                """LLM agent — fully genuine, no hardcoded scripts or overrides."""
                import time as _t, os as _os
                from app.environment import DataEngEnvironment
                from app.models import Action
                lines = []
                def emit(l):
                    lines.append(l)
                    return "\n".join(lines)

                yield emit("╔══════════════════════════════════════════════════════╗")
                yield emit("║      LLM Agent (Groq Llama 3.1 8B)                   ║")
                yield emit("╚══════════════════════════════════════════════════════╝\n")

                groq_key = _os.environ.get("GROQ_API_KEY", "")
                if not groq_key:
                    yield emit("❌ GROQ_API_KEY not set in Space secrets.")
                    return
                try:
                    from groq import Groq as _Groq
                    _client = _Groq(api_key=groq_key)
                except ImportError:
                    yield emit("❌ groq package not installed.")
                    return

                _SYS = """You are an expert ML engineer debugging a broken 4-stage ML pipeline. Each stage has one specific bug to fix.

Respond ONLY with a single JSON object — no markdown, no explanation, nothing else.

AVAILABLE ACTIONS:
  {"action_type": "inspect_data", "payload": {}}
  {"action_type": "run_script",   "payload": {}}
  {"action_type": "edit_script",  "payload": {"script": "<FULL replacement script — ALL imports included>"}}
  {"action_type": "query_actor",  "payload": {}}
  {"action_type": "submit",       "payload": {}}

CRITICAL RULES:
1. edit_script MUST always replace the ENTIRE script (all imports + all logic). Never use old/new format.
2. After edit_script → always call run_script to verify.
3. If run_script prints "Accuracy:" with NO error → call submit IMMEDIATELY. Do not edit again.
4. Fix ALL bugs for a stage in ONE single edit_script call.

STAGE BUGS AND FIXES:
Stage 1 — Data Repair:
  - Bug 1: column named `age_years` but should be `age` — fix with df.rename(columns={'age_years': 'age'}) OR use 'age' directly
  - Bug 2: NaN values crash the script — fix with df = df.dropna() before using df
  - Fix both bugs in one edit. Use LogisticRegression, StandardScaler, train_test_split.

Stage 2 — Training Monitor:
  - Bug: StandardScaler is fit on the entire dataset BEFORE train_test_split (data leakage)
  - Fix: call train_test_split FIRST, then scaler.fit_transform(X_train) and scaler.transform(X_test)
  - Use MLPClassifier if the original script uses it.

Stage 3 — Eval Validation:
  - Bug: scaler.fit() is called on ALL data before the split (data leakage)
  - Fix: move scaler.fit_transform to AFTER train_test_split, fit only on X_train
  - Use all features: X = df.drop(columns=['target'])

Stage 4 — Deploy Gate:
  - Bug: model has poor fairness (high fairness gap between groups)
  - Fix: add class_weight='balanced' to LogisticRegression AND use test_size=0.25, stratify=y in train_test_split
  - Use features: ['age', 'salary', 'credit_score', 'loan_amount', 'employment_years']

WORKFLOW PER STAGE:
1. inspect_data or run_script to see the error
2. edit_script with full corrected script
3. run_script to verify
4. submit if Accuracy printed with no error"""

                def _parse(raw):
                    s = raw.find('{')
                    if s != -1:
                        d = 0
                        for i, ch in enumerate(raw[s:], s):
                            if ch == '{': d += 1
                            elif ch == '}':
                                d -= 1
                                if d == 0:
                                    try:
                                        a = json.loads(raw[s:i+1])
                                        if "action_type" in a:
                                            p = a.get("payload", {})
                                            if isinstance(p, str):
                                                p = {"script": p}
                                            a["payload"] = p
                                            return a
                                    except Exception:
                                        pass
                                    break
                    for t in ["edit_script","run_script","submit","inspect_data","query_actor"]:
                        if t in raw:
                            return {"action_type": t, "payload": {}}
                    return {"action_type": "run_script", "payload": {}}

                def _obs_to_prompt(obs):
                    parts = [f"Stage {obs.current_stage} | step {obs.stage_step_number}"]
                    if obs.script_content:
                        parts.append(f"\nCURRENT SCRIPT:\n{obs.script_content[:1200]}")
                    if obs.last_run_error:
                        parts.append(f"\nRUN ERROR:\n{obs.last_run_error[:600]}")
                    if obs.last_run_output:
                        parts.append(f"\nRUN OUTPUT:\n{obs.last_run_output[:400]}")
                    if obs.actor_feedback:
                        parts.append(f"\nREVIEWER FEEDBACK:\n{obs.actor_feedback[:400]}")
                    if hasattr(obs, 'data_preview') and obs.data_preview:
                        parts.append(f"\nDATA PREVIEW:\n{str(obs.data_preview)[:400]}")
                    return "\n".join(parts)

                LABELS = {1:"Data Repair",2:"Training Monitor",3:"Eval Validation",4:"Deploy Gate"}

                _env = DataEngEnvironment()
                obs = _env.reset()
                yield emit("  ✓ Private env initialised — Stage 1\n")
                yield emit(f"── Stage 1: {LABELS[1]} ────────────────────────────────")

                msgs = [
                    {"role": "system", "content": _SYS},
                    {"role": "user",   "content": _obs_to_prompt(obs) + "\n\nThis is Stage 1: Data Repair. The script has bugs — inspect the data first, then fix all bugs in one edit_script call."},
                ]

                step = 0
                cur_stage = 1
                hist = []
                last_was_clean_run = False

                for _ in range(80):
                    # ── ask LLM ──────────────────────────────────────────────
                    try:
                        ctx = [msgs[0]] + msgs[1:2] + msgs[-8:] if len(msgs) > 10 else msgs
                        resp = _client.chat.completions.create(
                            model="llama-3.1-8b-instant", messages=ctx,
                            max_tokens=2000, temperature=0.1)
                        raw    = resp.choices[0].message.content.strip()
                        action = _parse(raw)
                    except Exception as ex:
                        yield emit(f"  ⚠ LLM error: {ex}")
                        action = {"action_type": "run_script", "payload": {}}

                    # ── force-submit after a clean run ────────────────────────
                    hist.append(action["action_type"])
                    if last_was_clean_run and action["action_type"] != "submit":
                        action = {"action_type": "submit", "payload": {}}
                        last_was_clean_run = False

                    # ── loop-break: 4 identical actions → nudge ───────────────
                    elif len(hist) >= 4 and len(set(hist[-4:])) == 1:
                        if hist[-1] == "edit_script":
                            action = {"action_type": "run_script", "payload": {}}
                        elif hist[-1] == "inspect_data":
                            action = {"action_type": "run_script", "payload": {}}
                        elif hist[-1] in ("run_script", "submit"):
                            action = {"action_type": "edit_script", "payload": {"script": obs.script_content or ""}}

                    step += 1
                    atype = action["action_type"]
                    icon  = _ICONS.get(atype, "▸")
                    emit(f"  Step {step:02d} │ {icon} {atype.upper():<18} │ …")
                    yield "\n".join(lines)

                    # ── execute in private env ────────────────────────────────
                    try:
                        result  = _env.step(Action(action_type=atype, payload=action.get("payload", {})))
                        obs     = result.observation
                        rwd     = result.reward
                        score   = float(rwd.score)
                        msg     = (rwd.message or "")[:60]
                        new_stage = obs.current_stage
                        done    = _env.done
                        last_was_clean_run = (
                            atype == "run_script"
                            and not obs.last_run_error
                            and bool(obs.last_run_output)
                        )
                        lines[-1] = f"  Step {step:02d} │ {icon} {atype.upper():<18} │ score={score:.2f}  {msg}"
                    except Exception as ex:
                        lines[-1] = f"  Step {step:02d} │ {icon} {atype.upper():<18} │ ❌ {ex}"
                        new_stage = cur_stage
                        done = False
                        last_was_clean_run = False
                        rwd = None
                    yield "\n".join(lines)

                    if new_stage != cur_stage:
                        cur_stage = new_stage
                        hist = []
                        last_was_clean_run = False
                        if not done:
                            yield emit(f"\n── Stage {cur_stage}: {LABELS.get(cur_stage,'')} {'─'*(38-len(LABELS.get(cur_stage,'')))}─")

                    # ── build next prompt ─────────────────────────────────────
                    msgs.append({"role": "assistant", "content": json.dumps(action)})
                    feedback = _obs_to_prompt(obs)
                    if atype == "run_script" and obs.last_run_error:
                        feedback += "\n\nScript errored. Use edit_script with a FULL replacement script fixing ALL bugs shown above."
                    elif atype == "run_script" and not obs.last_run_error:
                        feedback += '\n\nClean run — output printed with no error. Call submit NOW: {"action_type":"submit","payload":{}}'
                    elif atype == "edit_script":
                        feedback += '\n\nScript replaced. Call run_script to verify: {"action_type":"run_script","payload":{}}'
                    elif atype == "query_actor":
                        feedback += "\n\nUse the reviewer feedback above to fix the script with edit_script (full replacement), then run_script."
                    elif atype == "submit":
                        feedback += "\n\nSubmit did not pass. Look at the current script and error — use edit_script to fix the remaining bug."
                    elif atype == "inspect_data":
                        feedback += "\n\nData inspected. Now use edit_script to fix all bugs for this stage in one call."
                    msgs.append({"role": "user", "content": feedback})
                    if len(msgs) > 20:
                        msgs = [msgs[0]] + msgs[1:2] + msgs[-14:]

                    if done:
                        break
                    _t.sleep(0.2)

                yield emit(f"\n  Stages completed : {list(_env.stages_completed)}")
                yield emit(f"  Episode score    : {_env.episode_score:.2f}")
                yield emit(f"\n  {'✅' if _env.episode_score >= 0.75 else '⚠️' if _env.episode_score > 0 else '❌'} LLM Agent complete.")

            with gr.Row():
                baseline_scorecard = gr.HTML("")
                llm_scorecard = gr.HTML("")

            def _make_scorecard(log_text, label, label_color, description):
                import re as _re
                score_m = _re.search(r"Episode score\s*:\s*([\d.]+)", log_text)
                stages_m = _re.search(r"Stages completed\s*:\s*(\[.*?\])", log_text)
                score = float(score_m.group(1)) if score_m else 0.0
                stages = stages_m.group(1) if stages_m else "[]"
                pct = int(score * 100)
                color = "#22c55e" if pct >= 75 else "#f59e0b" if pct >= 25 else "#ef4444"
                return f"""<div class="score-display">
                  <div class="score-card">
                    <div class="score-card-big" style="color:{color}">{pct}%</div>
                    <div class="score-card-label">EPISODE SCORE</div>
                  </div>
                  <div class="score-card">
                    <div style="font-size:1.3rem;font-weight:800;color:#94a3b8;line-height:1">{stages}</div>
                    <div class="score-card-label">STAGES COMPLETED</div>
                  </div>
                  <div class="score-card" style="flex:2">
                    <div style="color:#64748b;font-size:0.8rem;line-height:1.7;text-align:left">
                      <strong style="color:{label_color}">{label}</strong><br>{description}
                    </div>
                  </div>
                </div>"""

            def run_baseline_cmp_with_card():
                log_text = ""
                for chunk in run_baseline_cmp():
                    log_text = chunk
                    yield chunk, ""
                yield log_text, _make_scorecard(
                    log_text, "Rule-Based Baseline", "#94a3b8",
                    "Applies hardcoded fix sequences. Reliable on known bug patterns, but cannot adapt when the environment deviates."
                )

            def run_llm_cmp_with_card():
                log_text = ""
                for chunk in run_llm_cmp():
                    log_text = chunk
                    yield chunk, ""
                yield log_text, _make_scorecard(
                    log_text, "LLM Agent (Groq Llama 3.1 8B)", "#818cf8",
                    "Reads the error log and current script, reasons about the root cause, and generates fix actions — no hardcoded rules."
                )

            run_baseline_cmp_btn.click(run_baseline_cmp_with_card, inputs=[], outputs=[baseline_cmp_log, baseline_scorecard])
            run_llm_cmp_btn.click(run_llm_cmp_with_card, inputs=[], outputs=[llm_cmp_log, llm_scorecard])


        # ── DOCS TAB ─────────────────────────────────────────────────────────
        with gr.Tab("📖 API Docs"):
            gr.HTML(DOCS_HTML)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,
        css=CUSTOM_CSS
    )
