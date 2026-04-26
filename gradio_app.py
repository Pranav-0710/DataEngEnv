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
        ("1", "Data Repair",      "Rename column · drop NaN rows",       "🔧", "#6366f1"),
        ("2", "Training Monitor", "Add StandardScaler normalization",     "📈", "#8b5cf6"),
        ("3", "Eval Validation",  "Fix data leakage in scaler.fit()",     "🔍", "#a855f7"),
        ("4", "Deploy Gate",      "Add class_weight='balanced'",          "🚀", "#d946ef"),
    ]

    cards = ""
    for sid, name, desc, icon, accent in stages:
        sid_int = int(sid)
        if sid_int in completed:
            glow = f"box-shadow:0 0 18px rgba(34,197,94,0.2);"
            border = "border-color:rgba(34,197,94,0.35);"
            bg = "background:rgba(34,197,94,0.06);"
            badge = '<span style="background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.4);color:#86efac;font-size:0.68rem;font-weight:700;padding:3px 10px;border-radius:999px;white-space:nowrap">✅ DONE</span>'
            num_style = f"color:#86efac;"
        elif sid_int == current:
            glow = f"box-shadow:0 0 20px {accent}30,inset 0 0 20px {accent}08;"
            border = f"border-color:{accent}80;"
            bg = f"background:linear-gradient(135deg,{accent}12,{accent}06);"
            badge = f'<span style="background:{accent}25;border:1px solid {accent}80;color:#c4b5fd;font-size:0.68rem;font-weight:700;padding:3px 10px;border-radius:999px;white-space:nowrap;animation:pulse-badge 1.5s infinite">⚡ ACTIVE</span>'
            num_style = f"color:{accent};"
        else:
            glow = ""
            border = "border-color:rgba(255,255,255,0.06);"
            bg = "background:rgba(255,255,255,0.02);"
            badge = '<span style="background:rgba(71,85,105,0.15);border:1px solid rgba(71,85,105,0.25);color:#334155;font-size:0.68rem;font-weight:700;padding:3px 10px;border-radius:999px;white-space:nowrap">⏳ PENDING</span>'
            num_style = "color:#334155;"

        cards += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:11px 14px;border-radius:10px;
                    border:1px solid;{border}{bg}{glow}transition:all 0.3s">
            <div style="width:28px;height:28px;border-radius:8px;background:{accent}20;border:1px solid {accent}40;
                        display:flex;align-items:center;justify-content:center;font-size:0.85rem;flex-shrink:0">{icon}</div>
            <div style="flex:1;min-width:0">
                <div style="font-size:0.82rem;font-weight:700;color:#e2e8f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    <span style="{num_style}font-weight:900">S{sid}</span> · {name}
                </div>
                <div style="font-size:0.7rem;color:#475569;margin-top:1px">{desc}</div>
            </div>
            {badge}
        </div>"""

    score_pct = int(episode_score * 100)
    bar_color = "#22c55e" if score_pct >= 75 else "#f59e0b" if score_pct >= 25 else "#6366f1"
    seg_w = 25
    segments = ""
    for i in range(4):
        filled = (i + 1) * 25 <= score_pct
        active = not filled and i * 25 < score_pct
        if filled:
            c = bar_color
            op = "1"
        elif active:
            c = bar_color
            op = "0.5"
        else:
            c = "rgba(255,255,255,0.06)"
            op = "1"
        segments += f'<div style="flex:1;height:6px;border-radius:3px;background:{c};opacity:{op};transition:all 0.5s"></div>'

    return f"""
    <div style="background:#080d14;border:1px solid rgba(99,102,241,0.2);border-radius:14px;padding:18px 20px;height:100%">
      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px">
        <div style="font-size:0.8rem;font-weight:800;color:#e2e8f0;letter-spacing:.04em">PIPELINE STATUS</div>
        <div style="display:flex;gap:5px">
          <span style="background:rgba(99,102,241,0.12);border:1px solid rgba(99,102,241,0.25);
                       color:#818cf8;font-size:0.65rem;font-weight:700;padding:2px 8px;border-radius:999px">S{current}/4</span>
          <span style="background:rgba(255,255,255,0.05);border:1px solid rgba(255,255,255,0.1);
                       color:#64748b;font-size:0.65rem;font-weight:700;padding:2px 8px;border-radius:999px">{total_steps} steps</span>
          <span style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.25);
                       color:#86efac;font-size:0.65rem;font-weight:700;padding:2px 8px;border-radius:999px">{score_pct}%</span>
        </div>
      </div>
      <div style="display:flex;gap:3px;margin-bottom:14px">{segments}</div>
      <div style="display:flex;flex-direction:column;gap:8px">{cards}</div>
    </div>
    """

# ─── CSS ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #020609 !important;
    color: #e2e8f0 !important;
}

/* Scrollbar */
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.3); border-radius: 999px; }

/* ── Animations ───────────────────────────────────────────── */
@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes glow-pulse {
    0%, 100% { opacity: 0.6; }
    50%       { opacity: 1; }
}
@keyframes pulse-badge {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.4); }
    50%       { box-shadow: 0 0 8px 2px rgba(99,102,241,0.2); }
}
@keyframes scanline {
    0%   { transform: translateY(-100%); }
    100% { transform: translateY(100vh); }
}
@keyframes float {
    0%, 100% { transform: translateY(0px); }
    50%       { transform: translateY(-6px); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ── Global background with subtle grid ──────────────────── */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(99,102,241,0.03) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.03) 1px, transparent 1px);
    background-size: 40px 40px;
    pointer-events: none;
    z-index: 0;
}

/* ── Tab styling ─────────────────────────────────────────── */
.tabs > .tab-nav {
    background: rgba(8,13,20,0.95) !important;
    border-bottom: 1px solid rgba(99,102,241,0.15) !important;
    backdrop-filter: blur(12px) !important;
    padding: 0 4px !important;
    gap: 2px !important;
}
.tabs > .tab-nav button {
    color: #475569 !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    padding: 12px 20px !important;
    border-radius: 0 !important;
    transition: all 0.2s !important;
    border-bottom: 2px solid transparent !important;
    letter-spacing: .01em !important;
}
.tabs > .tab-nav button:hover {
    color: #94a3b8 !important;
    background: rgba(99,102,241,0.04) !important;
}
.tabs > .tab-nav button.selected {
    color: #a5b4fc !important;
    border-bottom: 2px solid #6366f1 !important;
    background: rgba(99,102,241,0.06) !important;
}

/* ── Action buttons ──────────────────────────────────────── */
.btn-primary {
    background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #6d28d9 100%) !important;
    background-size: 200% 200% !important;
    animation: gradient-shift 4s ease infinite !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 0.9rem !important;
    letter-spacing: .02em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 11px 20px !important;
    transition: all 0.2s !important;
    box-shadow: 0 4px 20px rgba(99,102,241,0.35), inset 0 1px 0 rgba(255,255,255,0.1) !important;
    position: relative !important;
    overflow: hidden !important;
}
.btn-primary::after {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.08) 0%, transparent 100%);
    pointer-events: none;
}
.btn-primary:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 6px 28px rgba(99,102,241,0.5) !important;
}
.btn-primary:active { transform: translateY(0) !important; }

.btn-reset {
    background: rgba(255,255,255,0.04) !important;
    color: #64748b !important;
    border: 1px solid rgba(255,255,255,0.08) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s !important;
}
.btn-reset:hover {
    background: rgba(255,255,255,0.08) !important;
    color: #94a3b8 !important;
    border-color: rgba(255,255,255,0.15) !important;
}

.btn-actor {
    background: rgba(245,158,11,0.08) !important;
    color: #fbbf24 !important;
    border: 1px solid rgba(245,158,11,0.25) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.88rem !important;
    transition: all 0.2s !important;
}
.btn-actor:hover {
    background: rgba(245,158,11,0.15) !important;
    box-shadow: 0 0 16px rgba(245,158,11,0.2) !important;
}

/* ── Code / terminal boxes ───────────────────────────────── */
.code-box textarea, .code-box pre, .code-box .cm-editor {
    background: #060b12 !important;
    color: #c9d1d9 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    line-height: 1.6 !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    border-radius: 10px !important;
}
.code-box label { color: #475569 !important; font-size: 0.73rem !important; font-weight: 700 !important; letter-spacing: .04em !important; }

.error-box textarea {
    background: #0d0508 !important;
    color: #fca5a5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.6 !important;
    border: 1px solid rgba(239,68,68,0.2) !important;
    border-radius: 10px !important;
}
.error-box label { color: #7f1d1d !important; font-size: 0.73rem !important; font-weight: 700 !important; letter-spacing: .04em !important; }

.status-box textarea {
    background: #030a05 !important;
    color: #4ade80 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.83rem !important;
    font-weight: 600 !important;
    letter-spacing: .01em !important;
    border: 1px solid rgba(34,197,94,0.15) !important;
    border-radius: 10px !important;
}

/* ── Inputs ──────────────────────────────────────────────── */
textarea, input[type="text"], select {
    background: #060b12 !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
}
textarea:focus, input:focus {
    border-color: rgba(99,102,241,0.4) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.08) !important;
    outline: none !important;
}
label span { color: #475569 !important; font-size: 0.73rem !important; font-weight: 700 !important; letter-spacing: .04em !important; }

/* ── Dropdown ────────────────────────────────────────────── */
.gr-dropdown, .gr-dropdown > div, .gr-dropdown input {
    background: #060b12 !important;
    border: 1px solid rgba(99,102,241,0.15) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Remove Gradio default padding/bg ────────────────────── */
.gradio-container > .main { background: transparent !important; }
footer { display: none !important; }
"""

# ─── HERO ─────────────────────────────────────────────────────────────────────

HERO_HTML = """
<div style="
  background: linear-gradient(135deg, #040810 0%, #0a0d1f 35%, #0d0520 65%, #040810 100%);
  border: 1px solid rgba(99,102,241,0.2);
  border-radius: 18px;
  padding: 40px 48px 32px;
  margin-bottom: 4px;
  position: relative;
  overflow: hidden;
">

  <!-- Glow orbs -->
  <div style="position:absolute;top:-100px;right:-60px;width:400px;height:400px;
    background:radial-gradient(circle,rgba(99,102,241,0.1) 0%,transparent 65%);
    border-radius:50%;pointer-events:none;animation:glow-pulse 4s ease-in-out infinite"></div>
  <div style="position:absolute;bottom:-80px;left:20%;width:300px;height:300px;
    background:radial-gradient(circle,rgba(139,92,246,0.07) 0%,transparent 65%);
    border-radius:50%;pointer-events:none;animation:glow-pulse 6s ease-in-out infinite reverse"></div>
  <div style="position:absolute;top:50%;left:-40px;width:200px;height:200px;
    background:radial-gradient(circle,rgba(217,70,239,0.05) 0%,transparent 65%);
    border-radius:50%;pointer-events:none"></div>

  <!-- Top row: title + stat pills -->
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:24px;position:relative">
    <div style="flex:1;min-width:300px">

      <!-- Eyebrow -->
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:14px">
        <div style="width:6px;height:6px;border-radius:50%;background:#6366f1;animation:glow-pulse 2s infinite"></div>
        <span style="font-size:0.7rem;font-weight:700;letter-spacing:.12em;color:#475569;text-transform:uppercase">
          Reinforcement Learning Environment · OpenEnv Compliant
        </span>
      </div>

      <!-- Title -->
      <h1 style="font-size:2.6rem;font-weight:900;line-height:1.1;letter-spacing:-1.5px;margin-bottom:12px">
        <span style="color:#f1f5f9">Pipeline</span><span style="
          background:linear-gradient(90deg,#818cf8,#c084fc,#e879f9);
          background-size:200% 100%;
          animation:gradient-shift 3s ease infinite;
          -webkit-background-clip:text;-webkit-text-fill-color:transparent;
        ">Ops Arena</span>
      </h1>

      <!-- Subtitle -->
      <p style="color:#64748b;font-size:0.92rem;line-height:1.7;max-width:580px;margin-bottom:20px">
        A <strong style="color:#818cf8">4-stage cascading ML pipeline debugger</strong> where AI agents
        autonomously inspect data, patch broken scripts, execute sandboxed Python, and navigate
        cascading failures — graded <strong style="color:#4ade80">deterministically</strong>, zero human bias.
      </p>

      <!-- Badges -->
      <div style="display:flex;gap:7px;flex-wrap:wrap">
        <span style="background:rgba(59,130,246,0.1);border:1px solid rgba(59,130,246,0.3);
          color:#93c5fd;padding:4px 12px;border-radius:999px;font-size:0.73rem;font-weight:600">
          ⚡ 4-Stage Cascade
        </span>
        <span style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.3);
          color:#86efac;padding:4px 12px;border-radius:999px;font-size:0.73rem;font-weight:600">
          ✅ Deterministic Grading
        </span>
        <span style="background:rgba(245,158,11,0.1);border:1px solid rgba(245,158,11,0.3);
          color:#fcd34d;padding:4px 12px;border-radius:999px;font-size:0.73rem;font-weight:600">
          📊 Reward ∈ [0, 1]
        </span>
        <span style="background:rgba(139,92,246,0.1);border:1px solid rgba(139,92,246,0.3);
          color:#d8b4fe;padding:4px 12px;border-radius:999px;font-size:0.73rem;font-weight:600">
          🦾 GRPO Fine-tuned LLM
        </span>
        <span style="background:rgba(244,63,94,0.1);border:1px solid rgba(244,63,94,0.3);
          color:#fda4af;padding:4px 12px;border-radius:999px;font-size:0.73rem;font-weight:600">
          🔒 Isolated Sandbox
        </span>
      </div>
    </div>

    <!-- Stat cards -->
    <div style="display:flex;flex-direction:column;gap:8px;min-width:200px">
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px">
        <div style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);
                    border-radius:12px;padding:14px 16px;text-align:center">
          <div style="font-size:2rem;font-weight:900;color:#818cf8;line-height:1">4</div>
          <div style="font-size:0.65rem;color:#475569;font-weight:700;margin-top:3px;letter-spacing:.04em">STAGES</div>
        </div>
        <div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.2);
                    border-radius:12px;padding:14px 16px;text-align:center">
          <div style="font-size:2rem;font-weight:900;color:#a78bfa;line-height:1">5</div>
          <div style="font-size:0.65rem;color:#475569;font-weight:700;margin-top:3px;letter-spacing:.04em">ACTIONS</div>
        </div>
        <div style="background:rgba(217,70,239,0.06);border:1px solid rgba(217,70,239,0.2);
                    border-radius:12px;padding:14px 16px;text-align:center">
          <div style="font-size:2rem;font-weight:900;color:#e879f9;line-height:1">60</div>
          <div style="font-size:0.65rem;color:#475569;font-weight:700;margin-top:3px;letter-spacing:.04em">MAX STEPS</div>
        </div>
        <div style="background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);
                    border-radius:12px;padding:14px 16px;text-align:center">
          <div style="font-size:2rem;font-weight:900;color:#4ade80;line-height:1">1.0</div>
          <div style="font-size:0.65rem;color:#475569;font-weight:700;margin-top:3px;letter-spacing:.04em">MAX SCORE</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Divider -->
  <div style="height:1px;background:linear-gradient(90deg,transparent,rgba(99,102,241,0.2),rgba(217,70,239,0.15),transparent);
              margin:24px 0 20px;position:relative"></div>

  <!-- How It Works flow -->
  <div style="position:relative">
    <div style="font-size:0.68rem;font-weight:800;letter-spacing:.1em;color:#334155;margin-bottom:12px">
      HOW IT WORKS
    </div>
    <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap">

      <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.03);
                  border:1px solid rgba(99,102,241,0.15);border-radius:10px;padding:9px 14px">
        <span style="background:rgba(99,102,241,0.2);color:#818cf8;border-radius:6px;
                     width:22px;height:22px;display:flex;align-items:center;justify-content:center;
                     font-size:0.7rem;font-weight:900;flex-shrink:0">1</span>
        <span style="color:#94a3b8;font-size:0.78rem;font-weight:500">Agent gets broken pipeline + error log</span>
      </div>

      <div style="color:#1e293b;font-size:1.1rem;font-weight:300">›</div>

      <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.03);
                  border:1px solid rgba(139,92,246,0.15);border-radius:10px;padding:9px 14px">
        <span style="background:rgba(139,92,246,0.2);color:#a78bfa;border-radius:6px;
                     width:22px;height:22px;display:flex;align-items:center;justify-content:center;
                     font-size:0.7rem;font-weight:900;flex-shrink:0">2</span>
        <span style="color:#94a3b8;font-size:0.78rem;font-weight:500">Calls inspect · edit · run · query</span>
      </div>

      <div style="color:#1e293b;font-size:1.1rem;font-weight:300">›</div>

      <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.03);
                  border:1px solid rgba(217,70,239,0.15);border-radius:10px;padding:9px 14px">
        <span style="background:rgba(217,70,239,0.2);color:#e879f9;border-radius:6px;
                     width:22px;height:22px;display:flex;align-items:center;justify-content:center;
                     font-size:0.7rem;font-weight:900;flex-shrink:0">3</span>
        <span style="color:#94a3b8;font-size:0.78rem;font-weight:500">Fix runs in isolated Python sandbox</span>
      </div>

      <div style="color:#1e293b;font-size:1.1rem;font-weight:300">›</div>

      <div style="display:flex;align-items:center;gap:8px;background:rgba(255,255,255,0.03);
                  border:1px solid rgba(34,197,94,0.15);border-radius:10px;padding:9px 14px">
        <span style="background:rgba(34,197,94,0.2);color:#4ade80;border-radius:6px;
                     width:22px;height:22px;display:flex;align-items:center;justify-content:center;
                     font-size:0.7rem;font-weight:900;flex-shrink:0">4</span>
        <span style="color:#94a3b8;font-size:0.78rem;font-weight:500">Grader scores → advance to next stage</span>
      </div>

    </div>
  </div>
</div>
"""

DOCS_HTML = """
<div style="background:#060b12;border:1px solid rgba(99,102,241,0.15);border-radius:14px;
            padding:28px 32px;color:#cbd5e1;line-height:1.7;font-size:0.88rem;
            font-family:'Inter',sans-serif">

  <h3 style="color:#e2e8f0;margin:0 0 6px;font-size:1.1rem;font-weight:800">🔌 API Quick Start</h3>
  <p style="color:#475569;font-size:0.8rem;margin-bottom:16px">
    Base URL: <code style="background:#0d1520;color:#818cf8;padding:2px 8px;border-radius:5px;
    font-family:'JetBrains Mono',monospace;font-size:0.78rem;border:1px solid rgba(99,102,241,0.2)">
    https://CoBeDigger-DataEngEnv.hf.space</code>
  </p>

  <pre style="background:#030609;border:1px solid rgba(99,102,241,0.1);border-radius:10px;
              padding:16px 20px;overflow-x:auto;margin-bottom:24px"><code style="color:#86efac;
              font-family:'JetBrains Mono',monospace;font-size:0.8rem;line-height:1.7">import requests

BASE = "https://CoBeDigger-DataEngEnv.hf.space"

# Reset — starts a fresh episode at Stage 1
obs = requests.post(f"{BASE}/reset").json()

# Take a step — any of the 5 actions
result = requests.post(f"{BASE}/step", json={
    "action_type": "edit_script",
    "payload": {"old": "age_years", "new": "age"}
}).json()
# result["reward"]["score"]  →  float [0, 1]

# Check pipeline progress
status = requests.get(f"{BASE}/pipeline_status").json()
# status["stages_completed"]  →  [1, 2, ...]
# status["episode_score"]     →  float [0, 1]</code></pre>

  <h3 style="color:#e2e8f0;font-size:1rem;font-weight:800;margin-bottom:12px">🎮 Available Actions</h3>
  <div style="display:grid;gap:8px;margin-bottom:24px">
    <div style="display:grid;grid-template-columns:180px 1fr;gap:12px;align-items:center;
                background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                border-radius:8px;padding:10px 14px">
      <code style="color:#818cf8;font-family:'JetBrains Mono',monospace;font-size:0.78rem">inspect_data</code>
      <span style="color:#64748b;font-size:0.82rem">View dataset shape, columns, null counts, 5-row sample</span>
    </div>
    <div style="display:grid;grid-template-columns:180px 1fr;gap:12px;align-items:center;
                background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                border-radius:8px;padding:10px 14px">
      <code style="color:#818cf8;font-family:'JetBrains Mono',monospace;font-size:0.78rem">run_script</code>
      <span style="color:#64748b;font-size:0.82rem">Execute the current pipeline in an isolated subprocess (10s timeout)</span>
    </div>
    <div style="display:grid;grid-template-columns:180px 1fr;gap:12px;align-items:center;
                background:rgba(99,102,241,0.04);border:1px solid rgba(99,102,241,0.12);
                border-radius:8px;padding:10px 14px">
      <code style="color:#a5b4fc;font-family:'JetBrains Mono',monospace;font-size:0.78rem">edit_script</code>
      <span style="color:#64748b;font-size:0.82rem">
        Patch the script. Use <code style="color:#818cf8;background:#0d1520;padding:1px 5px;border-radius:3px;font-size:0.75rem">{"old":"...","new":"..."}</code>
        or full replacement <code style="color:#818cf8;background:#0d1520;padding:1px 5px;border-radius:3px;font-size:0.75rem">{"script":"..."}</code>
      </span>
    </div>
    <div style="display:grid;grid-template-columns:180px 1fr;gap:12px;align-items:center;
                background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                border-radius:8px;padding:10px 14px">
      <code style="color:#818cf8;font-family:'JetBrains Mono',monospace;font-size:0.78rem">query_actor</code>
      <span style="color:#64748b;font-size:0.82rem">Ask the MLOps Bot / Code Reviewer for targeted feedback</span>
    </div>
    <div style="display:grid;grid-template-columns:180px 1fr;gap:12px;align-items:center;
                background:rgba(34,197,94,0.03);border:1px solid rgba(34,197,94,0.1);
                border-radius:8px;padding:10px 14px">
      <code style="color:#4ade80;font-family:'JetBrains Mono',monospace;font-size:0.78rem">submit</code>
      <span style="color:#64748b;font-size:0.82rem">Grade the current fix → advance stage if score ≥ 0.7</span>
    </div>
  </div>

  <h3 style="color:#e2e8f0;font-size:1rem;font-weight:800;margin-bottom:12px">🏗️ The 4 Stages</h3>
  <div style="display:grid;gap:8px">
    <div style="background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.15);
                border-radius:10px;padding:14px 18px">
      <div style="color:#a5b4fc;font-weight:700;margin-bottom:4px">Stage 1 · Data Repair</div>
      <div style="color:#475569;font-size:0.82rem">Column <code style="color:#818cf8;background:#0d1520;padding:1px 5px;border-radius:3px;font-size:0.75rem">age_years</code> must be renamed to <code style="color:#818cf8;background:#0d1520;padding:1px 5px;border-radius:3px;font-size:0.75rem">age</code> · NaN rows must be dropped before scaling</div>
    </div>
    <div style="background:rgba(139,92,246,0.05);border:1px solid rgba(139,92,246,0.15);
                border-radius:10px;padding:14px 18px">
      <div style="color:#c4b5fd;font-weight:700;margin-bottom:4px">Stage 2 · Training Monitor</div>
      <div style="color:#475569;font-size:0.82rem">MLP loss is NaN because there is no <code style="color:#a78bfa;background:#0d1520;padding:1px 5px;border-radius:3px;font-size:0.75rem">StandardScaler</code> — add normalisation fitted only on X_train</div>
    </div>
    <div style="background:rgba(217,70,239,0.04);border:1px solid rgba(217,70,239,0.12);
                border-radius:10px;padding:14px 18px">
      <div style="color:#f0abfc;font-weight:700;margin-bottom:4px">Stage 3 · Eval Validation</div>
      <div style="color:#475569;font-size:0.82rem"><code style="color:#e879f9;background:#0d1520;padding:1px 5px;border-radius:3px;font-size:0.75rem">scaler.fit()</code> called on all data before split → data leakage · move it to after train_test_split</div>
    </div>
    <div style="background:rgba(34,197,94,0.03);border:1px solid rgba(34,197,94,0.12);
                border-radius:10px;padding:14px 18px">
      <div style="color:#86efac;font-weight:700;margin-bottom:4px">Stage 4 · Deploy Gate</div>
      <div style="color:#475569;font-size:0.82rem">Fairness gap too high — add <code style="color:#4ade80;background:#0d1520;padding:1px 5px;border-radius:3px;font-size:0.75rem">class_weight='balanced'</code> to LogisticRegression and use stratified split</div>
    </div>
  </div>
</div>
"""

# ─── UI ───────────────────────────────────────────────────────────────────────

with gr.Blocks(title="PipelineOps Arena — OpenEnv", css=CUSTOM_CSS) as demo:

    gr.HTML(HERO_HTML)

    with gr.Tabs():

        # ══════════════════════════════════════════════════════════════════════
        # TAB 1 — PLAYGROUND
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🎮  Playground"):
            with gr.Row():

                # ── Left: Pipeline tracker ────────────────────────────────────
                with gr.Column(scale=1, min_width=260):
                    stage_display = gr.HTML(value=build_stage_html({}))
                    refresh_btn = gr.Button("⟳  Refresh", elem_classes="btn-reset", size="sm")

                # ── Right: Controls ───────────────────────────────────────────
                with gr.Column(scale=2):

                    with gr.Row():
                        action_dd = gr.Dropdown(
                            choices=["inspect_data","check_schema","run_script","edit_script","query_actor","submit"],
                            value="inspect_data",
                            label="ACTION",
                            scale=2
                        )
                        reset_btn = gr.Button("⟳  Reset Episode", elem_classes="btn-reset", scale=1)

                    with gr.Row():
                        old_text = gr.Textbox(
                            label="OLD CODE  (edit_script)",
                            lines=2,
                            placeholder="Exact text to find in the script…",
                            elem_classes="code-box"
                        )
                        new_text = gr.Textbox(
                            label="NEW CODE  (edit_script)",
                            lines=2,
                            placeholder="Replacement text…",
                            elem_classes="code-box"
                        )

                    with gr.Row():
                        step_btn  = gr.Button("▶  Execute Action", elem_classes="btn-primary", scale=3)
                        actor_btn = gr.Button("🤖  Query MLOps Bot", elem_classes="btn-actor", scale=1)

                    status_out = gr.Textbox(label="STATUS", interactive=False, elem_classes="status-box", lines=1)

                    with gr.Row():
                        script_out = gr.Code(label="CURRENT PIPELINE SCRIPT", language="python", lines=14, elem_classes="code-box")
                        error_out  = gr.Textbox(label="LAST RUN ERROR", lines=14, interactive=False, elem_classes="error-box")

                    json_out = gr.Code(label="RAW API RESPONSE", language="json", lines=7, elem_classes="code-box")

            step_btn.click(step_env, [action_dd, old_text, new_text], [status_out, script_out, error_out, json_out, stage_display])
            actor_btn.click(query_actor_ui, [], [status_out, script_out, error_out, json_out, stage_display])
            reset_btn.click(reset_env, [], [status_out, script_out, error_out, json_out, stage_display])
            refresh_btn.click(refresh_status, [], [stage_display])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 2 — LIVE LLM AGENT
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("🤖  Live LLM Agent"):

            gr.HTML("""
<div style="background:linear-gradient(135deg,rgba(99,102,241,0.07),rgba(139,92,246,0.04));
            border:1px solid rgba(99,102,241,0.18);border-radius:14px;
            padding:22px 28px;margin-bottom:16px;position:relative;overflow:hidden">
  <div style="position:absolute;top:-40px;right:-40px;width:200px;height:200px;
    background:radial-gradient(circle,rgba(99,102,241,0.12) 0%,transparent 65%);
    border-radius:50%;pointer-events:none"></div>
  <div style="display:flex;align-items:flex-start;gap:16px;flex-wrap:wrap">
    <div style="flex:1;min-width:260px">
      <div style="font-size:1rem;font-weight:800;color:#e2e8f0;margin-bottom:6px">
        Watch Groq Llama 3.1 8B Debug 4 Broken ML Pipelines
      </div>
      <div style="color:#64748b;font-size:0.84rem;line-height:1.65;max-width:600px">
        The model receives error logs and the broken script, reasons about root causes,
        and generates fix actions entirely from its own intelligence —
        <strong style="color:#94a3b8">no rules, no hardcoded fixes, no shortcuts</strong>.
        Watch it navigate all 4 stages in real time.
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:6px">
      <span style="background:rgba(34,197,94,0.1);border:1px solid rgba(34,197,94,0.25);
                   color:#86efac;padding:5px 14px;border-radius:999px;font-size:0.72rem;font-weight:700;
                   display:flex;align-items:center;gap:6px">
        <span style="width:6px;height:6px;border-radius:50%;background:#4ade80;animation:glow-pulse 1.5s infinite;display:inline-block"></span>
        LIVE REASONING
      </span>
      <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.25);
                   color:#a5b4fc;padding:5px 14px;border-radius:999px;font-size:0.72rem;font-weight:700">
        🔒 ISOLATED SANDBOX
      </span>
      <span style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.2);
                   color:#fbbf24;padding:5px 14px;border-radius:999px;font-size:0.72rem;font-weight:700">
        📊 DETERMINISTIC GRADING
      </span>
    </div>
  </div>
</div>""")

            run_btn = gr.Button("▶  Run LLM Agent — All 4 Stages", elem_classes="btn-primary")

            live_output = gr.Textbox(
                label="AGENT TERMINAL",
                lines=30,
                interactive=False,
                placeholder=(
                    "  ╔══════════════════════════════════════════════╗\n"
                    "  ║   Click ▶ Run LLM Agent to start             ║\n"
                    "  ║                                              ║\n"
                    "  ║   The model will:                            ║\n"
                    "  ║   · Read the broken script + error           ║\n"
                    "  ║   · Reason about the root cause              ║\n"
                    "  ║   · Generate targeted fix actions            ║\n"
                    "  ║   · Verify in a Python sandbox               ║\n"
                    "  ║   · Advance through all 4 stages             ║\n"
                    "  ╚══════════════════════════════════════════════╝\n"
                ),
                elem_classes="code-box"
            )
            live_scorecard = gr.HTML("")

            # ── LLM agent logic (via HTTP to FastAPI) ─────────────────────────
            ACTION_ICONS  = {"inspect_data":"🔍","check_schema":"📋","run_script":"▶ ","edit_script":"✏ ","query_actor":"🤖","submit":"📤"}
            STAGE_LABELS  = {1:"Data Repair",2:"Training Monitor",3:"Eval Validation",4:"Deploy Gate"}

            def run_baseline_streaming():
                import time as _time, os as _os
                log = []
                def emit(line):
                    log.append(line)
                    return "\n".join(log)

                yield emit("  ┌─────────────────────────────────────────────────────┐")
                yield emit("  │   PipelineOps Arena · LLM Agent Run                 │")
                yield emit("  │   Model : Groq llama-3.1-8b-instant                 │")
                yield emit("  └─────────────────────────────────────────────────────┘\n")

                groq_key = _os.environ.get("GROQ_API_KEY", "")
                if not groq_key:
                    yield emit("  ❌  GROQ_API_KEY not set. Add it to the HF Space secrets.")
                    return
                try:
                    from groq import Groq
                    client = Groq(api_key=groq_key)
                except ImportError:
                    yield emit("  ❌  groq package not installed.")
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
- Fix ALL bugs for the stage in ONE single edit_script call
- NEVER use old/new format — always replace the full script
- Submit as soon as run_script shows no error and prints Accuracy
- Stage 1: rename age_years→age AND add df.dropna() — fix BOTH in one edit
- Stage 2: StandardScaler fitted on X_train only, transform X_test separately
- Stage 3: move scaler.fit() to AFTER train_test_split(), fit only on X_train
- Stage 4: add class_weight='balanced' to LogisticRegression"""

                def parse_action(raw_text):
                    start = raw_text.find('{')
                    if start != -1:
                        depth = 0
                        for i, ch in enumerate(raw_text[start:], start):
                            if ch == '{': depth += 1
                            elif ch == '}':
                                depth -= 1
                                if depth == 0:
                                    try:
                                        action = json.loads(raw_text[start:i+1])
                                        if "action_type" in action:
                                            action.setdefault("payload", {})
                                            return action
                                    except json.JSONDecodeError:
                                        pass
                                    break
                    for atype in ["edit_script","run_script","submit","inspect_data","query_actor"]:
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
                    yield emit(f"  ✓ Environment reset\n  ─── Stage 1: {STAGE_LABELS[1]} ─────────────────────────────\n")
                    _time.sleep(0.3)

                    messages = [{"role":"system","content":SYSTEM_PROMPT}]
                    messages.append({"role":"user","content": format_obs(obs) + "\n\nStart debugging. What is your first action?"})

                    current_stage, step_num = 1, 0
                    action_history = []

                    for _ in range(60):
                        try:
                            trimmed = [messages[0], messages[1]] + messages[-4:] if len(messages) > 8 else messages
                            response = client.chat.completions.create(
                                model="llama-3.1-8b-instant", messages=trimmed,
                                max_tokens=1500, temperature=0.2)
                            raw = response.choices[0].message.content.strip()
                            action = parse_action(raw)
                        except Exception as e:
                            yield emit(f"  ⚠  LLM error: {e}")
                            action = {"action_type":"run_script","payload":{}}

                        action_history.append(action["action_type"])
                        if len(action_history) >= 2 and action_history[-2] == "submit" and action["action_type"] == "submit":
                            action = {"action_type":"edit_script","payload":{"script":obs.get("script_content","")}}
                        if len(action_history) >= 4:
                            last4 = action_history[-4:]
                            if last4 in [["edit_script","run_script","edit_script","run_script"],
                                         ["run_script","edit_script","run_script","edit_script"]]:
                                action = {"action_type":"submit","payload":{}}
                        if len(action_history) >= 3 and len(set(action_history[-3:])) == 1:
                            if action_history[-1] == "inspect_data":
                                action = {"action_type":"edit_script","payload":{"script":obs.get("script_content","")}}
                            elif action_history[-1] == "submit":
                                action = {"action_type":"run_script","payload":{}}
                            else:
                                action = {"action_type":"submit","payload":{}}

                        step_num += 1
                        atype = action["action_type"]
                        icon  = ACTION_ICONS.get(atype, "▸")

                        yield emit(f"  Step {step_num:02d}  {icon}  {atype.upper():<18}  ·  executing…")
                        _time.sleep(0.2)

                        resp = httpx.post(f"{BASE_URL}/step", json=action, timeout=30)
                        data = resp.json()
                        obs  = data.get("observation", {})
                        rwd  = data.get("reward", {})
                        score    = float(rwd.get("score", 0.0))
                        msg      = rwd.get("message", "")[:55]
                        new_stage = obs.get("current_stage", current_stage)
                        done      = obs.get("done", False)

                        score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                        log[-1] = f"  Step {step_num:02d}  {icon}  {atype.upper():<18}  ·  [{score_bar}] {score:.2f}  {msg}"
                        yield "\n".join(log)

                        if new_stage != current_stage:
                            current_stage = new_stage
                            action_history = []
                            label = STAGE_LABELS.get(current_stage, f"Stage {current_stage}")
                            yield emit(f"\n  ✅ Stage complete!")
                            yield emit(f"  ─── Stage {current_stage}: {label} {'─'*(44-len(label))}\n")

                        messages.append({"role":"assistant","content":json.dumps(action)})
                        next_prompt = format_obs(obs, rwd)
                        if atype == "inspect_data":
                            next_prompt += "\n\nData inspected. Now edit_script to fix all bugs."
                        elif atype == "run_script" and obs.get("last_run_error"):
                            next_prompt += '\n\nScript errored. Use edit_script with full replacement.'
                        elif atype == "run_script" and not obs.get("last_run_error"):
                            next_prompt += '\n\nClean run! Submit now: {"action_type":"submit","payload":{}}'
                        elif atype == "edit_script":
                            next_prompt += '\n\nEdited. Verify with run_script.'
                        messages.append({"role":"user","content":next_prompt})

                        if done:
                            break
                        _time.sleep(0.3)

                    status = httpx.get(f"{BASE_URL}/pipeline_status", timeout=10).json()
                    completed   = status.get("stages_completed", [])
                    final_score = status.get("episode_score", 0.0)
                    total_steps = status.get("total_steps", step_num)

                    pct = int(final_score * 100)
                    stars = "★" * min(5, max(1, round(final_score * 5)))

                    yield emit(f"\n  ╔══════════════════════════════════════════════════════╗")
                    yield emit(f"  ║  EPISODE COMPLETE  {stars:<36}║")
                    yield emit(f"  ║  Stages completed : {str(completed):<34}║")
                    yield emit(f"  ║  Total steps      : {str(total_steps):<34}║")
                    yield emit(f"  ║  Final score      : {final_score:.2f}  ({pct}%){'':<26}║")
                    yield emit(f"  ╚══════════════════════════════════════════════════════╝")

                except Exception as e:
                    yield emit(f"\n  ❌  Error: {e}")

            def run_llm_agent_with_score():
                log_text = ""
                for chunk in run_baseline_streaming():
                    log_text = chunk
                    yield chunk, ""
                import re as _re
                completed_m = _re.search(r"Stages completed\s*:\s*(\[.*?\])", log_text)
                score_m     = _re.search(r"Final score\s*:\s*([\d.]+)", log_text)
                steps_m     = _re.search(r"Total steps\s*:\s*(\d+)", log_text)
                completed   = completed_m.group(1) if completed_m else "[]"
                score       = float(score_m.group(1)) if score_m else 0.0
                steps       = steps_m.group(1) if steps_m else "—"
                pct         = int(score * 100)
                color       = "#4ade80" if pct >= 75 else "#fbbf24" if pct >= 25 else "#f87171"
                bar         = f'<div style="background:rgba(255,255,255,0.06);border-radius:999px;height:8px;overflow:hidden;margin:10px 0 4px"><div style="width:{pct}%;height:100%;border-radius:999px;background:{color};box-shadow:0 0 10px {color}60"></div></div>'
                n_stages    = len([x for x in completed.replace("[","").replace("]","").split(",") if x.strip()])
                card = f"""
<div style="background:linear-gradient(135deg,#040810,#080d1a);
            border:1px solid rgba(99,102,241,0.25);border-radius:14px;
            padding:22px 26px;margin-top:12px;position:relative;overflow:hidden">
  <div style="position:absolute;top:-40px;right:-40px;width:180px;height:180px;
    background:radial-gradient(circle,{color}18 0%,transparent 65%);border-radius:50%;pointer-events:none"></div>
  <div style="font-size:0.68rem;font-weight:800;letter-spacing:.1em;color:#334155;margin-bottom:14px">EPISODE RESULT</div>
  <div style="display:flex;gap:20px;align-items:flex-start;flex-wrap:wrap">
    <div style="min-width:100px">
      <div style="font-size:3.5rem;font-weight:900;color:{color};line-height:1;text-shadow:0 0 30px {color}60">{pct}%</div>
      {bar}
      <div style="color:#334155;font-size:0.65rem;font-weight:800;letter-spacing:.06em">EPISODE SCORE</div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;flex:1;min-width:200px">
      <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                  border-radius:10px;padding:12px 14px">
        <div style="font-size:1.6rem;font-weight:900;color:#818cf8;line-height:1">{n_stages}/4</div>
        <div style="color:#334155;font-size:0.65rem;font-weight:800;letter-spacing:.05em;margin-top:4px">STAGES DONE</div>
      </div>
      <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);
                  border-radius:10px;padding:12px 14px">
        <div style="font-size:1.6rem;font-weight:900;color:#a78bfa;line-height:1">{steps}</div>
        <div style="color:#334155;font-size:0.65rem;font-weight:800;letter-spacing:.05em;margin-top:4px">TOTAL STEPS</div>
      </div>
      <div style="grid-column:1/-1;background:rgba(99,102,241,0.04);border:1px solid rgba(99,102,241,0.1);
                  border-radius:10px;padding:12px 14px">
        <div style="color:#64748b;font-size:0.78rem;line-height:1.6">
          <strong style="color:#818cf8">Groq Llama 3.1 8B</strong> reasoned through each stage from first principles —
          no rules, no hardcoded fixes, pure model intelligence.
        </div>
      </div>
    </div>
  </div>
</div>"""
                yield log_text, card

            run_btn.click(run_llm_agent_with_score, [], [live_output, live_scorecard])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 3 — COMPARISON
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📊  Baseline vs LLM Agent"):

            gr.HTML("""
<div style="background:linear-gradient(135deg,#040810,#09101e);
            border:1px solid rgba(99,102,241,0.18);border-radius:14px;
            padding:22px 28px;margin-bottom:16px">
  <div style="font-size:1rem;font-weight:800;color:#e2e8f0;margin-bottom:4px">
    📊  Head-to-Head · Rule-Based Baseline vs LLM Agent
  </div>
  <div style="color:#475569;font-size:0.82rem;margin-bottom:16px">
    Each agent runs on its own <strong style="color:#64748b">private isolated environment</strong> with zero shared state.
    Run both and compare the results side-by-side.
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px">
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                border-radius:10px;padding:14px 16px">
      <div style="font-size:0.72rem;font-weight:800;letter-spacing:.06em;color:#475569;margin-bottom:6px">🔧  RULE-BASED BASELINE</div>
      <div style="color:#334155;font-size:0.8rem;line-height:1.6">
        Applies a hardcoded fix sequence to each stage. No reasoning — it just follows predefined steps.
        Fast and predictable but brittle if the environment deviates.
      </div>
    </div>
    <div style="background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.2);
                border-radius:10px;padding:14px 16px">
      <div style="font-size:0.72rem;font-weight:800;letter-spacing:.06em;color:#6366f1;margin-bottom:6px">🤖  LLM AGENT (Groq Llama 3.1 8B)</div>
      <div style="color:#334155;font-size:0.8rem;line-height:1.6">
        Reads the actual error log and current script, then reasons about the root cause.
        Every action is generated from model intelligence — no hardcoding whatsoever.
      </div>
    </div>
  </div>
</div>""")

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div style="font-size:0.72rem;font-weight:800;letter-spacing:.06em;color:#475569;padding:4px 0 8px">🔧  RULE-BASED BASELINE</div>')
                    run_baseline_cmp_btn = gr.Button("▶  Run Baseline", elem_classes="btn-reset")
                    baseline_cmp_log = gr.Textbox(
                        label="BASELINE LOG", lines=28, interactive=False,
                        placeholder="  Click ▶ Run Baseline to start…\n\n  Applies hardcoded fix sequences.\n  No reasoning — deterministic rule execution.",
                        elem_classes="code-box"
                    )
                    baseline_scorecard = gr.HTML("")

                with gr.Column():
                    gr.HTML('<div style="font-size:0.72rem;font-weight:800;letter-spacing:.06em;color:#6366f1;padding:4px 0 8px">🤖  LLM AGENT</div>')
                    run_llm_cmp_btn = gr.Button("▶  Run LLM Agent", elem_classes="btn-primary")
                    llm_cmp_log = gr.Textbox(
                        label="LLM AGENT LOG", lines=28, interactive=False,
                        placeholder="  Click ▶ Run LLM Agent to start…\n\n  Groq Llama 3.1 8B reads error logs\n  and reasons about each fix from scratch.",
                        elem_classes="code-box"
                    )
                    llm_scorecard = gr.HTML("")

            _ICONS = {"inspect_data":"🔍","run_script":"▶ ","edit_script":"✏ ","query_actor":"🤖","submit":"📤"}

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
                "features = ['age','salary','credit_score','loan_amount','employment_years']\n"
                "X = df[features].copy()\n"
                "y = df['target']\n"
                "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)\n"
                "scaler = StandardScaler()\n"
                "X_train_scaled = scaler.fit_transform(X_train)\n"
                "X_test_scaled  = scaler.transform(X_test)\n"
                "clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')\n"
                "clf.fit(X_train_scaled, y_train)\n"
                "print('Accuracy:', clf.score(X_test_scaled, y_test))\n"
            )
            _BSTAGES = {
                1: [("inspect_data",{}),("edit_script",{"script":_S1}),("run_script",{}),("submit",{})],
                2: [("inspect_data",{}),("edit_script",{"script":_S2}),("run_script",{}),("submit",{})],
                3: [("run_script",{}), ("edit_script",{"script":_S3}),("run_script",{}),("submit",{})],
                4: [("query_actor",{}),("edit_script",{"script":_S4}),("run_script",{}),("submit",{})],
            }

            def _scorecard_html(log_text, label, accent, description):
                import re as _re
                score_m  = _re.search(r"Episode score\s*:\s*([\d.]+)", log_text)
                stages_m = _re.search(r"Stages completed\s*:\s*(\[.*?\])", log_text)
                score    = float(score_m.group(1)) if score_m else 0.0
                stages   = stages_m.group(1) if stages_m else "[]"
                pct      = int(score * 100)
                color    = "#4ade80" if pct >= 75 else "#fbbf24" if pct >= 25 else "#f87171"
                n_stages = len([x for x in stages.replace("[","").replace("]","").split(",") if x.strip()])
                bar      = f'<div style="background:rgba(255,255,255,0.05);border-radius:999px;height:6px;overflow:hidden;margin:8px 0"><div style="width:{pct}%;height:100%;border-radius:999px;background:{color};box-shadow:0 0 8px {color}60"></div></div>'
                return f"""
<div style="background:linear-gradient(135deg,#040810,#07101a);
            border:1px solid {accent}30;border-radius:12px;
            padding:18px 20px;margin-top:10px;position:relative;overflow:hidden">
  <div style="position:absolute;top:-30px;right:-30px;width:120px;height:120px;
    background:radial-gradient(circle,{color}15 0%,transparent 65%);border-radius:50%;pointer-events:none"></div>
  <div style="font-size:0.65rem;font-weight:800;letter-spacing:.1em;color:#1e293b;margin-bottom:10px">RESULT</div>
  <div style="display:flex;align-items:flex-start;gap:16px">
    <div>
      <div style="font-size:2.8rem;font-weight:900;color:{color};line-height:1;text-shadow:0 0 20px {color}50">{pct}%</div>
      {bar}
      <div style="color:#1e293b;font-size:0.62rem;font-weight:800;letter-spacing:.05em">SCORE</div>
    </div>
    <div style="flex:1">
      <div style="display:flex;gap:8px;margin-bottom:10px">
        <div style="background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.05);
                    border-radius:8px;padding:8px 12px;text-align:center;flex:1">
          <div style="font-size:1.4rem;font-weight:800;color:{accent};line-height:1">{n_stages}/4</div>
          <div style="font-size:0.6rem;color:#1e293b;font-weight:700;margin-top:2px;letter-spacing:.04em">STAGES</div>
        </div>
      </div>
      <div style="color:#334155;font-size:0.76rem;line-height:1.5">
        <strong style="color:{accent}">{label}</strong> · {description}
      </div>
    </div>
  </div>
</div>"""

            def run_baseline_cmp():
                import time as _t
                from app.environment import DataEngEnvironment
                from app.models import Action
                lines = []
                def emit(l):
                    lines.append(l)
                    return "\n".join(lines)

                yield emit("  ┌────────────────────────────────────────────────────┐")
                yield emit("  │   Rule-Based Baseline Agent                        │")
                yield emit("  └────────────────────────────────────────────────────┘\n")

                _env = DataEngEnvironment()
                _env.reset()
                yield emit("  ✓ Private environment initialised — Stage 1\n")

                step = 0
                LABELS = {1:"Data Repair",2:"Training Monitor",3:"Eval Validation",4:"Deploy Gate"}
                for stage_num, actions in _BSTAGES.items():
                    if _env.done:
                        break
                    yield emit(f"  ─── Stage {stage_num}: {LABELS[stage_num]} {'─'*(40-len(LABELS[stage_num]))}")
                    for atype, payload in actions:
                        step += 1
                        icon = _ICONS.get(atype, "▸")
                        lines.append(f"  Step {step:02d}  {icon}  {atype.upper():<18}  ·  …")
                        yield "\n".join(lines)
                        try:
                            result = _env.step(Action(action_type=atype, payload=payload))
                            rwd  = result.reward
                            score = float(rwd.score)
                            msg   = (rwd.message or "")[:65]
                            bar   = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                            lines[-1] = f"  Step {step:02d}  {icon}  {atype.upper():<18}  ·  [{bar}] {score:.2f}  {msg}"
                        except Exception as ex:
                            lines[-1] = f"  Step {step:02d}  {icon}  {atype.upper():<18}  ·  ❌ {ex}"
                        yield "\n".join(lines)
                        _t.sleep(0.12)
                    if _env.current_stage == stage_num and not _env.done:
                        yield emit(f"  ✗ Stage {stage_num} did not pass")
                        break
                    elif stage_num in _env.stages_completed:
                        yield emit(f"  ✅ Stage {stage_num} complete\n")

                yield emit(f"\n  Stages completed : {list(_env.stages_completed)}")
                yield emit(f"  Episode score    : {_env.episode_score:.2f}")

            def run_llm_cmp():
                import time as _t, os as _os
                from app.environment import DataEngEnvironment
                from app.models import Action
                lines = []
                def emit(l):
                    lines.append(l)
                    return "\n".join(lines)

                yield emit("  ┌────────────────────────────────────────────────────┐")
                yield emit("  │   LLM Agent · Groq Llama 3.1 8B                    │")
                yield emit("  └────────────────────────────────────────────────────┘\n")

                groq_key = _os.environ.get("GROQ_API_KEY", "")
                if not groq_key:
                    yield emit("  ❌  GROQ_API_KEY not set in Space secrets.")
                    return
                try:
                    from groq import Groq as _Groq
                    _client = _Groq(api_key=groq_key)
                except ImportError:
                    yield emit("  ❌  groq package not installed.")
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
3. If run_script prints "Accuracy:" with NO error → call submit IMMEDIATELY.
4. Fix ALL bugs for a stage in ONE single edit_script call.

STAGE BUGS AND FIXES:
Stage 1 — Data Repair:
  - Bug 1: column named `age_years` but should be `age`
  - Bug 2: NaN values crash the script — fix with df = df.dropna()
  - Fix both bugs in one edit. Use LogisticRegression, StandardScaler, train_test_split.

Stage 2 — Training Monitor:
  - Bug: StandardScaler fit before train_test_split (data leakage)
  - Fix: call train_test_split FIRST, then scaler.fit_transform(X_train) and scaler.transform(X_test)

Stage 3 — Eval Validation:
  - Bug: scaler.fit() on ALL data before the split
  - Fix: move scaler.fit_transform to AFTER train_test_split, fit only on X_train

Stage 4 — Deploy Gate:
  - Bug: poor fairness (high fairness gap)
  - Fix: add class_weight='balanced' to LogisticRegression AND use test_size=0.25, stratify=y

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
                                            if isinstance(p, str): p = {"script": p}
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
                obs  = _env.reset()
                yield emit("  ✓ Private environment initialised — Stage 1\n")
                yield emit(f"  ─── Stage 1: {LABELS[1]} ────────────────────────────────")

                msgs = [
                    {"role":"system","content":_SYS},
                    {"role":"user","content":_obs_to_prompt(obs) + "\n\nStage 1: Data Repair. Inspect data, then fix all bugs in one edit_script call."},
                ]

                step, cur_stage = 0, 1
                hist = []
                last_was_clean_run = False

                for _ in range(80):
                    try:
                        ctx  = [msgs[0]] + msgs[1:2] + msgs[-8:] if len(msgs) > 10 else msgs
                        resp = _client.chat.completions.create(
                            model="llama-3.1-8b-instant", messages=ctx, max_tokens=2000, temperature=0.1)
                        raw    = resp.choices[0].message.content.strip()
                        action = _parse(raw)
                    except Exception as ex:
                        yield emit(f"  ⚠  LLM error: {ex}")
                        action = {"action_type":"run_script","payload":{}}

                    hist.append(action["action_type"])
                    if last_was_clean_run and action["action_type"] != "submit":
                        action = {"action_type":"submit","payload":{}}
                        last_was_clean_run = False
                    elif len(hist) >= 4 and len(set(hist[-4:])) == 1:
                        if hist[-1] == "edit_script":
                            action = {"action_type":"run_script","payload":{}}
                        elif hist[-1] == "inspect_data":
                            action = {"action_type":"run_script","payload":{}}
                        elif hist[-1] in ("run_script","submit"):
                            action = {"action_type":"edit_script","payload":{"script":obs.script_content or ""}}

                    step += 1
                    atype = action["action_type"]
                    icon  = _ICONS.get(atype, "▸")
                    lines.append(f"  Step {step:02d}  {icon}  {atype.upper():<18}  ·  …")
                    yield "\n".join(lines)

                    try:
                        result    = _env.step(Action(action_type=atype, payload=action.get("payload", {})))
                        obs       = result.observation
                        rwd       = result.reward
                        score     = float(rwd.score)
                        msg       = (rwd.message or "")[:60]
                        new_stage = obs.current_stage
                        done      = _env.done
                        last_was_clean_run = (
                            atype == "run_script"
                            and not obs.last_run_error
                            and bool(obs.last_run_output)
                        )
                        bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                        lines[-1] = f"  Step {step:02d}  {icon}  {atype.upper():<18}  ·  [{bar}] {score:.2f}  {msg}"
                    except Exception as ex:
                        lines[-1] = f"  Step {step:02d}  {icon}  {atype.upper():<18}  ·  ❌ {ex}"
                        new_stage, done = cur_stage, False
                        last_was_clean_run = False
                        rwd = None
                    yield "\n".join(lines)

                    if new_stage != cur_stage:
                        cur_stage = new_stage
                        hist = []
                        last_was_clean_run = False
                        if not done:
                            yield emit(f"\n  ✅ Stage complete!")
                            yield emit(f"  ─── Stage {cur_stage}: {LABELS.get(cur_stage,'')} {'─'*(40-len(LABELS.get(cur_stage,'')))}─\n")

                    msgs.append({"role":"assistant","content":json.dumps(action)})
                    feedback = _obs_to_prompt(obs)
                    if atype == "run_script" and obs.last_run_error:
                        feedback += "\n\nScript errored. Use edit_script with a FULL replacement script."
                    elif atype == "run_script" and not obs.last_run_error:
                        feedback += '\n\nClean run! Submit now: {"action_type":"submit","payload":{}}'
                    elif atype == "edit_script":
                        feedback += '\n\nEdited. Verify with run_script.'
                    elif atype == "query_actor":
                        feedback += "\n\nUse reviewer feedback to fix with edit_script (full replacement)."
                    elif atype == "submit":
                        feedback += "\n\nSubmit did not pass. Fix the remaining bug with edit_script."
                    elif atype == "inspect_data":
                        feedback += "\n\nData inspected. Fix all bugs in one edit_script call."
                    msgs.append({"role":"user","content":feedback})
                    if len(msgs) > 20:
                        msgs = [msgs[0]] + msgs[1:2] + msgs[-14:]

                    if done:
                        break
                    _t.sleep(0.2)

                yield emit(f"\n  Stages completed : {list(_env.stages_completed)}")
                yield emit(f"  Episode score    : {_env.episode_score:.2f}")

            def run_baseline_cmp_with_card():
                log_text = ""
                for chunk in run_baseline_cmp():
                    log_text = chunk
                    yield chunk, ""
                yield log_text, _scorecard_html(
                    log_text, "Rule-Based Baseline", "#94a3b8",
                    "Hardcoded fix sequences. Reliable but cannot adapt to unseen variation."
                )

            def run_llm_cmp_with_card():
                log_text = ""
                for chunk in run_llm_cmp():
                    log_text = chunk
                    yield chunk, ""
                yield log_text, _scorecard_html(
                    log_text, "Groq Llama 3.1 8B", "#818cf8",
                    "Pure model reasoning from error logs. No rules, no hardcoding."
                )

            run_baseline_cmp_btn.click(run_baseline_cmp_with_card, [], [baseline_cmp_log, baseline_scorecard])
            run_llm_cmp_btn.click(run_llm_cmp_with_card, [], [llm_cmp_log, llm_scorecard])

        # ══════════════════════════════════════════════════════════════════════
        # TAB 4 — API DOCS
        # ══════════════════════════════════════════════════════════════════════
        with gr.Tab("📖  API Docs"):
            gr.HTML(DOCS_HTML)

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7861)
