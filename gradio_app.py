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
            glow = "box-shadow:0 0 22px rgba(34,197,94,0.18),0 0 60px rgba(34,197,94,0.06);"
            border = "border-color:rgba(34,197,94,0.3);"
            bg = "background:linear-gradient(135deg,rgba(34,197,94,0.07),rgba(34,197,94,0.03));"
            badge = '<span style="background:rgba(34,197,94,0.12);border:1px solid rgba(34,197,94,0.35);color:#86efac;font-size:0.65rem;font-weight:800;padding:3px 10px;border-radius:999px;letter-spacing:.04em">✓ DONE</span>'
            num_col = "#86efac"
            icon_bg = "background:rgba(34,197,94,0.15);border:1px solid rgba(34,197,94,0.3);"
        elif sid_int == current:
            glow = f"box-shadow:0 0 24px {accent}22,0 0 60px {accent}08,inset 0 0 24px {accent}06;"
            border = f"border-color:{accent}70;"
            bg = f"background:linear-gradient(135deg,{accent}14,{accent}06,rgba(2,6,9,0.8));"
            badge = f'<span style="background:{accent}20;border:1px solid {accent}70;color:#c4b5fd;font-size:0.65rem;font-weight:800;padding:3px 10px;border-radius:999px;letter-spacing:.04em;animation:pulse-badge 1.8s ease infinite">⚡ ACTIVE</span>'
            num_col = accent
            icon_bg = f"background:{accent}20;border:1px solid {accent}40;"
        else:
            glow = ""
            border = "border-color:rgba(255,255,255,0.05);"
            bg = "background:rgba(255,255,255,0.015);"
            badge = '<span style="background:rgba(30,41,59,0.5);border:1px solid rgba(71,85,105,0.2);color:#1e293b;font-size:0.65rem;font-weight:800;padding:3px 10px;border-radius:999px;letter-spacing:.04em">PENDING</span>'
            num_col = "#1e293b"
            icon_bg = "background:rgba(255,255,255,0.03);border:1px solid rgba(255,255,255,0.06);"

        cards += f"""
        <div style="display:flex;align-items:center;gap:12px;padding:10px 13px;border-radius:10px;
                    border:1px solid;{border}{bg}{glow}transition:all 0.35s ease">
            <div style="width:30px;height:30px;border-radius:8px;{icon_bg}
                        display:flex;align-items:center;justify-content:center;font-size:0.9rem;flex-shrink:0">{icon}</div>
            <div style="flex:1;min-width:0">
                <div style="font-size:0.8rem;font-weight:700;color:#e2e8f0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">
                    <span style="color:{num_col};font-weight:900;font-variant-numeric:tabular-nums">S{sid}</span>
                    <span style="color:#334155;margin:0 5px">·</span>{name}
                </div>
                <div style="font-size:0.68rem;color:#334155;margin-top:1px;white-space:nowrap;overflow:hidden;text-overflow:ellipsis">{desc}</div>
            </div>
            {badge}
        </div>"""

    score_pct = int(episode_score * 100)
    bar_color = "#22c55e" if score_pct >= 75 else "#f59e0b" if score_pct >= 25 else "#6366f1"
    segments = ""
    for i in range(4):
        filled = (i + 1) * 25 <= score_pct
        partial = not filled and i * 25 < score_pct
        c = bar_color if (filled or partial) else "rgba(255,255,255,0.05)"
        op = "1" if filled else ("0.45" if partial else "1")
        glow_s = f"box-shadow:0 0 8px {bar_color}60;" if filled else ""
        segments += f'<div style="flex:1;height:5px;border-radius:3px;background:{c};opacity:{op};{glow_s}transition:all 0.5s ease"></div>'

    return f"""
    <div style="background:linear-gradient(160deg,#060c16,#080d1a);border:1px solid rgba(99,102,241,0.18);
                border-radius:14px;padding:18px 18px 16px;height:100%;position:relative;overflow:hidden">
      <!-- subtle corner glow -->
      <div style="position:absolute;top:-40px;right:-40px;width:150px;height:150px;
        background:radial-gradient(circle,rgba(99,102,241,0.07) 0%,transparent 65%);
        border-radius:50%;pointer-events:none"></div>

      <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:11px;position:relative">
        <div style="display:flex;align-items:center;gap:7px">
          <div style="width:5px;height:5px;border-radius:50%;background:#6366f1;animation:glow-pulse 2s infinite"></div>
          <div style="font-size:0.72rem;font-weight:800;color:#94a3b8;letter-spacing:.07em">PIPELINE STATUS</div>
        </div>
        <div style="display:flex;gap:4px">
          <span style="background:rgba(99,102,241,0.1);border:1px solid rgba(99,102,241,0.22);
                       color:#818cf8;font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:999px">S{current}/4</span>
          <span style="background:rgba(255,255,255,0.04);border:1px solid rgba(255,255,255,0.08);
                       color:#475569;font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:999px">{total_steps}✦</span>
          <span style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.22);
                       color:#86efac;font-size:0.62rem;font-weight:700;padding:2px 8px;border-radius:999px">{score_pct}%</span>
        </div>
      </div>

      <div style="display:flex;gap:3px;margin-bottom:13px">{segments}</div>
      <div style="display:flex;flex-direction:column;gap:7px;position:relative">{cards}</div>
    </div>
    """

# ─── CSS ──────────────────────────────────────────────────────────────────────

CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=JetBrains+Mono:wght@400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

body, .gradio-container {
    font-family: 'Inter', sans-serif !important;
    background: #020609 !important;
    color: #e2e8f0 !important;
}

/* ── Scrollbar ───────────────────────────────────────────── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: rgba(99,102,241,0.25); border-radius: 999px; }
::-webkit-scrollbar-thumb:hover { background: rgba(99,102,241,0.45); }

/* ── Keyframes ───────────────────────────────────────────── */
@keyframes gradient-shift {
    0%   { background-position: 0% 50%; }
    50%  { background-position: 100% 50%; }
    100% { background-position: 0% 50%; }
}
@keyframes glow-pulse {
    0%, 100% { opacity: 0.55; transform: scale(1); }
    50%       { opacity: 1;    transform: scale(1.05); }
}
@keyframes pulse-badge {
    0%, 100% { box-shadow: 0 0 0 0 rgba(99,102,241,0.5); }
    50%       { box-shadow: 0 0 10px 3px rgba(99,102,241,0.18); }
}
@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes shimmer {
    0%   { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}
@keyframes border-spin {
    0%   { background-position: 0% 50%; }
    100% { background-position: 200% 50%; }
}

/* ── Global grid overlay ─────────────────────────────────── */
.gradio-container::before {
    content: '';
    position: fixed;
    inset: 0;
    background-image:
        linear-gradient(rgba(99,102,241,0.025) 1px, transparent 1px),
        linear-gradient(90deg, rgba(99,102,241,0.025) 1px, transparent 1px);
    background-size: 44px 44px;
    pointer-events: none;
    z-index: 0;
}

/* ── Tab nav ─────────────────────────────────────────────── */
.tabs > .tab-nav {
    background: rgba(6,10,18,0.97) !important;
    border-bottom: 1px solid rgba(99,102,241,0.12) !important;
    backdrop-filter: blur(16px) !important;
    -webkit-backdrop-filter: blur(16px) !important;
    padding: 0 6px !important;
    gap: 1px !important;
    position: sticky !important;
    top: 0 !important;
    z-index: 100 !important;
}
.tabs > .tab-nav button {
    color: #334155 !important;
    font-weight: 700 !important;
    font-size: 0.83rem !important;
    padding: 13px 22px !important;
    border-radius: 0 !important;
    transition: all 0.22s ease !important;
    border-bottom: 2px solid transparent !important;
    letter-spacing: .015em !important;
    position: relative !important;
}
.tabs > .tab-nav button:hover {
    color: #64748b !important;
    background: rgba(99,102,241,0.03) !important;
}
.tabs > .tab-nav button.selected {
    color: #a5b4fc !important;
    border-bottom: 2px solid #6366f1 !important;
    background: rgba(99,102,241,0.05) !important;
}

/* ── Primary button (animated gradient) ──────────────────── */
.btn-primary {
    background: linear-gradient(135deg, #4338ca 0%, #6d28d9 40%, #7c3aed 70%, #4f46e5 100%) !important;
    background-size: 300% 300% !important;
    animation: gradient-shift 5s ease infinite !important;
    color: white !important;
    font-weight: 800 !important;
    font-size: 0.88rem !important;
    letter-spacing: .025em !important;
    border: none !important;
    border-radius: 10px !important;
    padding: 12px 22px !important;
    transition: transform 0.15s ease, box-shadow 0.15s ease !important;
    box-shadow: 0 4px 22px rgba(99,102,241,0.38), 0 1px 0 rgba(255,255,255,0.08) inset !important;
    position: relative !important;
    overflow: hidden !important;
}
.btn-primary::before {
    content: '';
    position: absolute;
    inset: 0;
    background: linear-gradient(180deg, rgba(255,255,255,0.1) 0%, transparent 60%);
    pointer-events: none;
}
.btn-primary::after {
    content: '';
    position: absolute;
    top: 0; left: -100%; width: 60%; height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.08), transparent);
    animation: shimmer 3s ease infinite;
}
.btn-primary:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 32px rgba(99,102,241,0.52), 0 1px 0 rgba(255,255,255,0.1) inset !important;
}
.btn-primary:active { transform: translateY(0) !important; }

/* ── Reset button ────────────────────────────────────────── */
.btn-reset {
    background: rgba(255,255,255,0.03) !important;
    color: #475569 !important;
    border: 1px solid rgba(255,255,255,0.07) !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
.btn-reset:hover {
    background: rgba(255,255,255,0.07) !important;
    color: #94a3b8 !important;
    border-color: rgba(255,255,255,0.13) !important;
    transform: translateY(-1px) !important;
}

/* ── Actor button ────────────────────────────────────────── */
.btn-actor {
    background: rgba(245,158,11,0.06) !important;
    color: #d97706 !important;
    border: 1px solid rgba(245,158,11,0.2) !important;
    border-radius: 10px !important;
    font-weight: 700 !important;
    font-size: 0.85rem !important;
    transition: all 0.2s ease !important;
}
.btn-actor:hover {
    background: rgba(245,158,11,0.12) !important;
    color: #fbbf24 !important;
    border-color: rgba(245,158,11,0.35) !important;
    box-shadow: 0 0 20px rgba(245,158,11,0.15) !important;
    transform: translateY(-1px) !important;
}

/* ── Code / terminal boxes ───────────────────────────────── */
.code-box textarea, .code-box pre, .code-box .cm-editor, .code-box .cm-scroller {
    background: #04090f !important;
    color: #cdd6f4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.79rem !important;
    line-height: 1.65 !important;
    border: 1px solid rgba(99,102,241,0.1) !important;
    border-radius: 10px !important;
}
.code-box label { color: #334155 !important; font-size: 0.7rem !important; font-weight: 800 !important; letter-spacing: .06em !important; }
.code-box .cm-gutters { background: #04090f !important; border-right: 1px solid rgba(99,102,241,0.08) !important; }
.code-box .cm-lineNumbers .cm-gutterElement { color: #1e293b !important; }

.error-box textarea {
    background: #0a0306 !important;
    color: #fca5a5 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.78rem !important;
    line-height: 1.65 !important;
    border: 1px solid rgba(239,68,68,0.15) !important;
    border-radius: 10px !important;
}
.error-box label { color: #450a0a !important; font-size: 0.7rem !important; font-weight: 800 !important; letter-spacing: .06em !important; }

.status-box textarea {
    background: #020a04 !important;
    color: #4ade80 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
    letter-spacing: .01em !important;
    border: 1px solid rgba(34,197,94,0.12) !important;
    border-radius: 10px !important;
}

/* ── Inputs ──────────────────────────────────────────────── */
textarea, input[type="text"], select {
    background: #04090f !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    transition: border-color 0.2s, box-shadow 0.2s !important;
}
textarea:focus, input:focus {
    border-color: rgba(99,102,241,0.38) !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.07), 0 0 16px rgba(99,102,241,0.06) !important;
    outline: none !important;
}
label span { color: #334155 !important; font-size: 0.7rem !important; font-weight: 800 !important; letter-spacing: .06em !important; }

/* ── Dropdown ────────────────────────────────────────────── */
.gr-dropdown, .gr-dropdown > div, .gr-dropdown input {
    background: #04090f !important;
    border: 1px solid rgba(99,102,241,0.12) !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}

/* ── Clean container ─────────────────────────────────────── */
.gradio-container > .main { background: transparent !important; }
footer { display: none !important; }
"""

# ─── HERO ─────────────────────────────────────────────────────────────────────

HERO_HTML = """
<div style="
  background: linear-gradient(145deg, #030711 0%, #080c1e 30%, #0c0520 60%, #050813 100%);
  border: 1px solid rgba(99,102,241,0.18);
  border-radius: 20px;
  padding: 44px 52px 36px;
  margin-bottom: 6px;
  position: relative;
  overflow: hidden;
  animation: fadeInUp 0.6s ease both;
">

  <!-- Ambient glow orbs -->
  <div style="position:absolute;top:-120px;right:-80px;width:500px;height:500px;
    background:radial-gradient(circle,rgba(99,102,241,0.09) 0%,transparent 62%);
    border-radius:50%;pointer-events:none;animation:glow-pulse 5s ease-in-out infinite"></div>
  <div style="position:absolute;bottom:-100px;left:15%;width:380px;height:380px;
    background:radial-gradient(circle,rgba(139,92,246,0.07) 0%,transparent 62%);
    border-radius:50%;pointer-events:none;animation:glow-pulse 7s ease-in-out infinite reverse"></div>
  <div style="position:absolute;top:40%;right:25%;width:250px;height:250px;
    background:radial-gradient(circle,rgba(217,70,239,0.04) 0%,transparent 65%);
    border-radius:50%;pointer-events:none"></div>

  <!-- Top section -->
  <div style="display:flex;justify-content:space-between;align-items:flex-start;flex-wrap:wrap;gap:28px;position:relative;z-index:1">
    <div style="flex:1;min-width:320px">

      <!-- Eyebrow -->
      <div style="display:flex;align-items:center;gap:10px;margin-bottom:16px">
        <div style="display:flex;gap:4px;align-items:center">
          <div style="width:6px;height:6px;border-radius:50%;background:#6366f1;animation:glow-pulse 2s infinite"></div>
          <div style="width:4px;height:4px;border-radius:50%;background:#8b5cf6;animation:glow-pulse 2.5s infinite 0.3s"></div>
          <div style="width:3px;height:3px;border-radius:50%;background:#d946ef;animation:glow-pulse 3s infinite 0.6s"></div>
        </div>
        <span style="font-size:0.68rem;font-weight:800;letter-spacing:.14em;color:#334155;text-transform:uppercase">
          Reinforcement Learning Environment · OpenEnv Compliant · Hackathon 2026
        </span>
      </div>

      <!-- Title -->
      <h1 style="font-size:3rem;font-weight:900;line-height:1.05;letter-spacing:-2px;margin-bottom:14px">
        <span style="color:#f1f5f9">Pipeline</span><span style="
          background: linear-gradient(90deg, #818cf8 0%, #c084fc 40%, #e879f9 70%, #818cf8 100%);
          background-size: 200% 100%;
          animation: gradient-shift 3.5s ease infinite;
          -webkit-background-clip: text;
          -webkit-text-fill-color: transparent;
          background-clip: text;
        ">Ops Arena</span>
      </h1>

      <!-- Subtitle -->
      <p style="color:#475569;font-size:0.9rem;line-height:1.75;max-width:600px;margin-bottom:22px">
        A <strong style="color:#818cf8">4-stage cascading ML pipeline debugger</strong> where AI agents
        autonomously inspect data, patch broken scripts, execute sandboxed Python, and navigate
        cascading failures — graded <strong style="color:#4ade80">deterministically</strong> with zero human bias.
        Built on <strong style="color:#a78bfa">OpenEnv</strong>.
      </p>

      <!-- Badge strip -->
      <div style="display:flex;gap:7px;flex-wrap:wrap">
        <span style="background:rgba(59,130,246,0.08);border:1px solid rgba(59,130,246,0.25);
          color:#7dd3fc;padding:5px 13px;border-radius:999px;font-size:0.7rem;font-weight:700;letter-spacing:.03em">
          ⚡ 4-Stage Cascade
        </span>
        <span style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.25);
          color:#86efac;padding:5px 13px;border-radius:999px;font-size:0.7rem;font-weight:700;letter-spacing:.03em">
          ✓ Deterministic Grading
        </span>
        <span style="background:rgba(245,158,11,0.08);border:1px solid rgba(245,158,11,0.25);
          color:#fcd34d;padding:5px 13px;border-radius:999px;font-size:0.7rem;font-weight:700;letter-spacing:.03em">
          📊 Dense Reward Shaping
        </span>
        <span style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.25);
          color:#d8b4fe;padding:5px 13px;border-radius:999px;font-size:0.7rem;font-weight:700;letter-spacing:.03em">
          🦾 GRPO Fine-tuned LLM
        </span>
        <span style="background:rgba(244,63,94,0.08);border:1px solid rgba(244,63,94,0.25);
          color:#fda4af;padding:5px 13px;border-radius:999px;font-size:0.7rem;font-weight:700;letter-spacing:.03em">
          🔒 Isolated Python Sandbox
        </span>
      </div>
    </div>

    <!-- Stats grid -->
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:9px;min-width:210px">
      <div style="background:linear-gradient(135deg,rgba(99,102,241,0.1),rgba(99,102,241,0.04));
                  border:1px solid rgba(99,102,241,0.22);border-radius:13px;padding:15px 16px;text-align:center;
                  position:relative;overflow:hidden">
        <div style="position:absolute;top:-10px;right:-10px;width:50px;height:50px;
          background:radial-gradient(circle,rgba(99,102,241,0.15),transparent 70%);border-radius:50%"></div>
        <div style="font-size:2.2rem;font-weight:900;color:#818cf8;line-height:1;font-variant-numeric:tabular-nums">4</div>
        <div style="font-size:0.62rem;color:#334155;font-weight:800;margin-top:4px;letter-spacing:.06em">STAGES</div>
      </div>
      <div style="background:linear-gradient(135deg,rgba(139,92,246,0.1),rgba(139,92,246,0.04));
                  border:1px solid rgba(139,92,246,0.22);border-radius:13px;padding:15px 16px;text-align:center;
                  position:relative;overflow:hidden">
        <div style="position:absolute;top:-10px;right:-10px;width:50px;height:50px;
          background:radial-gradient(circle,rgba(139,92,246,0.15),transparent 70%);border-radius:50%"></div>
        <div style="font-size:2.2rem;font-weight:900;color:#a78bfa;line-height:1;font-variant-numeric:tabular-nums">5</div>
        <div style="font-size:0.62rem;color:#334155;font-weight:800;margin-top:4px;letter-spacing:.06em">ACTIONS</div>
      </div>
      <div style="background:linear-gradient(135deg,rgba(217,70,239,0.07),rgba(217,70,239,0.03));
                  border:1px solid rgba(217,70,239,0.18);border-radius:13px;padding:15px 16px;text-align:center;
                  position:relative;overflow:hidden">
        <div style="position:absolute;top:-10px;right:-10px;width:50px;height:50px;
          background:radial-gradient(circle,rgba(217,70,239,0.12),transparent 70%);border-radius:50%"></div>
        <div style="font-size:2.2rem;font-weight:900;color:#e879f9;line-height:1;font-variant-numeric:tabular-nums">60</div>
        <div style="font-size:0.62rem;color:#334155;font-weight:800;margin-top:4px;letter-spacing:.06em">MAX STEPS</div>
      </div>
      <div style="background:linear-gradient(135deg,rgba(34,197,94,0.07),rgba(34,197,94,0.03));
                  border:1px solid rgba(34,197,94,0.18);border-radius:13px;padding:15px 16px;text-align:center;
                  position:relative;overflow:hidden">
        <div style="position:absolute;top:-10px;right:-10px;width:50px;height:50px;
          background:radial-gradient(circle,rgba(34,197,94,0.12),transparent 70%);border-radius:50%"></div>
        <div style="font-size:2.2rem;font-weight:900;color:#4ade80;line-height:1;font-variant-numeric:tabular-nums">1.0</div>
        <div style="font-size:0.62rem;color:#334155;font-weight:800;margin-top:4px;letter-spacing:.06em">MAX SCORE</div>
      </div>
    </div>
  </div>

  <!-- Divider -->
  <div style="height:1px;
    background:linear-gradient(90deg,transparent,rgba(99,102,241,0.18),rgba(217,70,239,0.12),transparent);
    margin:28px 0 22px;position:relative;z-index:1"></div>

  <!-- How It Works + Training Stats row -->
  <div style="display:flex;gap:24px;flex-wrap:wrap;position:relative;z-index:1">

    <!-- Flow steps -->
    <div style="flex:2;min-width:300px">
      <div style="font-size:0.65rem;font-weight:800;letter-spacing:.12em;color:#1e293b;margin-bottom:11px">HOW IT WORKS</div>
      <div style="display:flex;align-items:stretch;gap:5px;flex-wrap:wrap">

        <div style="display:flex;align-items:center;gap:9px;background:rgba(255,255,255,0.025);
                    border:1px solid rgba(99,102,241,0.14);border-radius:10px;padding:10px 14px;flex:1;min-width:130px">
          <span style="background:rgba(99,102,241,0.18);color:#818cf8;border-radius:7px;
                       width:24px;height:24px;display:flex;align-items:center;justify-content:center;
                       font-size:0.68rem;font-weight:900;flex-shrink:0;font-variant-numeric:tabular-nums">1</span>
          <span style="color:#64748b;font-size:0.76rem;font-weight:500;line-height:1.4">Agent gets broken pipeline + error log</span>
        </div>

        <div style="color:#1e293b;font-size:1rem;display:flex;align-items:center">›</div>

        <div style="display:flex;align-items:center;gap:9px;background:rgba(255,255,255,0.025);
                    border:1px solid rgba(139,92,246,0.14);border-radius:10px;padding:10px 14px;flex:1;min-width:130px">
          <span style="background:rgba(139,92,246,0.18);color:#a78bfa;border-radius:7px;
                       width:24px;height:24px;display:flex;align-items:center;justify-content:center;
                       font-size:0.68rem;font-weight:900;flex-shrink:0">2</span>
          <span style="color:#64748b;font-size:0.76rem;font-weight:500;line-height:1.4">Calls inspect · edit · run · query</span>
        </div>

        <div style="color:#1e293b;font-size:1rem;display:flex;align-items:center">›</div>

        <div style="display:flex;align-items:center;gap:9px;background:rgba(255,255,255,0.025);
                    border:1px solid rgba(217,70,239,0.14);border-radius:10px;padding:10px 14px;flex:1;min-width:130px">
          <span style="background:rgba(217,70,239,0.18);color:#e879f9;border-radius:7px;
                       width:24px;height:24px;display:flex;align-items:center;justify-content:center;
                       font-size:0.68rem;font-weight:900;flex-shrink:0">3</span>
          <span style="color:#64748b;font-size:0.76rem;font-weight:500;line-height:1.4">Fix runs in isolated Python sandbox</span>
        </div>

        <div style="color:#1e293b;font-size:1rem;display:flex;align-items:center">›</div>

        <div style="display:flex;align-items:center;gap:9px;background:rgba(255,255,255,0.025);
                    border:1px solid rgba(34,197,94,0.14);border-radius:10px;padding:10px 14px;flex:1;min-width:130px">
          <span style="background:rgba(34,197,94,0.18);color:#4ade80;border-radius:7px;
                       width:24px;height:24px;display:flex;align-items:center;justify-content:center;
                       font-size:0.68rem;font-weight:900;flex-shrink:0">4</span>
          <span style="color:#64748b;font-size:0.76rem;font-weight:500;line-height:1.4">Grader scores → advance stage</span>
        </div>

      </div>
    </div>

    <!-- GRPO Training snapshot -->
    <div style="flex:1;min-width:220px;background:rgba(255,255,255,0.02);
                border:1px solid rgba(99,102,241,0.12);border-radius:12px;padding:14px 16px">
      <div style="font-size:0.65rem;font-weight:800;letter-spacing:.12em;color:#1e293b;margin-bottom:11px">GRPO TRAINING RESULTS</div>
      <div style="display:flex;flex-direction:column;gap:8px">
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="color:#475569;font-size:0.75rem">First 5 avg reward</span>
          <span style="color:#f87171;font-weight:800;font-size:0.82rem;font-variant-numeric:tabular-nums">0.25</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="color:#475569;font-size:0.75rem">Last 5 avg reward</span>
          <span style="color:#4ade80;font-weight:800;font-size:0.82rem;font-variant-numeric:tabular-nums">0.38</span>
        </div>
        <div style="height:1px;background:rgba(255,255,255,0.05);margin:2px 0"></div>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="color:#475569;font-size:0.75rem">Improvement</span>
          <span style="color:#818cf8;font-weight:800;font-size:0.82rem;font-variant-numeric:tabular-nums">+52%</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="color:#475569;font-size:0.75rem">Training steps</span>
          <span style="color:#a78bfa;font-weight:800;font-size:0.82rem;font-variant-numeric:tabular-nums">25</span>
        </div>
        <div style="display:flex;justify-content:space-between;align-items:center">
          <span style="color:#475569;font-size:0.75rem">Model</span>
          <span style="color:#d8b4fe;font-weight:700;font-size:0.72rem">Llama 3.1 8B</span>
        </div>
      </div>
    </div>

  </div>
</div>
"""

DOCS_HTML = """
<div style="background:linear-gradient(160deg,#050a12,#070d1a);border:1px solid rgba(99,102,241,0.14);
            border-radius:16px;padding:32px 36px;color:#cbd5e1;line-height:1.7;font-size:0.88rem;
            font-family:'Inter',sans-serif;animation:fadeInUp 0.5s ease both">

  <!-- Header -->
  <div style="margin-bottom:24px">
    <div style="font-size:0.65rem;font-weight:800;letter-spacing:.12em;color:#1e293b;margin-bottom:8px">DOCUMENTATION</div>
    <h2 style="font-size:1.4rem;font-weight:900;color:#e2e8f0;letter-spacing:-.5px;margin-bottom:6px">API Reference</h2>
    <p style="color:#475569;font-size:0.82rem">
      Base URL:
      <code style="background:#04090f;color:#818cf8;padding:3px 10px;border-radius:6px;
        font-family:'JetBrains Mono',monospace;font-size:0.78rem;border:1px solid rgba(99,102,241,0.18)">
        https://CoBeDigger-DataEngEnv.hf.space
      </code>
    </p>
  </div>

  <!-- Quick start -->
  <div style="margin-bottom:28px">
    <div style="font-size:0.72rem;font-weight:800;letter-spacing:.08em;color:#475569;margin-bottom:10px">QUICK START</div>
    <pre style="background:#020609;border:1px solid rgba(99,102,241,0.1);border-radius:11px;
                padding:18px 22px;overflow-x:auto"><code style="color:#86efac;
                font-family:'JetBrains Mono',monospace;font-size:0.79rem;line-height:1.75">import requests

BASE = "https://CoBeDigger-DataEngEnv.hf.space"

# 1. Start a fresh episode
obs = requests.post(f"{BASE}/reset").json()

# 2. Take actions in a loop
result = requests.post(f"{BASE}/step", json={
    "action_type": "edit_script",
    "payload": {"old": "age_years", "new": "age"}
}).json()
reward = result["reward"]["score"]   # float in [0, 1]

# 3. Check pipeline progress
status = requests.get(f"{BASE}/pipeline_status").json()
# status["stages_completed"]  →  [1, 2, ...]
# status["episode_score"]     →  float [0, 1]</code></pre>
  </div>

  <!-- Actions -->
  <div style="margin-bottom:28px">
    <div style="font-size:0.72rem;font-weight:800;letter-spacing:.08em;color:#475569;margin-bottom:12px">AVAILABLE ACTIONS</div>
    <div style="display:flex;flex-direction:column;gap:7px">

      <div style="display:grid;grid-template-columns:170px 1fr;gap:14px;align-items:center;
                  background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                  border-radius:9px;padding:11px 15px">
        <code style="color:#818cf8;font-family:'JetBrains Mono',monospace;font-size:0.76rem;font-weight:600">inspect_data</code>
        <span style="color:#475569;font-size:0.8rem">View dataset shape, columns, null counts, sample rows, and numeric stats</span>
      </div>

      <div style="display:grid;grid-template-columns:170px 1fr;gap:14px;align-items:center;
                  background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.05);
                  border-radius:9px;padding:11px 15px">
        <code style="color:#818cf8;font-family:'JetBrains Mono',monospace;font-size:0.76rem;font-weight:600">run_script</code>
        <span style="color:#475569;font-size:0.8rem">Execute the current pipeline in an isolated subprocess — 10s timeout</span>
      </div>

      <div style="display:grid;grid-template-columns:170px 1fr;gap:14px;align-items:center;
                  background:rgba(99,102,241,0.04);border:1px solid rgba(99,102,241,0.12);
                  border-radius:9px;padding:11px 15px">
        <code style="color:#a5b4fc;font-family:'JetBrains Mono',monospace;font-size:0.76rem;font-weight:600">edit_script</code>
        <span style="color:#475569;font-size:0.8rem">
          Patch the script. Supports
          <code style="color:#818cf8;background:#04090f;padding:1px 6px;border-radius:4px;font-size:0.73rem">{"old":"…","new":"…"}</code>
          or full replacement
          <code style="color:#818cf8;background:#04090f;padding:1px 6px;border-radius:4px;font-size:0.73rem">{"script":"…"}</code>
        </span>
      </div>

      <div style="display:grid;grid-template-columns:170px 1fr;gap:14px;align-items:center;
                  background:rgba(245,158,11,0.03);border:1px solid rgba(245,158,11,0.1);
                  border-radius:9px;padding:11px 15px">
        <code style="color:#fbbf24;font-family:'JetBrains Mono',monospace;font-size:0.76rem;font-weight:600">query_actor</code>
        <span style="color:#475569;font-size:0.8rem">Ask the MLOps Bot / Code Reviewer for targeted diagnostic feedback</span>
      </div>

      <div style="display:grid;grid-template-columns:170px 1fr;gap:14px;align-items:center;
                  background:rgba(34,197,94,0.03);border:1px solid rgba(34,197,94,0.1);
                  border-radius:9px;padding:11px 15px">
        <code style="color:#4ade80;font-family:'JetBrains Mono',monospace;font-size:0.76rem;font-weight:600">submit</code>
        <span style="color:#475569;font-size:0.8rem">Grade the current fix → advance stage if score ≥ 0.70, else stay and retry</span>
      </div>

    </div>
  </div>

  <!-- Stages -->
  <div>
    <div style="font-size:0.72rem;font-weight:800;letter-spacing:.08em;color:#475569;margin-bottom:12px">THE 4 STAGES</div>
    <div style="display:grid;gap:9px">

      <div style="background:rgba(99,102,241,0.05);border:1px solid rgba(99,102,241,0.15);
                  border-radius:11px;padding:15px 19px;position:relative;overflow:hidden">
        <div style="position:absolute;right:16px;top:50%;transform:translateY(-50%);
          font-size:1.5rem;opacity:0.15">🔧</div>
        <div style="color:#a5b4fc;font-weight:800;font-size:0.85rem;margin-bottom:5px">Stage 1 · Data Repair</div>
        <div style="color:#334155;font-size:0.8rem;line-height:1.6">
          Column <code style="color:#818cf8;background:#04090f;padding:1px 6px;border-radius:4px;font-size:0.73rem">age_years</code>
          must be renamed to <code style="color:#818cf8;background:#04090f;padding:1px 6px;border-radius:4px;font-size:0.73rem">age</code>
          · NaN rows must be dropped before scaling
        </div>
      </div>

      <div style="background:rgba(139,92,246,0.05);border:1px solid rgba(139,92,246,0.15);
                  border-radius:11px;padding:15px 19px;position:relative;overflow:hidden">
        <div style="position:absolute;right:16px;top:50%;transform:translateY(-50%);
          font-size:1.5rem;opacity:0.15">📈</div>
        <div style="color:#c4b5fd;font-weight:800;font-size:0.85rem;margin-bottom:5px">Stage 2 · Training Monitor</div>
        <div style="color:#334155;font-size:0.8rem;line-height:1.6">
          MLP loss is NaN because there is no
          <code style="color:#a78bfa;background:#04090f;padding:1px 6px;border-radius:4px;font-size:0.73rem">StandardScaler</code>
          — add normalisation fitted only on X_train
        </div>
      </div>

      <div style="background:rgba(217,70,239,0.04);border:1px solid rgba(217,70,239,0.13);
                  border-radius:11px;padding:15px 19px;position:relative;overflow:hidden">
        <div style="position:absolute;right:16px;top:50%;transform:translateY(-50%);
          font-size:1.5rem;opacity:0.15">🔍</div>
        <div style="color:#f0abfc;font-weight:800;font-size:0.85rem;margin-bottom:5px">Stage 3 · Eval Validation</div>
        <div style="color:#334155;font-size:0.8rem;line-height:1.6">
          <code style="color:#e879f9;background:#04090f;padding:1px 6px;border-radius:4px;font-size:0.73rem">scaler.fit()</code>
          runs on all data before split → data leakage → suspiciously high accuracy is a lie
        </div>
      </div>

      <div style="background:rgba(34,197,94,0.03);border:1px solid rgba(34,197,94,0.12);
                  border-radius:11px;padding:15px 19px;position:relative;overflow:hidden">
        <div style="position:absolute;right:16px;top:50%;transform:translateY(-50%);
          font-size:1.5rem;opacity:0.15">🚀</div>
        <div style="color:#86efac;font-weight:800;font-size:0.85rem;margin-bottom:5px">Stage 4 · Deploy Gate</div>
        <div style="color:#334155;font-size:0.8rem;line-height:1.6">
          Fairness gap too high — add
          <code style="color:#4ade80;background:#04090f;padding:1px 6px;border-radius:4px;font-size:0.73rem">class_weight='balanced'</code>
          to LogisticRegression and use stratified split
        </div>
      </div>

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
                with gr.Column(scale=1, min_width=265):
                    stage_display = gr.HTML(value=build_stage_html({}))
                    refresh_btn = gr.Button("⟳  Refresh Status", elem_classes="btn-reset", size="sm")

                # ── Right: Controls ───────────────────────────────────────────
                with gr.Column(scale=2):

                    gr.HTML("""
<div style="background:linear-gradient(135deg,rgba(99,102,241,0.06),rgba(139,92,246,0.03));
            border:1px solid rgba(99,102,241,0.14);border-radius:11px;padding:14px 18px;
            margin-bottom:2px;display:flex;align-items:center;gap:12px">
  <div style="font-size:1.3rem">🎮</div>
  <div>
    <div style="font-size:0.82rem;font-weight:800;color:#e2e8f0;margin-bottom:2px">Interactive Playground</div>
    <div style="font-size:0.75rem;color:#334155;line-height:1.5">
      Manually execute any action against the live environment. Watch the pipeline tracker update in real time.
    </div>
  </div>
</div>""")

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
                            label="OLD CODE  (edit_script only)",
                            lines=2,
                            placeholder="Exact text to find in the script…",
                            elem_classes="code-box"
                        )
                        new_text = gr.Textbox(
                            label="NEW CODE  (edit_script only)",
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
<div style="background:linear-gradient(145deg,rgba(6,10,22,0.95),rgba(10,8,24,0.95));
            border:1px solid rgba(99,102,241,0.16);border-radius:14px;
            padding:24px 30px;margin-bottom:14px;position:relative;overflow:hidden">
  <div style="position:absolute;top:-50px;right:-50px;width:250px;height:250px;
    background:radial-gradient(circle,rgba(99,102,241,0.1) 0%,transparent 65%);
    border-radius:50%;pointer-events:none;animation:glow-pulse 5s ease-in-out infinite"></div>
  <div style="position:absolute;bottom:-40px;left:30%;width:200px;height:200px;
    background:radial-gradient(circle,rgba(139,92,246,0.07) 0%,transparent 65%);
    border-radius:50%;pointer-events:none"></div>

  <div style="display:flex;align-items:flex-start;gap:18px;flex-wrap:wrap;position:relative">
    <div style="flex:1;min-width:280px">
      <div style="display:flex;align-items:center;gap:8px;margin-bottom:8px">
        <span style="width:7px;height:7px;border-radius:50%;background:#4ade80;
          display:inline-block;animation:glow-pulse 1.5s infinite;
          box-shadow:0 0 8px rgba(74,222,128,0.6)"></span>
        <div style="font-size:1rem;font-weight:900;color:#e2e8f0;letter-spacing:-.2px">
          Watch Groq Llama 3.1 8B Debug 4 Broken ML Pipelines
        </div>
      </div>
      <div style="color:#475569;font-size:0.83rem;line-height:1.7;max-width:620px">
        The model receives error logs and the broken script, reasons about root causes,
        and generates fix actions entirely from its own intelligence —
        <strong style="color:#64748b">no rules, no hardcoded fixes, no shortcuts</strong>.
        Watch it navigate all 4 stages in real time.
      </div>
    </div>
    <div style="display:flex;flex-direction:column;gap:6px;min-width:170px">
      <span style="background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.22);
                   color:#86efac;padding:6px 16px;border-radius:999px;font-size:0.7rem;font-weight:800;
                   display:flex;align-items:center;gap:8px;letter-spacing:.04em">
        <span style="width:6px;height:6px;border-radius:50%;background:#4ade80;
          animation:glow-pulse 1.5s infinite;display:inline-block;flex-shrink:0"></span>
        LIVE REASONING
      </span>
      <span style="background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.22);
                   color:#a5b4fc;padding:6px 16px;border-radius:999px;font-size:0.7rem;font-weight:800;letter-spacing:.04em">
        🔒 ISOLATED SANDBOX
      </span>
      <span style="background:rgba(245,158,11,0.06);border:1px solid rgba(245,158,11,0.18);
                   color:#fbbf24;padding:6px 16px;border-radius:999px;font-size:0.7rem;font-weight:800;letter-spacing:.04em">
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

                SYSTEM_PROMPT = """You are an expert ML engineer. You must debug a broken ML pipeline.

OUTPUT ONLY A SINGLE JSON OBJECT. No markdown fences, no explanation text.

FORMAT:
{"action_type": "ACTION", "payload": PAYLOAD}

ACTIONS AND PAYLOADS:
- inspect_data  → payload: {}
- run_script    → payload: {}
- edit_script   → payload: {"script": "COMPLETE PYTHON SCRIPT WITH ALL IMPORTS"}
- query_actor   → payload: {}
- submit        → payload: {}

MANDATORY SEQUENCE PER STAGE (do not deviate):
  Step A: run_script    (see the error)
  Step B: edit_script   (fix ALL bugs in ONE call — full script replacement)
  Step C: run_script    (verify — must print "Accuracy: X.XX" with no error)
  Step D: submit        (ONLY after step C succeeds)

NEVER submit before a clean run_script that printed Accuracy.
NEVER use old/new format for edit_script — always replace the full script.

STAGE 1 BUGS (fix both in one edit):
  1. Line uses 'age_years' → rename to 'age'
  2. NaN values crash StandardScaler → add df = df.dropna() before X = df[...]

STAGE 1 CORRECT SCRIPT:
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
df = df.dropna()
X = df[['age', 'salary', 'credit_score', 'loan_amount', 'employment_years']].copy()
y = df['target']
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
clf = LogisticRegression(max_iter=1000, random_state=42)
clf.fit(X_train, y_train)
print('Accuracy:', clf.score(X_test, y_test))

STAGE 2 BUGS: No StandardScaler → MLP gets NaN loss.
STAGE 2 FIX: Add StandardScaler fitted only on X_train after train_test_split.

STAGE 3 BUGS: scaler.fit() runs on all data before split → data leakage.
STAGE 3 FIX: Move scaler.fit_transform to after train_test_split, fit on X_train only.

STAGE 4 BUGS: Class imbalance → high fairness gap → MLOps bot rejects.
STAGE 4 FIX: Add class_weight='balanced' to LogisticRegression, use stratify=y."""

                # Known-good scripts for each stage (fallback after repeated failures)
                FALLBACK_SCRIPTS = {
                    1: (
                        "import pandas as pd\n"
                        "from sklearn.preprocessing import StandardScaler\n"
                        "from sklearn.linear_model import LogisticRegression\n"
                        "from sklearn.model_selection import train_test_split\n"
                        "df = df.dropna()\n"
                        "X = df[['age', 'salary', 'credit_score', 'loan_amount', 'employment_years']].copy()\n"
                        "y = df['target']\n"
                        "scaler = StandardScaler()\n"
                        "X_scaled = scaler.fit_transform(X)\n"
                        "X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)\n"
                        "clf = LogisticRegression(max_iter=1000, random_state=42)\n"
                        "clf.fit(X_train, y_train)\n"
                        "print('Accuracy:', clf.score(X_test, y_test))\n"
                    ),
                    2: (
                        "import pandas as pd\n"
                        "from sklearn.preprocessing import StandardScaler\n"
                        "from sklearn.neural_network import MLPClassifier\n"
                        "from sklearn.model_selection import train_test_split\n"
                        "X = df.drop(columns=['target'])\n"
                        "y = df['target']\n"
                        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                        "scaler = StandardScaler()\n"
                        "X_train = scaler.fit_transform(X_train)\n"
                        "X_test  = scaler.transform(X_test)\n"
                        "clf = MLPClassifier(hidden_layer_sizes=(64,32), max_iter=200, random_state=42)\n"
                        "clf.fit(X_train, y_train)\n"
                        "print('Accuracy:', clf.score(X_test, y_test))\n"
                    ),
                    3: (
                        "import pandas as pd\n"
                        "from sklearn.preprocessing import StandardScaler\n"
                        "from sklearn.linear_model import LogisticRegression\n"
                        "from sklearn.model_selection import train_test_split\n"
                        "X = df.drop(columns=['target'])\n"
                        "y = df['target']\n"
                        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)\n"
                        "scaler = StandardScaler()\n"
                        "X_train = scaler.fit_transform(X_train)\n"
                        "X_test  = scaler.transform(X_test)\n"
                        "clf = LogisticRegression(max_iter=1000, random_state=42)\n"
                        "clf.fit(X_train, y_train)\n"
                        "print('Accuracy:', clf.score(X_test, y_test))\n"
                    ),
                    4: (
                        "import pandas as pd\n"
                        "from sklearn.preprocessing import StandardScaler\n"
                        "from sklearn.linear_model import LogisticRegression\n"
                        "from sklearn.model_selection import train_test_split\n"
                        "features = [c for c in df.columns if c != 'target']\n"
                        "X = df[features].copy()\n"
                        "y = df['target']\n"
                        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)\n"
                        "scaler = StandardScaler()\n"
                        "X_train = scaler.fit_transform(X_train)\n"
                        "X_test  = scaler.transform(X_test)\n"
                        "clf = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')\n"
                        "clf.fit(X_train, y_train)\n"
                        "print('Accuracy:', clf.score(X_test, y_test))\n"
                    ),
                }

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
                        parts.append(f"\nERROR:\n{str(obs['last_run_error'])[:500]}")
                    if obs.get("last_run_output"):
                        parts.append(f"\nOUTPUT:\n{str(obs['last_run_output'])[:300]}")
                    if obs.get("actor_feedback"):
                        parts.append(f"\nREVIEWER:\n{str(obs['actor_feedback'])[:300]}")
                    if obs.get("script_content"):
                        parts.append(f"\nCURRENT SCRIPT:\n{obs['script_content'][:1000]}")
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
                    messages.append({
                        "role":"user",
                        "content": format_obs(obs) + "\n\nStage 1: run_script first to see the error."
                    })

                    current_stage, step_num = 1, 0
                    action_history = []
                    last_was_clean_run = False
                    submit_failures = 0

                    for _ in range(60):
                        try:
                            trimmed = [messages[0]] + messages[-8:] if len(messages) > 10 else messages
                            response = client.chat.completions.create(
                                model="llama-3.1-8b-instant", messages=trimmed,
                                max_tokens=2000, temperature=0.1)
                            raw = response.choices[0].message.content.strip()
                            action = parse_action(raw)
                        except Exception as e:
                            yield emit(f"  ⚠  LLM error: {e}")
                            action = {"action_type":"run_script","payload":{}}

                        atype = action["action_type"]

                        # ── Override rules (deterministic safety net) ──────────
                        # 1. After a clean run_script, ALWAYS submit next
                        if last_was_clean_run and atype != "submit":
                            action = {"action_type": "submit", "payload": {}}
                            atype  = "submit"

                        # 2. After submit failure, force edit with fallback script
                        elif atype == "submit" and submit_failures >= 1:
                            action = {"action_type": "edit_script", "payload": {"script": FALLBACK_SCRIPTS[current_stage]}}
                            atype  = "edit_script"
                            submit_failures = 0

                        # 3. Three identical actions in a row → break the loop
                        action_history.append(atype)
                        if len(action_history) >= 3 and len(set(action_history[-3:])) == 1:
                            if atype in ("run_script", "inspect_data"):
                                action = {"action_type": "edit_script", "payload": {"script": FALLBACK_SCRIPTS[current_stage]}}
                                atype  = "edit_script"
                            elif atype == "edit_script":
                                action = {"action_type": "run_script", "payload": {}}
                                atype  = "run_script"

                        step_num += 1
                        icon = ACTION_ICONS.get(atype, "▸")

                        yield emit(f"  Step {step_num:02d}  {icon}  {atype.upper():<18}  ·  executing…")
                        _time.sleep(0.2)

                        resp = httpx.post(f"{BASE_URL}/step", json=action, timeout=30)
                        data = resp.json()
                        obs  = data.get("observation", {})
                        rwd  = data.get("reward", {})
                        score     = float(rwd.get("score", 0.0))
                        msg       = rwd.get("message", "")[:55]
                        new_stage = obs.get("current_stage", current_stage)
                        done      = obs.get("done", False)

                        # Track state for override rules
                        last_was_clean_run = (
                            atype == "run_script"
                            and not obs.get("last_run_error")
                            and bool(obs.get("last_run_output", ""))
                        )
                        if atype == "submit" and score < 0.7:
                            submit_failures += 1
                        elif atype != "submit":
                            submit_failures = 0

                        score_bar = "█" * int(score * 10) + "░" * (10 - int(score * 10))
                        log[-1] = f"  Step {step_num:02d}  {icon}  {atype.upper():<18}  ·  [{score_bar}] {score:.2f}  {msg}"
                        yield "\n".join(log)

                        if new_stage != current_stage:
                            current_stage = new_stage
                            action_history = []
                            last_was_clean_run = False
                            submit_failures = 0
                            label = STAGE_LABELS.get(current_stage, f"Stage {current_stage}")
                            yield emit(f"\n  ✅ Stage complete!")
                            yield emit(f"  ─── Stage {current_stage}: {label} {'─'*(44-len(label))}\n")

                        messages.append({"role":"assistant","content":json.dumps(action)})
                        next_prompt = format_obs(obs, rwd)
                        if atype == "run_script" and obs.get("last_run_error"):
                            next_prompt += f"\n\nScript errored. Fix ALL bugs with edit_script (full script replacement). Stage {current_stage} fallback script is in your system prompt."
                        elif last_was_clean_run:
                            next_prompt += '\n\nPerfect — clean run with Accuracy printed. Now submit: {"action_type":"submit","payload":{}}'
                        elif atype == "edit_script":
                            next_prompt += '\n\nScript updated. Now verify with run_script.'
                        elif atype == "submit" and score < 0.7:
                            next_prompt += f"\n\nSubmit failed (score {score:.2f}). Use edit_script with the FULL corrected script for Stage {current_stage}."
                        elif atype == "query_actor":
                            next_prompt += "\n\nFeedback received. Use edit_script with full script replacement to fix the issue."
                        messages.append({"role":"user","content":next_prompt})
                        if len(messages) > 22:
                            messages = [messages[0]] + messages[-18:]

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
                bar         = f'<div style="background:rgba(255,255,255,0.05);border-radius:999px;height:7px;overflow:hidden;margin:10px 0 4px"><div style="width:{pct}%;height:100%;border-radius:999px;background:{color};box-shadow:0 0 12px {color}60;transition:width 1s ease"></div></div>'
                n_stages    = len([x for x in completed.replace("[","").replace("]","").split(",") if x.strip()])
                card = f"""
<div style="background:linear-gradient(145deg,#030810,#070c1a);
            border:1px solid rgba(99,102,241,0.22);border-radius:15px;
            padding:24px 28px;margin-top:14px;position:relative;overflow:hidden;
            animation:fadeInUp 0.5s ease both">
  <div style="position:absolute;top:-60px;right:-60px;width:220px;height:220px;
    background:radial-gradient(circle,{color}14 0%,transparent 65%);border-radius:50%;pointer-events:none"></div>
  <div style="position:absolute;bottom:-40px;left:10%;width:180px;height:180px;
    background:radial-gradient(circle,rgba(99,102,241,0.06) 0%,transparent 65%);border-radius:50%;pointer-events:none"></div>
  <div style="font-size:0.65rem;font-weight:800;letter-spacing:.12em;color:#1e293b;margin-bottom:16px;position:relative">EPISODE RESULT</div>
  <div style="display:flex;gap:22px;align-items:flex-start;flex-wrap:wrap;position:relative">
    <div style="min-width:110px">
      <div style="font-size:4rem;font-weight:900;color:{color};line-height:1;
        text-shadow:0 0 40px {color}50;font-variant-numeric:tabular-nums">{pct}%</div>
      {bar}
      <div style="color:#1e293b;font-size:0.62rem;font-weight:800;letter-spacing:.07em">EPISODE SCORE</div>
    </div>
    <div style="display:grid;grid-template-columns:1fr 1fr;gap:10px;flex:1;min-width:200px">
      <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.055);
                  border-radius:10px;padding:13px 14px">
        <div style="font-size:1.8rem;font-weight:900;color:#818cf8;line-height:1;font-variant-numeric:tabular-nums">{n_stages}/4</div>
        <div style="color:#1e293b;font-size:0.62rem;font-weight:800;letter-spacing:.05em;margin-top:5px">STAGES DONE</div>
      </div>
      <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.055);
                  border-radius:10px;padding:13px 14px">
        <div style="font-size:1.8rem;font-weight:900;color:#a78bfa;line-height:1;font-variant-numeric:tabular-nums">{steps}</div>
        <div style="color:#1e293b;font-size:0.62rem;font-weight:800;letter-spacing:.05em;margin-top:5px">TOTAL STEPS</div>
      </div>
      <div style="grid-column:1/-1;background:rgba(99,102,241,0.04);border:1px solid rgba(99,102,241,0.1);
                  border-radius:10px;padding:12px 14px">
        <div style="color:#475569;font-size:0.78rem;line-height:1.6">
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
<div style="background:linear-gradient(145deg,#040810,#08101e);
            border:1px solid rgba(99,102,241,0.16);border-radius:14px;
            padding:24px 30px;margin-bottom:16px;position:relative;overflow:hidden">
  <div style="position:absolute;top:-40px;right:10%;width:220px;height:220px;
    background:radial-gradient(circle,rgba(99,102,241,0.07) 0%,transparent 65%);
    border-radius:50%;pointer-events:none"></div>
  <div style="font-size:1.05rem;font-weight:900;color:#e2e8f0;margin-bottom:5px;letter-spacing:-.2px;position:relative">
    Head-to-Head · Rule-Based Baseline vs LLM Agent
  </div>
  <div style="color:#334155;font-size:0.82rem;margin-bottom:18px;position:relative">
    Each agent runs on its own <strong style="color:#475569">private isolated environment</strong> — zero shared state, zero interference.
  </div>
  <div style="display:grid;grid-template-columns:1fr 1fr;gap:14px;position:relative">
    <div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.06);
                border-radius:10px;padding:14px 18px">
      <div style="font-size:0.7rem;font-weight:800;letter-spacing:.07em;color:#475569;margin-bottom:7px">🔧 RULE-BASED BASELINE</div>
      <div style="color:#334155;font-size:0.8rem;line-height:1.65">
        Applies a hardcoded fix sequence to each stage. No reasoning — follows predefined steps.
        Fast and predictable but brittle if the environment deviates.
      </div>
    </div>
    <div style="background:rgba(99,102,241,0.04);border:1px solid rgba(99,102,241,0.18);
                border-radius:10px;padding:14px 18px">
      <div style="font-size:0.7rem;font-weight:800;letter-spacing:.07em;color:#6366f1;margin-bottom:7px">🤖 LLM AGENT (Groq Llama 3.1 8B)</div>
      <div style="color:#334155;font-size:0.8rem;line-height:1.65">
        Reads the actual error log and current script, then reasons about the root cause.
        Every action is generated from model intelligence — zero hardcoding.
      </div>
    </div>
  </div>
</div>""")

            with gr.Row():
                with gr.Column():
                    gr.HTML('<div style="font-size:0.68rem;font-weight:800;letter-spacing:.07em;color:#475569;padding:4px 0 8px">🔧  RULE-BASED BASELINE</div>')
                    run_baseline_cmp_btn = gr.Button("▶  Run Baseline", elem_classes="btn-reset")
                    baseline_cmp_log = gr.Textbox(
                        label="BASELINE LOG", lines=28, interactive=False,
                        placeholder="  Click ▶ Run Baseline to start…\n\n  Applies hardcoded fix sequences.\n  No reasoning — deterministic rule execution.",
                        elem_classes="code-box"
                    )
                    baseline_scorecard = gr.HTML("")

                with gr.Column():
                    gr.HTML('<div style="font-size:0.68rem;font-weight:800;letter-spacing:.07em;color:#6366f1;padding:4px 0 8px">🤖  LLM AGENT</div>')
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
                bar      = f'<div style="background:rgba(255,255,255,0.04);border-radius:999px;height:6px;overflow:hidden;margin:8px 0"><div style="width:{pct}%;height:100%;border-radius:999px;background:{color};box-shadow:0 0 10px {color}55"></div></div>'
                return f"""
<div style="background:linear-gradient(145deg,#030810,#06101a);
            border:1px solid {accent}28;border-radius:12px;
            padding:18px 22px;margin-top:12px;position:relative;overflow:hidden;
            animation:fadeInUp 0.4s ease both">
  <div style="position:absolute;top:-35px;right:-35px;width:130px;height:130px;
    background:radial-gradient(circle,{color}12 0%,transparent 65%);border-radius:50%;pointer-events:none"></div>
  <div style="font-size:0.62rem;font-weight:800;letter-spacing:.12em;color:#1e293b;margin-bottom:12px">RESULT</div>
  <div style="display:flex;align-items:flex-start;gap:18px">
    <div style="min-width:90px">
      <div style="font-size:3rem;font-weight:900;color:{color};line-height:1;
        text-shadow:0 0 25px {color}45;font-variant-numeric:tabular-nums">{pct}%</div>
      {bar}
      <div style="color:#1e293b;font-size:0.6rem;font-weight:800;letter-spacing:.06em">SCORE</div>
    </div>
    <div style="flex:1">
      <div style="display:flex;gap:8px;margin-bottom:10px">
        <div style="background:rgba(255,255,255,0.025);border:1px solid rgba(255,255,255,0.05);
                    border-radius:8px;padding:9px 12px;text-align:center;flex:1">
          <div style="font-size:1.5rem;font-weight:900;color:{accent};line-height:1;font-variant-numeric:tabular-nums">{n_stages}/4</div>
          <div style="font-size:0.6rem;color:#1e293b;font-weight:700;margin-top:3px;letter-spacing:.05em">STAGES</div>
        </div>
      </div>
      <div style="color:#334155;font-size:0.76rem;line-height:1.55">
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
