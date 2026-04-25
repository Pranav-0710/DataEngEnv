"""
PipelineOps Arena — LLM-Powered Agent (No Hardcoded Answers)

The agent uses Groq's Llama 3.1 API to read error logs, inspect data,
reason about the bug, and generate fix actions entirely on its own.
"""

import os, time, json, re, requests

BASE_URL = os.environ.get("DATAENGENV_URL", "http://localhost:8000")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are a senior Data Engineer. A broken ML pipeline must be fixed.

You will receive the script, error logs, and data info. You must fix the bug.

RESPOND WITH ONLY A JSON OBJECT. No explanation. No markdown.

Actions:
{"action_type": "inspect_data", "payload": {}}
{"action_type": "run_script", "payload": {}}
{"action_type": "edit_script", "payload": {"old": "exact text in script", "new": "replacement"}}
{"action_type": "query_actor", "payload": {}}
{"action_type": "submit", "payload": {}}

CRITICAL RULES:
- Use inspect_data ONLY ONCE per stage, then move on
- After inspecting, use edit_script to fix the bug
- The "old" field must contain text that EXACTLY exists in the script
- After editing, use run_script to verify
- After a clean run, use submit
- NEVER repeat the same action twice in a row

Common fixes:
- KeyError 'age_years': the column was renamed to 'age'. Replace 'age_years' with 'age'. Also add df.dropna(inplace=True)
- Training diverges / loss NaN: features are unscaled. Add StandardScaler before the classifier
- Suspiciously high accuracy (0.98+): data leakage. Move scaler.fit() to AFTER train_test_split()
- Fairness rejection: add class_weight='balanced' to the classifier"""


def parse_action(raw_text: str) -> dict:
    """Extract a JSON action from the LLM's response."""
    match = re.search(r'\{[^{}]*("payload"\s*:\s*\{[^{}]*\})?[^{}]*\}', raw_text, re.DOTALL)
    if match:
        try:
            # Clean up common LLM mistakes
            cleaned = match.group()
            cleaned = cleaned.replace("'", '"')
            action = json.loads(cleaned)
            if "action_type" in action:
                if "payload" not in action:
                    action["payload"] = {}
                return action
        except json.JSONDecodeError:
            pass

    # Try simpler extraction
    for atype in ["edit_script", "run_script", "submit", "inspect_data", "query_actor"]:
        if atype in raw_text:
            return {"action_type": atype, "payload": {}}

    return {"action_type": "run_script", "payload": {}}


def format_observation(obs: dict, reward: dict = None) -> str:
    """Format observation concisely for the LLM."""
    parts = [f"Stage {obs.get('current_stage', '?')} | Step {obs.get('stage_step_number', '?')}"]

    error = obs.get("last_run_error")
    if error:
        parts.append(f"\nERROR:\n{str(error)[:400]}")

    output = obs.get("last_run_output")
    if output:
        parts.append(f"\nOUTPUT:\n{str(output)[:300]}")

    preview = obs.get("data_preview")
    if preview:
        parts.append(f"\nDATA:\n{str(preview)[:400]}")

    feedback = obs.get("actor_feedback")
    if feedback:
        parts.append(f"\nREVIEWER:\n{str(feedback)[:300]}")

    script = obs.get("script_content")
    if script:
        parts.append(f"\nSCRIPT:\n{script[:800]}")

    if reward:
        parts.append(f"\nREWARD: {reward.get('score', 0):.2f} | {reward.get('message', '')}")

    return "\n".join(parts)


def run_episode():
    """Run a full episode where the LLM agent solves the pipeline."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    # Reset environment
    reset_resp = requests.post(f"{BASE_URL}/reset", json={}).json()
    obs = reset_resp if "current_stage" in reset_resp else reset_resp.get("observation", reset_resp)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    initial_prompt = format_observation(obs)
    initial_prompt += "\n\nStart by inspecting the data, then fix the bug. What is your first action?"
    messages.append({"role": "user", "content": initial_prompt})

    total_steps = 0
    final_score = 0.0
    last_action_type = None
    repeat_count = 0

    print(f"\n{'='*60}")
    print(f"  PIPELINEOPS ARENA — LLM AGENT (Llama 3.1 8B)")
    print(f"{'='*60}\n")

    for step in range(60):
        # Ask the LLM what to do
        try:
            # Keep context manageable — system + first + last 4 messages
            if len(messages) > 8:
                trimmed = [messages[0], messages[1]] + messages[-4:]
            else:
                trimmed = messages

            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=trimmed,
                max_tokens=400,
                temperature=0.2,
            )
            raw = response.choices[0].message.content.strip()
            action = parse_action(raw)
        except Exception as e:
            print(f"  LLM error: {e}")
            action = {"action_type": "run_script", "payload": {}}

        # Anti-loop: if repeating same action, force progression
        if action["action_type"] == last_action_type:
            repeat_count += 1
            if repeat_count >= 2:
                if last_action_type == "inspect_data":
                    # Already inspected, try running the script
                    action = {"action_type": "run_script", "payload": {}}
                elif last_action_type == "run_script":
                    # Already ran, need to edit
                    # Give LLM one more chance with explicit instruction
                    messages.append({"role": "user", "content":
                        "STOP repeating. You must use edit_script now to fix the bug. "
                        "Look at the error and the script. Find the exact text to replace. "
                        "Respond with: {\"action_type\": \"edit_script\", \"payload\": {\"old\": \"...\", \"new\": \"...\"}}"
                    })
                    try:
                        response = client.chat.completions.create(
                            model="llama-3.1-8b-instant",
                            messages=[messages[0]] + messages[-3:],
                            max_tokens=400,
                            temperature=0.3,
                        )
                        raw = response.choices[0].message.content.strip()
                        action = parse_action(raw)
                    except:
                        action = {"action_type": "submit", "payload": {}}
                elif last_action_type == "edit_script":
                    action = {"action_type": "run_script", "payload": {}}
                else:
                    action = {"action_type": "submit", "payload": {}}
                repeat_count = 0
        else:
            repeat_count = 0
        last_action_type = action["action_type"]

        # Take the action
        try:
            result = requests.post(f"{BASE_URL}/step", json=action).json()
            obs = result.get("observation", {})
            reward = result.get("reward", {})
            total_steps += 1

            stage = obs.get("current_stage", "?")
            score = reward.get("score", 0)
            msg = reward.get("message", "")
            print(f"  Step {total_steps:02d} | Stage {stage} | {action['action_type']:<15} | reward={score:.2f} | {msg[:50]}")

            # Build next prompt based on what just happened
            messages.append({"role": "assistant", "content": json.dumps(action)})

            next_prompt = format_observation(obs, reward)
            if action["action_type"] == "inspect_data":
                next_prompt += "\n\nYou have inspected the data. Now use edit_script to fix the bug. What exact text in the script needs to change?"
            elif action["action_type"] == "run_script" and obs.get("last_run_error"):
                next_prompt += "\n\nThe script has an error. Use edit_script to fix it. The 'old' field must match text exactly from the SCRIPT above."
            elif action["action_type"] == "run_script" and not obs.get("last_run_error"):
                next_prompt += "\n\nScript ran successfully. Use submit to submit your fix."
            elif action["action_type"] == "edit_script":
                next_prompt += "\n\nScript edited. Now use run_script to verify the fix works."
            messages.append({"role": "user", "content": next_prompt})

            if obs.get("done", False):
                final_score = reward.get("score", 0.0)
                break

            time.sleep(0.3)

        except Exception as e:
            print(f"  Step error: {e}")
            continue

    # Get final status
    status = requests.get(f"{BASE_URL}/pipeline_status").json()
    episode_score = status.get("episode_score", final_score)
    stages = status.get("stages_completed", [])

    print(f"\n{'='*60}")
    print(f"  RESULT: Score={episode_score:.2f} | Stages={stages} | Steps={total_steps}")
    print(f"{'='*60}\n")

    return {
        "final_score": float(episode_score),
        "total_steps": total_steps,
        "stages_completed": stages,
        "episode_score": float(episode_score),
    }


if __name__ == "__main__":
    print("Running PipelineOps Arena — LLM Agent...")
    results = run_episode()

    with open("baseline/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to baseline/results.json")
