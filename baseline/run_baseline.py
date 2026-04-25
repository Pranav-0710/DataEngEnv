"""
PipelineOps Arena — LLM-Powered Agent (No Hardcoded Answers)

The agent uses Groq's Llama 3.1 API to read error logs, inspect data,
reason about the bug, and generate fix actions entirely on its own.
"""

import os, time, json, re, requests

BASE_URL = os.environ.get("DATAENGENV_URL", "http://localhost:8000")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

SYSTEM_PROMPT = """You are a senior Data Engineer debugging a broken ML pipeline.
The pipeline has 4 stages. You must fix each stage by submitting a correct fix.

You will receive observations containing:
- script_content: the current Python script
- last_run_error: any error from the last run
- last_run_output: stdout from the last run
- data_preview: dataset info (columns, nulls, stats)
- actor_feedback: feedback from the MLOps reviewer bot

Available actions (respond with ONLY valid JSON, nothing else):

1. Inspect data first:
   {"action_type": "inspect_data", "payload": {}}

2. Check column schema:
   {"action_type": "check_schema", "payload": {}}

3. Run the script to see errors/output:
   {"action_type": "run_script", "payload": {}}

4. Edit the script (find and replace):
   {"action_type": "edit_script", "payload": {"old": "exact text to find", "new": "replacement text"}}

5. Ask the MLOps reviewer for feedback:
   {"action_type": "query_actor", "payload": {}}

6. Submit when you believe the fix is correct:
   {"action_type": "submit", "payload": {}}

STRATEGY:
- Start by inspecting data OR running the script to see the error
- Read the error carefully and diagnose the root cause
- Use edit_script to fix the bug (the "old" text must EXACTLY match the script)
- Run the script again to verify your fix works
- Submit once the script runs without errors

IMPORTANT: Respond with ONLY a JSON object. No explanation, no markdown, no extra text."""


def parse_action(raw_text: str) -> dict:
    """Extract a JSON action from the LLM's response."""
    # Try to find JSON in the response
    match = re.search(r'\{[^{}]*\}', raw_text, re.DOTALL)
    if match:
        try:
            action = json.loads(match.group())
            if "action_type" in action:
                return action
        except json.JSONDecodeError:
            pass
    # Fallback
    return {"action_type": "inspect_data", "payload": {}}


def format_observation(obs: dict, reward: dict = None) -> str:
    """Format the environment observation into a readable prompt for the LLM."""
    parts = []
    parts.append(f"Current Stage: {obs.get('current_stage', '?')}")
    parts.append(f"Step: {obs.get('stage_step_number', '?')}")

    if obs.get("last_run_error"):
        parts.append(f"\nERROR:\n{str(obs['last_run_error'])[:500]}")

    if obs.get("last_run_output"):
        parts.append(f"\nOUTPUT:\n{str(obs['last_run_output'])[:500]}")

    if obs.get("data_preview"):
        parts.append(f"\nDATA PREVIEW:\n{str(obs['data_preview'])[:600]}")

    if obs.get("actor_feedback"):
        parts.append(f"\nREVIEWER FEEDBACK:\n{obs['actor_feedback'][:400]}")

    if obs.get("script_content"):
        parts.append(f"\nCURRENT SCRIPT:\n{obs['script_content'][:1000]}")

    if reward:
        parts.append(f"\nREWARD: {reward.get('score', 0)} — {reward.get('message', '')}")

    return "\n".join(parts)


def run_episode():
    """Run a full episode where the LLM agent solves the pipeline."""
    from groq import Groq
    client = Groq(api_key=GROQ_API_KEY)

    # Reset environment
    reset_resp = requests.post(f"{BASE_URL}/reset", json={}).json()
    obs = reset_resp if "current_stage" in reset_resp else reset_resp.get("observation", reset_resp)

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages.append({"role": "user", "content": format_observation(obs)})

    total_steps = 0
    final_score = 0.0

    print(f"\n{'='*60}")
    print(f"  PIPELINEOPS ARENA — LLM AGENT (Llama 3.1 8B)")
    print(f"{'='*60}\n")

    for step in range(60):
        # Ask the LLM what to do
        try:
            # Keep context manageable — system + last 6 messages
            trimmed = [messages[0]] + messages[-6:]
            response = client.chat.completions.create(
                model="llama-3.1-8b-instant",
                messages=trimmed,
                max_tokens=300,
                temperature=0.1,
            )
            raw = response.choices[0].message.content.strip()
            action = parse_action(raw)
        except Exception as e:
            print(f"  LLM error: {e}")
            action = {"action_type": "inspect_data", "payload": {}}

        # Take the action
        try:
            result = requests.post(f"{BASE_URL}/step", json=action).json()
            obs = result.get("observation", {})
            reward = result.get("reward", {})
            total_steps += 1

            # Print what happened
            stage = obs.get("current_stage", "?")
            score = reward.get("score", 0)
            msg = reward.get("message", "")
            print(f"  Step {total_steps:02d} | Stage {stage} | {action['action_type']:<15} | reward={score:.2f} | {msg[:50]}")

            # Add to conversation
            messages.append({"role": "assistant", "content": json.dumps(action)})
            messages.append({"role": "user", "content": format_observation(obs, reward)})

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
    print("Running PipelineOps Arena — LLM Agent (no hardcoded answers)...")
    results = run_episode()

    with open("baseline/results.json", "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to baseline/results.json")
