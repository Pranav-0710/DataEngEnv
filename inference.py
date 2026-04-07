"""
OpenEnv inference entrypoint.
Starts the FastAPI server in a background thread, waits for it to be ready,
then runs the baseline agent which prints the required [START]/[STEP]/[END] blocks.
"""
import sys
import time
import threading
import httpx
import uvicorn

def start_server():
    """Run the FastAPI app in a background daemon thread."""
    config = uvicorn.Config("app.main:app", host="0.0.0.0", port=8000, log_level="warning")
    server = uvicorn.Server(config)
    server.run()

def wait_for_server(url="http://localhost:8000/health", timeout=30):
    """Block until the server is accepting connections."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            r = httpx.get(url, timeout=2)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False

def main():
    # Start server in background
    t = threading.Thread(target=start_server, daemon=True)
    t.start()

    # Wait until the server is ready
    if not wait_for_server():
        print("[START] task=startup_failed", flush=True)
        print("[STEP] step=0 reward=0.0", flush=True)
        print("[END] task=startup_failed score=0.0 steps=0", flush=True)
        sys.exit(1)

    # Run the baseline agent (prints [START]/[STEP]/[END] per task)
    from baseline.run_baseline import main as run_baseline
    run_baseline()

if __name__ == "__main__":
    main()
