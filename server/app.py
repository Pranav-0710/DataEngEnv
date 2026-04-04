"""
OpenEnv-required server entry point.
Re-exports the FastAPI app from app.main for multi-mode deployment.
"""
from app.main import app, main

if __name__ == "__main__":
    main()
