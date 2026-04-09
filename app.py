# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
app.py
FastAPI application for the Teacher Workspace Env Environment.

This module creates an HTTP server that exposes the TeacherWorkspaceEnvironment
over HTTP and WebSocket endpoints, compatible with EnvClient.

Endpoints:
    - POST /reset: Reset the environment
    - POST /step: Execute an action
    - GET /state: Get current environment state
    - GET /schema: Get action/observation schemas
    - WS /ws: WebSocket endpoint for persistent sessions

Usage:
    # Development (with auto-reload):
    uvicorn server.app:app --reload --host 0.0.0.0 --port 8000

    # Production:
    uvicorn server.app:app --host 0.0.0.0 --port 8000 --workers 4

    # Or run directly:
    python -m server.app
"""

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from models import TeacherAction, TeacherObservation
    from server.teacher_workspace_env_environment import TeacherWorkspaceEnvironment
except ModuleNotFoundError:
    from ..models import TeacherAction, TeacherObservation
    from .teacher_workspace_env_environment import TeacherWorkspaceEnvironment

# Create the app with web interface and README integration
app = create_app(
    TeacherWorkspaceEnvironment,
    TeacherAction,
    TeacherObservation,
    env_name="teacher_workspace_env",
    max_concurrent_envs=1,  # increase this number to allow more concurrent WebSocket sessions
)

import gradio as gr
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from app import demo
gr.mount_gradio_app(app, demo, path="/")


def main() -> None:
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m teacher_workspace_env.server.app

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn teacher_workspace_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)


if __name__ == '__main__':
    main()
