# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Teacher Workspace Env Environment Client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import TeacherAction, TeacherObservation


class TeacherWorkspaceEnv(
    EnvClient[TeacherAction, TeacherObservation, State]
):
    """
    Client for the Teacher Workspace Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with TeacherWorkspaceEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.task_name)
        ...
        ...     result = client.step(TeacherAction(tool_name="list_classrooms", params={}))
        ...     print(result.observation.result)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = TeacherWorkspaceEnv.from_docker_image("teacher_workspace_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(TeacherAction(tool_name="list_sheets", params={}))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: TeacherAction) -> Dict:
        """
        Convert TeacherAction to JSON payload for step message.

        Args:
            action: TeacherAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return {
            "tool_name": action.tool_name,
            "params": action.params,
        }

    def _parse_result(self, payload: Dict) -> StepResult[TeacherObservation]:
        """
        Parse server response into StepResult[TeacherObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with TeacherObservation
        """
        obs_data = payload.get("observation", {})
        observation = TeacherObservation(
            success=obs_data.get("success", True),
            result=obs_data.get("result"),
            error=obs_data.get("error"),
            classrooms=obs_data.get("classrooms", {}),
            sheets=obs_data.get("sheets", {}),
            inbox=obs_data.get("inbox", []),
            sent=obs_data.get("sent", []),
            drafts=obs_data.get("drafts", []),
            calendar=obs_data.get("calendar", []),
            labels=obs_data.get("labels", []),
            students=obs_data.get("students", []),
            parents=obs_data.get("parents", []),
            step=obs_data.get("step", 0),
            task_name=obs_data.get("task_name", ""),
            task_prompt=obs_data.get("task_prompt", ""),
            done=payload.get("done", obs_data.get("done", False)),
            reward=payload.get("reward", obs_data.get("reward", 0.0)),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> State:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return State(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
        )
