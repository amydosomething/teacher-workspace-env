"""
models.py
Data models for the Teacher Workspace Environment.
Simulates Google Workspace (Classroom + Sheets + Gmail + Calendar/Meet)
for a teacher's daily administrative workflow.
"""

from typing import Any, Dict, List, Literal, Optional
from pydantic import Field
from openenv.core.env_server.types import Action, Observation

# ── Valid tool names ────────────────────────────────────────────────────────
ToolName = Literal[
    # Google Classroom - read
    "list_classrooms",
    "get_classroom",
    "list_announcements",
    # Google Classroom - write
    "create_classroom",
    "delete_classroom",
    "create_announcement",
    "delete_announcement",
    "add_comment",
    "delete_comment",
    # Google Sheets - read
    "list_sheets",
    "get_cells",
    # Google Sheets - write
    "create_sheet",
    "delete_sheet",
    "update_cell",
    "add_note",
    "set_formula",
    "sort_range",
    "filter_range",
    # Gmail - read
    "list_inbox",
    "read_mail",
    "search_mail",
    # Gmail - write
    "create_draft",
    "send_mail",
    "categorise_mail",
    "star_mail",
    "mark_unread",
    "mark_important",
    "mark_spam",
    "delete_mail",
    "create_label",
    "assign_label",
    # Calendar / Meet - read
    "list_events",
    "get_event",
    # Calendar / Meet - write
    "create_event",
    "create_meet_event",
]


class TeacherAction(Action):
    """
    A single action the agent can take inside the Teacher Workspace.
    Uses a tool_name + params pattern so all 36 tools share one action class.

    Example:
        TeacherAction(tool_name="create_classroom",
                      params={"name": "Math 101", "section": "A"})
        TeacherAction(tool_name="send_mail",
                      params={"to": "alice@school.edu",
                               "subject": "Your grade",
                               "body": "You scored 88/100."})
    """
    tool_name: ToolName = Field(
        ...,
        description="Name of the tool to invoke."
    )
    params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Keyword arguments forwarded to the tool handler."
    )


class TeacherObservation(Observation):
    """
    What the agent sees after every step (or reset).

    Snapshot fields give the agent a live view of the full workspace state
    so it can plan its next action without needing explicit 'list' calls
    first (though those are still available and required for grader credit).
    """
    # ── Result of the last action ──────────────────────────────────────────
    success: bool = Field(
        True,
        description="Whether the last tool call succeeded."
    )
    result: Optional[Any] = Field(
        None,
        description="Return value of the tool (dict, list, string, or None)."
    )
    error: Optional[str] = Field(
        None,
        description="Error message if success=False, else None."
    )

    # ── Live workspace snapshot ────────────────────────────────────────────
    classrooms: Dict[str, Any] = Field(
        default_factory=dict,
        description="All classrooms keyed by class_id."
    )
    sheets: Dict[str, Any] = Field(
        default_factory=dict,
        description="All spreadsheet sheets keyed by sheet_name."
    )
    inbox: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All emails in the inbox (latest first)."
    )
    sent: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All sent emails."
    )
    drafts: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All email drafts."
    )
    calendar: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All calendar events."
    )
    labels: List[str] = Field(
        default_factory=list,
        description="All Gmail labels the teacher has created."
    )
    students: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Student roster with names/emails/ids."
    )
    parents: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="Parent contacts with names/emails/student_id mapping."
    )

    # ── Episode metadata ───────────────────────────────────────────────────
    step: int = Field(0, description="Current step number.")
    task_name: str = Field("", description="Active task identifier.")
    task_prompt: str = Field("", description="Natural-language task description shown to the agent.")
    done: bool = Field(False, description="True when the episode is complete.")
    reward: float = Field(0.0, description="Reward earned by the last action.")