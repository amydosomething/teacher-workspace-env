"""
teacher_workspace_env_environment.py
Teacher Workspace Environment Implementation.

Simulates a Google Workspace environment (Classroom + Sheets + Gmail +
Calendar/Meet) for a teacher's daily administrative workflow.

All state is pure Python — no external APIs, no databases, no network calls.
"""
import re
from uuid import uuid4
from typing import Any, Dict, List, Optional
from openenv.core.env_server.interfaces import Environment
from openenv.core.env_server.types import State

try:
    from ..models import TeacherAction, TeacherObservation
except ImportError:
    from models import TeacherAction, TeacherObservation


# ── Read-only tools (reward = 0.0) ─────────────────────────────────────────
READ_TOOLS = {
    "list_classrooms", "get_classroom", "list_announcements",
    "list_sheets", "get_cells",
    "list_inbox", "read_mail", "search_mail",
    "list_events", "get_event",
}

# ── Per-task action whitelist ───────────────────────────────────────────────
def _build_allowed_actions(state: dict) -> dict:
    student_emails = {s["email"] for s in state["students"]}
    parent_emails  = {p["email"] for p in state["parents"]}

    def _get_failing_student_ids() -> set:
        """
        Get student IDs whose expected grade is < 60.
        Uses raw scores (C, D, E columns) directly so sort order
        of F column doesn't affect the result.
        """
        mg = state.get("sheets", {}).get("Math Gradebook", {})
        cells = mg.get("cells", {})
        failing = set()
        for row in range(2, 7):
            sid = cells.get(f"B{row}")
            c   = cells.get(f"C{row}")
            d   = cells.get(f"D{row}")
            e   = cells.get(f"E{row}")
            if sid and all(isinstance(v, (int, float)) for v in [c, d, e]):
                avg = (c + d + e) / 3
                if avg < 60:
                    failing.add(sid)
        return failing

    def _get_failing_student_names() -> list:
        failing_ids = _get_failing_student_ids()
        return [
            s["name"].split()[0]
            for s in state["students"]
            if s["id"] in failing_ids
        ]

    def _failing_student_rows() -> list:
        """
        Returns row numbers where the student is failing.
        Uses raw C/D/E scores, not F (which may be reordered by sort).
        """
        mg = state.get("sheets", {}).get("Math Gradebook", {})
        cells = mg.get("cells", {})
        failing_ids = _get_failing_student_ids()
        rows = []
        for row in range(2, 7):
            sid = cells.get(f"B{row}")
            if sid in failing_ids:
                rows.append(row)
        return rows

    def is_valid_note_cell(params: dict) -> bool:
        if params.get("sheet_name") != "Math Gradebook":
            return False
        cell = params.get("cell", "")
        if not cell.startswith("G"):
            return False
        try:
            row = int(cell[1:])
        except ValueError:
            return False
        # FIX 1: use raw C/D/E scores, not _failing_student_rows() which
        # depended on F being populated first — causing valid add_note calls
        # to be penalized when F column hadn't been written yet.
        mg = state.get("sheets", {}).get("Math Gradebook", {})
        cells = mg.get("cells", {})
        sid = cells.get(f"B{row}")
        c = cells.get(f"C{row}")
        d = cells.get(f"D{row}")
        e = cells.get(f"E{row}")
        if not (sid and all(isinstance(v, (int, float)) for v in [c, d, e])):
            return False
        return (c + d + e) / 3 < 60

    def is_valid_grade_cell(params: dict) -> bool:
        if params.get("sheet_name") != "Math Gradebook":
            return False
        cell = params.get("cell", "")
        if not cell.startswith("F"):
            return False
        try:
            row = int(cell[1:])
            return 2 <= row <= 6
        except ValueError:
            return False

    def is_valid_formula_cell(params: dict) -> bool:
        """set_formula allowed in Task 2 only for F2:F6 in Math Gradebook."""
        if params.get("sheet_name") != "Math Gradebook":
            return False
        cell = params.get("cell", "")
        if not cell.startswith("F"):
            return False
        try:
            row = int(cell[1:])
            return 2 <= row <= 6
        except ValueError:
            return False

    def is_failing_meet(params: dict) -> bool:
        mg = state.get("sheets", {}).get("Math Gradebook", {})
        cells = mg.get("cells", {})
        failing_names = set()
        for row in range(2, 7):
            c = cells.get(f"C{row}")
            d = cells.get(f"D{row}")
            e = cells.get(f"E{row}")
            name = cells.get(f"A{row}", "")
            if all(isinstance(v, (int, float)) for v in [c, d, e]):
                if (c + d + e) / 3 < 60:
                    failing_names.add(name.split()[0])
        title = params.get("title", "")
        return any(name in title for name in failing_names)

    return {
        "setup_new_course": {
            "create_classroom":    lambda p: "Computer Science 101" in p.get("name", ""),
            "create_sheet":        lambda p: p.get("sheet_name") == "CS Gradebook",
            "create_announcement": lambda p: True,
        },
        "grade_and_notify": {
            "update_cell": is_valid_grade_cell,
            "set_formula": is_valid_formula_cell,  # ← added
            "add_note":    is_valid_note_cell,
            "send_mail":   lambda p: p.get("to") in student_emails,
            "star_mail":   lambda p: p.get("mail_id") == "mail_002",
        },
        "end_of_semester": {
            "set_formula":       lambda p: p.get("sheet_name") == "Math Gradebook",
            "sort_range":        lambda p: (
                p.get("sheet_name") == "Math Gradebook" and
                p.get("column") == "F"
            ),
            "create_label":      lambda p: p.get("name") == "End of Semester",
            "send_mail":         lambda p: p.get("to") in parent_emails,
            "assign_label": lambda p: (
                p.get("label") == "End of Semester"
                and any(
                    m.get("to") in parent_emails
                    and "End of Semester Report" in m.get("subject", "")
                    and m.get("mail_id") == p.get("mail_id")
                    for m in state.get("sent", [])
                )
            ),
            "create_meet_event": is_failing_meet,
        },
    }


# ══════════════════════════════════════════════════════════════════════════════
# SEED DATA  – realistic, interdependent data across all four apps
# ══════════════════════════════════════════════════════════════════════════════

def _build_initial_state() -> Dict[str, Any]:
    """
    Returns a fresh workspace state seeded with realistic data.

    The data is intentionally interdependent:
    - Students in Classroom match rows in the Sheets gradebook.
    - Unread emails in Gmail are from those same students / parents.
    - A parent-teacher meeting already exists in Calendar for reference.
    """
    students = [
        {"name": "Alice Johnson",  "email": "alice@students.school.edu",  "id": "s001"},
        {"name": "Bob Martinez",   "email": "bob@students.school.edu",    "id": "s002"},
        {"name": "Clara Singh",    "email": "clara@students.school.edu",  "id": "s003"},
        {"name": "David Lee",      "email": "david@students.school.edu",  "id": "s004"},
        {"name": "Eva Patel",      "email": "eva@students.school.edu",    "id": "s005"},
    ]
    parents = [
        {"name": "Mr. Johnson",  "email": "johnson.parent@gmail.com",  "student_id": "s001"},
        {"name": "Mrs. Martinez","email": "martinez.parent@gmail.com", "student_id": "s002"},
        {"name": "Mr. Singh",    "email": "singh.parent@gmail.com",    "student_id": "s003"},
        {"name": "Mrs. Lee",     "email": "lee.parent@gmail.com",      "student_id": "s004"},
        {"name": "Mr. Patel",    "email": "patel.parent@gmail.com",    "student_id": "s005"},
    ]

    # ── Google Classroom ───────────────────────────────────────────────────
    classrooms = {
        "cls_math101": {
            "class_id":      "cls_math101",
            "name":          "Mathematics 101",
            "section":       "Grade 10 - Section A",
            "description":   "Algebra and geometry fundamentals",
            "students":      students,
            "announcements": [
                {
                    "announcement_id": "ann_001",
                    "text":  "Midterm exam next Friday. Please review chapters 4–6.",
                    "date":  "2025-03-10",
                    "comments": [],
                }
            ],
        },
        "cls_sci101": {
            "class_id":      "cls_sci101",
            "name":          "Science 101",
            "section":       "Grade 10 - Section A",
            "description":   "Introduction to physics and chemistry",
            "students":      students,
            "announcements": [],
        },
    }

    # ── Google Sheets gradebook ────────────────────────────────────────────
    # One sheet per subject; rows mirror the student list above.
    sheets = {
        "Math Gradebook": {
            "sheet_name": "Math Gradebook",
            "cells": {
                "A1": "Student Name", "B1": "Student ID",
                "C1": "Midterm (%)",  "D1": "Assignment 1 (%)",
                "E1": "Assignment 2 (%)", "F1": "Final Grade (%)",
                "G1": "Notes",

                "A2": "Alice Johnson",  "B2": "s001",
                "C2": 78, "D2": 85, "E2": 90, "F2": "", "G2": "",

                "A3": "Bob Martinez",   "B3": "s002",
                "C3": 55, "D3": 60, "E3": 58, "F3": "", "G3": "Needs extra support",

                "A4": "Clara Singh",    "B4": "s003",
                "C4": 92, "D4": 95, "E4": 97, "F4": "", "G4": "",

                "A5": "David Lee",      "B5": "s004",
                "C5": 70, "D5": 72, "E5": 68, "F5": "", "G5": "",

                "A6": "Eva Patel",      "B6": "s005",
                "C6": 88, "D6": 91, "E6": 85, "F6": "", "G6": "",
            },
            "notes": {},
            "formulas": {},
        },
        "Science Gradebook": {
            "sheet_name": "Science Gradebook",
            "cells": {
                "A1": "Student Name", "B1": "Student ID",
                "C1": "Lab Report (%)", "D1": "Quiz 1 (%)",
                "E1": "Final Grade (%)", "F1": "Notes",

                "A2": "Alice Johnson",  "B2": "s001", "C2": 80, "D2": 76, "E2": "", "F2": "",
                "A3": "Bob Martinez",   "B3": "s002", "C3": 50, "D3": 48, "E3": "", "F3": "Absent for quiz",
                "A4": "Clara Singh",    "B4": "s003", "C4": 95, "D4": 98, "E4": "", "F4": "",
                "A5": "David Lee",      "B5": "s004", "C5": 74, "D5": 70, "E5": "", "F5": "",
                "A6": "Eva Patel",      "B6": "s005", "C6": 85, "D6": 88, "E6": "", "F6": "",
            },
            "notes": {},
            "formulas": {},
        },
    }

    # ── Gmail inbox ────────────────────────────────────────────────────────
    # Mix of student questions, parent concerns, and admin mail.
    inbox = [
        {
            "mail_id":   "mail_001",
            "from":      "bob@students.school.edu",
            "from_name": "Bob Martinez",
            "to":        "teacher@school.edu",
            "subject":   "Help with algebra homework",
            "body":      "Hi, I am struggling with quadratic equations. Can we schedule extra help?",
            "date":      "2025-03-12",
            "read":      False,
            "starred":   False,
            "important": False,
            "spam":      False,
            "labels":    [],
            "category":  "",
        },
        {
            "mail_id":   "mail_002",
            "from":      "martinez.parent@gmail.com",
            "from_name": "Mrs. Martinez",
            "to":        "teacher@school.edu",
            "subject":   "Bob's performance concern",
            "body":      "Dear Teacher, Bob seems to be struggling lately. Could we arrange a meeting?",
            "date":      "2025-03-11",
            "read":      False,
            "starred":   False,
            "important": False,
            "spam":      False,
            "labels":    [],
            "category":  "",
        },
        {
            "mail_id":   "mail_003",
            "from":      "principal@school.edu",
            "from_name": "Principal Adams",
            "to":        "teacher@school.edu",
            "subject":   "End-of-semester grade submission deadline",
            "body":      "Please submit all final grades by April 15th via the gradebook system.",
            "date":      "2025-03-10",
            "read":      True,
            "starred":   False,
            "important": True,
            "spam":      False,
            "labels":    [],
            "category":  "admin",
        },
        {
            "mail_id":   "mail_004",
            "from":      "alice@students.school.edu",
            "from_name": "Alice Johnson",
            "to":        "teacher@school.edu",
            "subject":   "Assignment submission",
            "body":      "Hi! I have submitted my Assignment 2. Please let me know if you received it.",
            "date":      "2025-03-09",
            "read":      True,
            "starred":   False,
            "important": False,
            "spam":      False,
            "labels":    [],
            "category":  "",
        },
        {
            "mail_id":   "mail_005",
            "from":      "noreply@schoolads.com",
            "from_name": "School Ads",
            "to":        "teacher@school.edu",
            "subject":   "Buy discounted stationery!",
            "body":      "Click here for amazing deals on school supplies!",
            "date":      "2025-03-08",
            "read":      False,
            "starred":   False,
            "important": False,
            "spam":      False,
            "labels":    [],
            "category":  "",
        },
    ]

    # ── Gmail sent + drafts ────────────────────────────────────────────────
    sent = []
    drafts = []
    labels = ["Important", "Parents", "Students", "Admin"]

    # ── Google Calendar ────────────────────────────────────────────────────
    calendar = [
        {
            "event_id":     "evt_001",
            "title":        "Staff Meeting",
            "date":         "2025-03-15",
            "time":         "09:00",
            "participants": ["teacher@school.edu", "principal@school.edu"],
            "meet_link":    None,
            "description":  "Monthly staff sync",
        },
        {
            "event_id":     "evt_002",
            "title":        "Math 101 Midterm Exam",
            "date":         "2025-03-20",
            "time":         "10:00",
            "participants": ["teacher@school.edu"] + [s["email"] for s in students],
            "meet_link":    None,
            "description":  "Midterm covering chapters 4-6",
        },
    ]

    return {
        "classrooms": classrooms,
        "sheets":     sheets,
        "inbox":      inbox,
        "sent":       sent,
        "drafts":     drafts,
        "labels":     labels,
        "calendar":   calendar,
        "students":   students,
        "parents":    parents,
    }


# ══════════════════════════════════════════════════════════════════════════════
# TASKS
# ══════════════════════════════════════════════════════════════════════════════

TASKS = {
    "setup_new_course": {
        "name":   "setup_new_course",
        "prompt": (
            "A new elective course 'Computer Science 101' is starting next week for "
            "Grade 10 Section A. You need to:\n"
            "1. Create a new classroom named 'Computer Science 101' with section "
            "'Grade 10 - Section A'.\n"
            "2. Create a new gradebook sheet called 'CS Gradebook' with headers: "
            "Student Name, Student ID, Assignment 1 (%), Assignment 2 (%), "
            "Final Grade (%), Notes.\n"
            "3. Post a welcome announcement in the new classroom: "
            "'Welcome to Computer Science 101! Please check the gradebook for "
            "upcoming assignments.'\n"
            "Complete all three steps."
        ),
        "difficulty": "easy",
    },

    "grade_and_notify": {
        "name":   "grade_and_notify",
        "prompt": (
            "End-of-week grading duties:\n"
            "1. Calculate the Math final grade for each student as the average of "
            "Midterm (%), Assignment 1 (%), and Assignment 2 (%) and update column F "
            "in 'Math Gradebook' (rows 2-6).\n"
            "   For grade calculations: average = (C + D + E) / 3, computed precisely. "
            "   Example: (55 + 60 + 58) / 3 = 57.67. Do NOT round to nearest whole number.\n"
            "2. In the 'Math Gradebook', find the student whose final grade is below 60. "
            "   Add a note to column G of that student's row: 'Recommended for tutoring'.\n"
            "   (Look up which row that student is in based on their computed final grade.)\n"
            "3. Send an individual email to each student with their Math final grade.\n"
            "   Subject: 'Your Math 101 Final Grade'\n"
            "   Body: 'Hi <name>, your final Math grade is <grade>%. Keep it up!'\n"
            "4. Star the email from Mrs. Martinez (mail_002) — it needs follow-up.\n"
            "Complete all steps."
        ),
        "difficulty": "medium",
    },

    "end_of_semester": {
        "name":   "end_of_semester",
        "prompt": (
            "End-of-semester admin tasks:\n"
            "1. In 'Math Gradebook', set a formula in F2 to calculate Alice's final "
            "grade as AVERAGE(C2,D2,E2). Do the same for rows 3-6 for all students.\n"
            "2. Sort the 'Math Gradebook' by column F (Final Grade) in descending order.\n"
            "3. Create a Gmail label called 'End of Semester'.\n"
            "4. Send a grade report email to each student's parent with subject "
            "'End of Semester Report' and body: "
            "'Dear <parent_name>, your child <student_name> has completed the semester. "
            "Please contact us to discuss their progress.'\n"
            "5. Assign the label 'End of Semester' to all parent emails you just sent.\n"
            "6. Schedule a parent-teacher meeting for each failing student that is the student whose final grade is less than 60 in the gradebook"
            "as a Google Meet event titled 'Parent Meeting - <student_name>' on "
            "2025-04-20 at 14:00, inviting the parent's email.\n"
            "Complete all steps."
        ),
        "difficulty": "hard",
    },
}


# ══════════════════════════════════════════════════════════════════════════════
# ENVIRONMENT
# ══════════════════════════════════════════════════════════════════════════════

class TeacherWorkspaceEnvironment(Environment):
    """
    Teacher Workspace RL Environment.

    The agent acts as a teacher managing a school day through four simulated
    Google Workspace apps. All state lives in-memory as Python dicts.
    """

    SUPPORTS_CONCURRENT_SESSIONS: bool = True

    def __init__(self):
        self._state:    Optional[Dict[str, Any]] = None
        self._ep_state: Optional[State]          = None
        self._task:     Optional[Dict[str, Any]] = None
        self._rewards:  List[float]              = []
        self._rewarded: Dict[str, bool]          = {}

    # ──────────────────────────────────────────────────────────────────────
    # OpenEnv interface
    # ──────────────────────────────────────────────────────────────────────

    def reset(self, task_name: str = "setup_new_course", **kwargs) -> TeacherObservation:
        """Reset workspace to a fresh seeded state and load the requested task."""
        self._state    = _build_initial_state()
        self._ep_state = State(episode_id=str(uuid4()), step_count=0)
        self._task     = TASKS.get(task_name, TASKS["setup_new_course"])
        self._rewards  = []
        self._last_action  = None   # add this
        self._repeat_count = 0  

        return self._make_obs(
            success=True,
            result={"message": "Workspace ready. Read the task prompt and begin."},
            reward=0.0,
            done=False,
        )

    def step(self, action: TeacherAction) -> TeacherObservation:  # type: ignore[override]
        """Route the action to the correct handler and return an observation."""
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._ep_state.step_count += 1

        tool = action.tool_name
        params = action.params or {}

            # Penalise repeated identical read calls (infinite loop behavior)
        if not hasattr(self, '_last_action'):
            self._last_action = None
            self._repeat_count = 0

        if tool == self._last_action and tool in READ_TOOLS:
            self._repeat_count += 1
            if self._repeat_count >= 2:
                obs = self._make_obs(
                    success=False,
                    result=None,
                    error=f"Repeated read '{tool}' with no write in between — unproductive loop.",
                    reward=-0.05,
                    done=False,
                )
                self._rewards.append(-0.05)
                return obs
        else:
            self._repeat_count = 0
        self._last_action = tool

        # Dispatch
        handler = getattr(self, f"_tool_{tool}", None)
        if handler is None:
            obs = self._make_obs(
                success=False,
                result=None,
                error=f"Unknown tool: {tool}",
                reward=-0.05,
                done=False,
            )
            self._rewards.append(-0.05)
            return obs

        try:
            result, reward, done = handler(**params)

            audit_penalty = self._audit_action(tool, params)
            if audit_penalty < 0:
                # Override reward to 0 first, then apply penalty
                # This prevents partial_reward canceling out the audit
                reward = audit_penalty

            self._rewards.append(reward)
            return self._make_obs(
                success=True,
                result=result,
                reward=reward,
                done=done,
                error=(
                    f"Unnecessary or incorrect action: "
                    f"{abs(audit_penalty):.2f} penalty applied"
                ) if audit_penalty < 0 else None,
            )
        except TypeError as e:
            obs = self._make_obs(
                success=False,
                result=None,
                error=f"Bad params for {tool}: {e}",
                reward=-0.05,
                done=False,
            )
            self._rewards.append(-0.05)
            return obs

    @property
    def state(self) -> State:
        if self._ep_state is None:
            raise RuntimeError("Call reset() first.")
        return self._ep_state

    # ──────────────────────────────────────────────────────────────────────
    # Observation builder
    # ──────────────────────────────────────────────────────────────────────

    def _make_obs(
        self,
        success: bool,
        result: Any,
        reward: float,
        done: bool,
        error: Optional[str] = None,
    ) -> TeacherObservation:
        s = self._state or {}
        return TeacherObservation(
            success=success,
            result=result,
            error=error,
            classrooms=s.get("classrooms", {}),
            sheets=s.get("sheets", {}),
            inbox=s.get("inbox", []),
            sent=s.get("sent", []),
            drafts=s.get("drafts", []),
            calendar=s.get("calendar", []),
            labels=s.get("labels", []),
            students=s.get("students", []),
            parents=s.get("parents", []),
            step=self._ep_state.step_count if self._ep_state else 0,
            task_name=self._task["name"] if self._task else "",
            task_prompt=self._task["prompt"] if self._task else "",
            done=done,
            reward=reward,
        )

    # ══════════════════════════════════════════════════════════════════════
    # ── GOOGLE CLASSROOM HANDLERS ─────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    def _tool_list_classrooms(self) -> tuple:
        """Read – returns all classrooms. reward=0.0"""
        classrooms = [
            {"class_id": cid, "name": c["name"], "section": c["section"]}
            for cid, c in self._state["classrooms"].items()
        ]
        return classrooms, 0.0, False

    def _tool_get_classroom(self, class_id: str) -> tuple:
        """Read – returns full details of one classroom."""
        cls = self._state["classrooms"].get(class_id)
        if cls is None:
            raise TypeError(f"Classroom '{class_id}' not found.")
        return cls, 0.0, False

    def _tool_list_announcements(self, class_id: str) -> tuple:
        """Read – returns announcements for a classroom."""
        cls = self._state["classrooms"].get(class_id)
        if cls is None:
            raise TypeError(f"Classroom '{class_id}' not found.")
        return cls["announcements"], 0.0, False

    def _tool_create_classroom(self, name: str, section: str,
                                description: str = "") -> tuple:
        """Write – creates a new classroom."""
        class_id = f"cls_{uuid4().hex[:8]}"
        self._state["classrooms"][class_id] = {
            "class_id":      class_id,
            "name":          name,
            "section":       section,
            "description":   description,
            "students":      self._state["students"],  # same cohort
            "announcements": [],
        }
        reward = self._partial_reward(0.25, "create_classroom")
        done   = self._check_done()
        return {"class_id": class_id, "name": name}, reward, done

    def _tool_delete_classroom(self, class_id: str) -> tuple:
        """Write – deletes a classroom. Penalises if class has students."""
        cls = self._state["classrooms"].pop(class_id, None)
        if cls is None:
            raise TypeError(f"Classroom '{class_id}' not found.")
        penalty = -0.1 if cls.get("students") else 0.0
        return {"deleted": class_id}, penalty, False

    def _tool_create_announcement(self, class_id: str, text: str) -> tuple:
        """Write – posts an announcement to a classroom."""
        cls = self._state["classrooms"].get(class_id)
        if cls is None:
            raise TypeError(f"Classroom '{class_id}' not found.")
        ann_id = f"ann_{uuid4().hex[:6]}"
        announcement = {
            "announcement_id": ann_id,
            "text":            text,
            "date":            "2025-03-13",
            "comments":        [],
        }
        cls["announcements"].append(announcement)
        reward = self._partial_reward(0.2, "create_announcement")
        done   = self._check_done()
        return {"announcement_id": ann_id}, reward, done

    def _tool_delete_announcement(self, class_id: str,
                                   announcement_id: str) -> tuple:
        """Write – deletes an announcement."""
        cls = self._state["classrooms"].get(class_id)
        if cls is None:
            raise TypeError(f"Classroom '{class_id}' not found.")
        before = len(cls["announcements"])
        cls["announcements"] = [
            a for a in cls["announcements"]
            if a["announcement_id"] != announcement_id
        ]
        if len(cls["announcements"]) == before:
            raise TypeError(f"Announcement '{announcement_id}' not found.")
        return {"deleted": announcement_id}, 0.05, False

    def _tool_add_comment(self, announcement_id: str, text: str) -> tuple:
        """Write – adds a comment to an announcement."""
        for cls in self._state["classrooms"].values():
            for ann in cls["announcements"]:
                if ann["announcement_id"] == announcement_id:
                    comment = {
                        "comment_id": f"cmt_{uuid4().hex[:6]}",
                        "text":       text,
                        "date":       "2025-03-13",
                    }
                    ann["comments"].append(comment)
                    return {"comment_id": comment["comment_id"]}, 0.05, False
        raise TypeError(f"Announcement '{announcement_id}' not found.")

    def _tool_delete_comment(self, announcement_id: str,
                              comment_id: str) -> tuple:
        """Write – deletes a comment."""
        for cls in self._state["classrooms"].values():
            for ann in cls["announcements"]:
                if ann["announcement_id"] == announcement_id:
                    before = len(ann["comments"])
                    ann["comments"] = [
                        c for c in ann["comments"]
                        if c["comment_id"] != comment_id
                    ]
                    if len(ann["comments"]) == before:
                        raise TypeError(f"Comment '{comment_id}' not found.")
                    return {"deleted": comment_id}, 0.0, False
        raise TypeError(f"Announcement '{announcement_id}' not found.")

    # ══════════════════════════════════════════════════════════════════════
    # ── GOOGLE SHEETS HANDLERS ────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    def _tool_list_sheets(self) -> tuple:
        """Read – returns names of all sheets."""
        return list(self._state["sheets"].keys()), 0.0, False

    def _tool_get_cells(self, sheet_name: str,
                         cell_range: str = "A1:Z100") -> tuple:
        """Read – returns cells from a sheet. cell_range is informational."""
        sheet = self._state["sheets"].get(sheet_name)
        if sheet is None:
            raise TypeError(f"Sheet '{sheet_name}' not found.")
        return sheet["cells"], 0.0, False

    def _tool_create_sheet(self, sheet_name: str,
                            headers: Optional[List[str]] = None) -> tuple:
        """Write – creates a new spreadsheet sheet with optional headers."""
        if sheet_name in self._state["sheets"]:
            raise TypeError(f"Sheet '{sheet_name}' already exists.")
        cells: Dict[str, Any] = {}
        if headers:
            for i, h in enumerate(headers):
                col = chr(ord("A") + i)
                cells[f"{col}1"] = h
        self._state["sheets"][sheet_name] = {
            "sheet_name": sheet_name,
            "cells":      cells,
            "notes":      {},
            "formulas":   {},
        }
        reward = self._partial_reward(0.2, "create_sheet")
        done   = self._check_done()
        return {"sheet_name": sheet_name, "headers": headers}, reward, done

    def _tool_delete_sheet(self, sheet_name: str) -> tuple:
        """Write – deletes a sheet."""
        if sheet_name not in self._state["sheets"]:
            raise TypeError(f"Sheet '{sheet_name}' not found.")
        del self._state["sheets"][sheet_name]
        return {"deleted": sheet_name}, 0.0, False

    def _tool_update_cell(self, sheet_name: str, cell: str,
                           value: Any) -> tuple:
        """Write – sets the value of a single cell."""
        sheet = self._state["sheets"].get(sheet_name)
        if sheet is None:
            raise TypeError(f"Sheet '{sheet_name}' not found.")
        sheet["cells"][cell] = value
        reward = self._partial_reward(0.05, f"update_cell_{sheet_name}_{cell}")
        done   = self._check_done()
        return {"cell": cell, "value": value}, reward, done

    def _tool_add_note(self, sheet_name: str, cell: str, note: str) -> tuple:
        """Write – adds a text note to a cell."""
        sheet = self._state["sheets"].get(sheet_name)
        if sheet is None:
            raise TypeError(f"Sheet '{sheet_name}' not found.")
        sheet["notes"][cell] = note
        reward = self._partial_reward(0.1, f"add_note_{sheet_name}_{cell}")
        done   = self._check_done()
        return {"cell": cell, "note": note}, reward, done

    def _tool_set_formula(self, sheet_name: str, cell: str,
                           formula: str) -> tuple:
        """
        Write – stores a formula string and computes a numeric result.
        Supports AVERAGE(C#,D#,E#) over cells in the same sheet.
        """
        sheet = self._state["sheets"].get(sheet_name)
        if sheet is None:
            raise TypeError(f"Sheet '{sheet_name}' not found.")

        computed = self._eval_formula(sheet, formula)
        sheet["formulas"][cell] = formula
        sheet["cells"][cell]    = computed

        reward = self._partial_reward(0.1, f"set_formula_{sheet_name}_{cell}")
        done   = self._check_done()
        return {"cell": cell, "formula": formula, "computed": computed}, reward, done

    def _tool_sort_range(self, sheet_name: str, column: str,
                          ascending: bool = False) -> tuple:
        """Write – sorts data rows (rows 2+) by a given column."""
        sheet = self._state["sheets"].get(sheet_name)
        if sheet is None:
            raise TypeError(f"Sheet '{sheet_name}' not found.")

        cells = sheet["cells"]
        # Determine how many data rows exist
        row_num = 2
        rows = []
        while f"A{row_num}" in cells or f"{column}{row_num}" in cells:
            row = {}
            for col_char in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
                key = f"{col_char}{row_num}"
                if key in cells:
                    row[col_char] = cells[key]
            rows.append(row)
            row_num += 1

        sort_val = lambda r: (r.get(column, 0) if isinstance(r.get(column, 0), (int, float)) else 0)
        rows.sort(key=sort_val, reverse=not ascending)

        # Write sorted rows back
        for i, row in enumerate(rows, start=2):
            for col_char, val in row.items():
                cells[f"{col_char}{i}"] = val

        reward = self._partial_reward(0.1, f"sort_{sheet_name}_{column}")
        done   = self._check_done()
        return {"sorted_by": column, "ascending": ascending, "rows": len(rows)}, reward, done

    def _tool_filter_range(self, sheet_name: str, column: str,
                            operator: str, value: Any) -> tuple:
        """Read – returns rows where column <op> value. Does not mutate state."""
        sheet = self._state["sheets"].get(sheet_name)
        if sheet is None:
            raise TypeError(f"Sheet '{sheet_name}' not found.")

        cells = sheet["cells"]
        matching = []
        row_num = 2
        while f"A{row_num}" in cells or f"{column}{row_num}" in cells:
            cell_val = cells.get(f"{column}{row_num}")
            try:
                cell_num = float(cell_val) if cell_val is not None else None
                ref_num  = float(value)
                match = (
                    (operator == ">"  and cell_num is not None and cell_num >  ref_num) or
                    (operator == ">=" and cell_num is not None and cell_num >= ref_num) or
                    (operator == "<"  and cell_num is not None and cell_num <  ref_num) or
                    (operator == "<=" and cell_num is not None and cell_num <= ref_num) or
                    (operator == "==" and cell_num is not None and cell_num == ref_num)
                )
            except (TypeError, ValueError):
                match = (operator == "==" and str(cell_val) == str(value))

            if match:
                row = {}
                for col_char in "ABCDEFG":
                    k = f"{col_char}{row_num}"
                    if k in cells:
                        row[col_char] = cells[k]
                row["_row"] = row_num
                matching.append(row)
            row_num += 1

        return matching, 0.0, False

    # ══════════════════════════════════════════════════════════════════════
    # ── GMAIL HANDLERS ────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    def _tool_list_inbox(self) -> tuple:
        """Read – returns all inbox emails (summary view)."""
        summary = [
            {
                "mail_id": m["mail_id"],
                "from":    m["from_name"],
                "subject": m["subject"],
                "date":    m["date"],
                "read":    m["read"],
                "starred": m["starred"],
            }
            for m in self._state["inbox"]
        ]
        return summary, 0.0, False

    def _tool_read_mail(self, mail_id: str) -> tuple:
        """Read – returns full content of one email and marks it as read."""
        mail = self._find_mail(mail_id)
        mail["read"] = True
        return mail, 0.0, False

    def _tool_search_mail(self, query: str) -> tuple:
        """Read – searches inbox by subject or sender (case-insensitive)."""
        q = query.lower()
        results = [
            m for m in self._state["inbox"]
            if q in m["subject"].lower()
            or q in m["from"].lower()
            or q in m["from_name"].lower()
            or q in m["body"].lower()
        ]
        return results, 0.0, False

    def _tool_create_draft(self, to: str, subject: str, body: str) -> tuple:
        """Write – saves an email draft."""
        draft_id = f"dft_{uuid4().hex[:6]}"
        draft = {
            "draft_id": draft_id,
            "to":       to,
            "subject":  subject,
            "body":     body,
            "date":     "2025-03-13",
        }
        self._state["drafts"].append(draft)
        return {"draft_id": draft_id}, 0.05, False

    def _tool_send_mail(self, to: str, subject: str, body: str) -> tuple:
        """Write – sends an email. Also adds it to sent box."""
        mail_id = f"mail_{uuid4().hex[:6]}"
        mail = {
            "mail_id": mail_id,
            "from":    "teacher@school.edu",
            "to":      to,
            "subject": subject,
            "body":    body,
            "date":    "2025-03-13",
            "labels":  [],
        }
        self._state["sent"].append(mail)
        reward = self._partial_reward(0.1, f"send_mail_{to}")
        done   = self._check_done()
        return {"mail_id": mail_id, "to": to}, reward, done

    def _tool_categorise_mail(self, mail_id: str, category: str) -> tuple:
        """Write – sets the category field of an email."""
        mail = self._find_mail(mail_id)
        mail["category"] = category
        reward = self._partial_reward(0.05, f"categorise_{mail_id}")
        done   = self._check_done()
        return {"mail_id": mail_id, "category": category}, reward, done

    def _tool_star_mail(self, mail_id: str) -> tuple:
        """Write – stars an email."""
        mail = self._find_mail(mail_id)
        mail["starred"] = True
        reward = self._partial_reward(0.1, f"star_{mail_id}")
        done   = self._check_done()
        return {"mail_id": mail_id, "starred": True}, reward, done

    def _tool_mark_unread(self, mail_id: str) -> tuple:
        """Write – marks an email as unread."""
        mail = self._find_mail(mail_id)
        mail["read"] = False
        return {"mail_id": mail_id, "read": False}, 0.05, False

    def _tool_mark_important(self, mail_id: str) -> tuple:
        """Write – marks an email as important."""
        mail = self._find_mail(mail_id)
        mail["important"] = True
        reward = self._partial_reward(0.05, f"important_{mail_id}")
        done   = self._check_done()
        return {"mail_id": mail_id, "important": True}, reward, done

    def _tool_mark_spam(self, mail_id: str) -> tuple:
        """Write – marks an email as spam."""
        mail = self._find_mail(mail_id)
        mail["spam"] = True
        reward = self._partial_reward(0.1, f"spam_{mail_id}")
        done   = self._check_done()
        return {"mail_id": mail_id, "spam": True}, reward, done

    def _tool_delete_mail(self, mail_id: str) -> tuple:
        """Write – deletes an email from inbox."""
        before = len(self._state["inbox"])
        self._state["inbox"] = [
            m for m in self._state["inbox"] if m["mail_id"] != mail_id
        ]
        if len(self._state["inbox"]) == before:
            raise TypeError(f"Mail '{mail_id}' not found.")
        return {"deleted": mail_id}, 0.0, False

    def _tool_create_label(self, name: str) -> tuple:
        """Write – creates a new Gmail label."""
        if name in self._state["labels"]:
            raise TypeError(f"Label '{name}' already exists.")
        self._state["labels"].append(name)
        reward = self._partial_reward(0.1, f"create_label_{name}")
        done   = self._check_done()
        return {"label": name}, reward, done

    def _tool_assign_label(self, mail_id: str, label: str) -> tuple:
        """Write – assigns a label to a sent or inbox email."""
        # Search both inbox and sent
        mail = None
        for m in self._state["inbox"] + self._state["sent"]:
            if m["mail_id"] == mail_id:
                mail = m
                break
        if mail is None:
            raise TypeError(f"Mail '{mail_id}' not found.")
        if label not in self._state["labels"]:
            raise TypeError(f"Label '{label}' does not exist. Create it first.")
        if label not in mail["labels"]:
            mail["labels"].append(label)
        # FIX 2: only give reward if the label is one the active task actually requires.
        # Prevents spurious reward for off-task labels like "Grades" in grade_and_notify,
        # or labelling auto-generated meet-invite emails in end_of_semester.
        task_name = self._task["name"] if self._task else ""
        required_labels = {
            "end_of_semester": "End of Semester",
        }
        required_label = required_labels.get(task_name)

        # For end_of_semester, also verify the mail being labelled is an actual
        # parent report email (sent box, correct subject) — not a meet invite.
        if required_label and label == required_label:
            sent_mail = next((m for m in self._state["sent"] if m["mail_id"] == mail_id), None)
            parent_emails = {p["email"] for p in self._state["parents"]}
            if (
                sent_mail
                and sent_mail.get("to") in parent_emails
                and "End of Semester Report" in sent_mail.get("subject", "")
            ):
                reward = self._partial_reward(0.05, f"assign_label_{mail_id}_{label}")
            else:
                reward = 0.0
        else:
            reward = 0.0
        done   = self._check_done()
        return {"mail_id": mail_id, "label": label}, reward, done

    # ══════════════════════════════════════════════════════════════════════
    # ── CALENDAR / MEET HANDLERS ──────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    def _tool_list_events(self) -> tuple:
        """Read – returns all calendar events."""
        return self._state["calendar"], 0.0, False

    def _tool_get_event(self, event_id: str) -> tuple:
        """Read – returns full details of one event."""
        evt = next(
            (e for e in self._state["calendar"] if e["event_id"] == event_id),
            None,
        )
        if evt is None:
            raise TypeError(f"Event '{event_id}' not found.")
        return evt, 0.0, False

    def _tool_create_event(self, title: str, date: str, time: str,
                            participants: Optional[List[str]] = None,
                            description: str = "") -> tuple:
        """Write – creates a calendar event (no Meet link)."""
        event_id = f"evt_{uuid4().hex[:6]}"
        event = {
            "event_id":     event_id,
            "title":        title,
            "date":         date,
            "time":         time,
            "participants": participants or [],
            "meet_link":    None,
            "description":  description,
        }
        self._state["calendar"].append(event)
        reward = self._partial_reward(0.15, f"create_event_{title}")
        done   = self._check_done()
        return {"event_id": event_id, "title": title}, reward, done

    def _tool_create_meet_event(self, title: str, date: str, time: str,
                                 participants: Optional[List[str]] = None,
                                 description: str = "") -> tuple:
        """Write – creates a calendar event with an auto-generated Meet link.
        Also auto-sends invite emails to all participants."""
        event_id  = f"evt_{uuid4().hex[:6]}"
        meet_link = f"https://meet.google.com/{uuid4().hex[:3]}-{uuid4().hex[:4]}-{uuid4().hex[:3]}"
        event = {
            "event_id":     event_id,
            "title":        title,
            "date":         date,
            "time":         time,
            "participants": participants or [],
            "meet_link":    meet_link,
            "description":  description,
        }
        self._state["calendar"].append(event)

        # Auto-send invite emails to participants
        for email in (participants or []):
            invite = {
                "mail_id": f"mail_{uuid4().hex[:6]}",
                "from":    "teacher@school.edu",
                "to":      email,
                "subject": f"Meeting Invite: {title}",
                "body":    (
                    f"You are invited to '{title}' on {date} at {time}.\n"
                    f"Join via Google Meet: {meet_link}"
                ),
                "date":    "2025-03-13",
                "labels":  [],
            }
            self._state["sent"].append(invite)

        reward = self._partial_reward(0.2, f"create_meet_{title}")
        done   = self._check_done()
        return {"event_id": event_id, "meet_link": meet_link}, reward, done

    # ══════════════════════════════════════════════════════════════════════
    # ── GRADERS ───────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    def grade(self) -> float:
        """
        Run the grader for the active task and return a score in [0.0, 1.0].
        Called externally by the inference script after the episode ends.
        """
        if self._task is None:
            return 0.0
        task_name = self._task["name"]
        if task_name == "setup_new_course":
            return self._grade_setup_new_course()
        if task_name == "grade_and_notify":
            return self._grade_grade_and_notify()
        if task_name == "end_of_semester":
            return self._grade_end_of_semester()
        return 0.0

    def _grade_setup_new_course(self) -> float:
        """
        Checks:
        1. A classroom named 'Computer Science 101' exists (+0.35)
        2. A sheet named 'CS Gradebook' exists with correct headers (+0.35)
        3. An announcement mentioning 'Welcome' exists in the new classroom (+0.30)
        """
        score = 0.0
        cs_class = next(
            (c for c in self._state["classrooms"].values()
             if "Computer Science 101" in c["name"]),
            None,
        )
        if cs_class:
            score += 0.35
            # Check welcome announcement
            has_welcome = any(
                "welcome" in a["text"].lower()
                for a in cs_class["announcements"]
            )
            if has_welcome:
                score += 0.30

        # Check CS Gradebook sheet
        cs_sheet = self._state["sheets"].get("CS Gradebook")
        if cs_sheet:
            headers = list(cs_sheet["cells"].values())
            required = ["Student Name", "Student ID", "Final Grade (%)"]
            if all(any(r.lower() in str(h).lower() for h in headers) for r in required):
                score += 0.35

        return round(min(score, 1.0), 2)

    def _grade_grade_and_notify(self) -> float:
        """
        Checks:
        1. Final grades computed in Math Gradebook F2:F6 (+0.30)
        2. Note added to failing student's G cell (+0.15)
        3. Emails sent to all 5 students with correct grade (+0.40)
        4. mail_002 starred (+0.15)
        """
        score = 0.0
        sheet = self._state["sheets"].get("Math Gradebook", {})
        cells = sheet.get("cells", {})

        expected_grades = {
            2: 84.33,  # Alice
            3: 57.67,  # Bob   ← failing
            4: 94.67,  # Clara
            5: 70.0,   # David
            6: 88.0,   # Eva
        }

        # 1. Final grades correct
        grades_correct = sum(
            1 for row, exp in expected_grades.items()
            if isinstance(cells.get(f"F{row}"), (int, float))
            and abs(cells.get(f"F{row}") - exp) < 0.1
        )
        score += (grades_correct / 5) * 0.30

        # 2. Note in G cell of failing student (row where F < 60)
        notes = sheet.get("notes", {})
        failing_row = next(
            (row for row, exp in expected_grades.items() if exp < 60), None
        )
        if failing_row and notes.get(f"G{failing_row}"):
            score += 0.15

        # 3. Emails to students with correct grade in body
        correct_emails = 0
        for student in self._state["students"]:
            sent = [m for m in self._state["sent"] if m["to"] == student["email"]]
            if not sent:
                continue
            body = sent[-1]["body"]
            row = next(
                (r for r in range(2, 7) if cells.get(f"B{r}") == student["id"]), None
            )
            expected = expected_grades.get(row)
            if expected:
                name_ok  = student["name"].split()[0].lower() in body.lower()
                numbers  = [float(x) for x in re.findall(r"\d+\.?\d*", body)]
                grade_ok = any(abs(n - expected) < 0.1 for n in numbers)
                if name_ok and grade_ok:
                    correct_emails += 1
        score += (correct_emails / 5) * 0.40

        # 4. Star mail_002
        mail_002 = self._find_mail_safe("mail_002")
        if mail_002 and mail_002.get("starred"):
            score += 0.15

        return round(min(score, 1.0), 2)

    def _grade_end_of_semester(self) -> float:
        """
        Checks:
        1. Formulas set in F2:F6 of Math Gradebook (+0.20)
        2. Math Gradebook sorted by F descending (+0.10)
        3. Label 'End of Semester' created (+0.10)
        4. Report emails sent to all 5 parents (+0.25)
        5. Sent emails labelled 'End of Semester' (+0.15)
        6. Meet events created for failing students (grade < 60) (+0.20)
        """
        score = 0.0
        sheet = self._state["sheets"].get("Math Gradebook", {})
        cells = sheet.get("cells", {})
        formulas = sheet.get("formulas", {})

        # 1. Formulas in F2:F6
        # 1. Formulas exist and computed values correct
        expected_grades = {
            "s001": 84.33, "s002": 57.67, "s003": 94.67, "s004": 70.0, "s005": 88.0
        }
        formula_correct = 0
        for r in range(2, 7):
            if formulas.get(f"F{r}"):
                sid = cells.get(f"B{r}")
                exp = expected_grades.get(sid)
                computed = cells.get(f"F{r}")
                if exp and isinstance(computed, (int, float)) and abs(computed - exp) < 0.1:
                    formula_correct += 1
        score += (formula_correct / 5) * 0.20

        # 2. Sorted descending by F
        grades = [cells.get(f"F{r}") for r in range(2, 7)]
        numeric = [g for g in grades if isinstance(g, (int, float))]
        if len(numeric) >= 2 and all(
            numeric[i] >= numeric[i + 1] for i in range(len(numeric) - 1)
        ):
            score += 0.10

        # 3. Label exists
        if "End of Semester" in self._state["labels"]:
            score += 0.10

        # 4. Parent emails with correct parent and student names
        correct_parent_emails = 0
        for parent in self._state["parents"]:
            sent = [m for m in self._state["sent"] if m["to"] == parent["email"]]
            if not sent:
                continue
            body = sent[-1]["body"]
            student = next((s for s in self._state["students"] if s["id"] == parent["student_id"]), None)
            if student:
                parent_name_ok = parent["name"].split()[-1].lower() in body.lower()
                student_name_ok = student["name"].split()[0].lower() in body.lower()
                if parent_name_ok and student_name_ok:
                    correct_parent_emails += 1
        score += (correct_parent_emails / 5) * 0.25

        # 5. Sent emails with 'End of Semester' label
        # Only count emails that are actual parent report emails —
        # i.e. sent to a parent email with subject 'End of Semester Report'
        # This prevents meet invite emails (auto-generated by create_meet_event)
        # from counting toward label credit.
        parent_emails = {p["email"] for p in self._state["parents"]}

        labelled = sum(
            1 for m in self._state["sent"]
            if "End of Semester" in m.get("labels", [])
            and m.get("to") in parent_emails
            and "End of Semester Report" in m.get("subject", "")
        )
        score += min(labelled / 5, 1.0) * 0.15

        # 6. Meet events for failing students only — penalise spurious extras
        failing_students = [
            s for s in self._state["students"]
            if expected_grades.get(s["id"], 100) < 60
        ]
        passing_students = [
            s for s in self._state["students"]
            if expected_grades.get(s["id"], 100) >= 60
        ]

        spurious_meets = sum(
            1 for s in passing_students
            if any(
                e.get("meet_link") and s["name"].split()[0] in e.get("title", "")
                for e in self._state["calendar"]
            )
        )

        correct_meets = 0
        for student in failing_students:
            parent = next(
                (p for p in self._state["parents"] if p["student_id"] == student["id"]),
                None,
            )
            if parent:
                match = any(
                    e.get("meet_link")
                    and student["name"].split()[0] in e.get("title", "")
                    and parent["email"] in e.get("participants", [])
                    for e in self._state["calendar"]
                )
                if match:
                    correct_meets += 1

        meet_score = (correct_meets / max(len(failing_students), 1)) - (spurious_meets * 0.20)
        score += max(meet_score, 0.0) * 0.20

        return round(min(score, 1.0), 2)

    # ══════════════════════════════════════════════════════════════════════
    # ── HELPERS ───────────────────────────────────────────────────────────
    # ══════════════════════════════════════════════════════════════════════

    def _find_mail(self, mail_id: str) -> Dict[str, Any]:
        """Find an email in inbox or sent; raise TypeError if not found."""
        for m in self._state["inbox"] + self._state["sent"]:
            if m["mail_id"] == mail_id:
                return m
        raise TypeError(f"Mail '{mail_id}' not found in inbox or sent.")

    def _find_mail_safe(self, mail_id: str) -> Optional[Dict[str, Any]]:
        """Find an email without raising."""
        for m in self._state["inbox"]:
            if m["mail_id"] == mail_id:
                return m
        return None

    def _get_student_math_grade(self, student_id: str) -> Optional[float]:
        """Return the computed final Math grade for a student, or None."""
        sheet = self._state["sheets"].get("Math Gradebook", {})
        cells = sheet.get("cells", {})
        for row in range(2, 7):
            if cells.get(f"B{row}") == student_id:
                val = cells.get(f"F{row}")
                if isinstance(val, (int, float)):
                    return float(val)
        return None

    def _eval_formula(self, sheet: Dict[str, Any], formula: str) -> Any:
        """
        Evaluate a simple AVERAGE(C#,D#,E#) formula over sheet cells.
        Returns the numeric result or the original formula string if unparseable.
        """
        import re
        m = re.match(r"AVERAGE\(([^)]+)\)", formula.strip(), re.IGNORECASE)
        if not m:
            return formula  # unsupported formula — store as-is
        refs = [r.strip() for r in m.group(1).split(",")]
        values = []
        for ref in refs:
            val = sheet["cells"].get(ref)
            if isinstance(val, (int, float)):
                values.append(float(val))
        if not values:
            return formula
        return round(sum(values) / len(values), 2)


    def _audit_action(self, tool: str, params: dict) -> float:
        """
        Returns -0.10 penalty if the action is not required by the task,
        or targets the wrong subject.
        Read-only tools are always free.
        Returns 0.0 if the action is legitimate.
        """
        if tool in READ_TOOLS:
            return 0.0

        task_name = self._task["name"] if self._task else ""
        allowed_map = _build_allowed_actions(self._state)
        allowed = allowed_map.get(task_name, {})

        if tool not in allowed:
            return -0.10

        validator = allowed[tool]
        if not validator(params):
            return -0.10

        return 0.0
    
    def _partial_reward(self, amount: float, key: str) -> float:
        """
        Returns `amount` the first time this key is seen this episode,
        0.0 on subsequent calls (prevents reward hacking by repeating actions).
        """
        if not hasattr(self, "_rewarded") or self._ep_state is None:
            return amount
        full_key = f"{self._ep_state.episode_id}:{key}"
        if full_key in self._rewarded:
            return 0.0
        self._rewarded[full_key] = True
        return amount

    def _check_done(self) -> bool:
        """
        An episode ends when the grader score reaches 1.0.
        We run a lightweight check here so the agent gets a done=True signal.
        """
        return self.grade() >= 1.0