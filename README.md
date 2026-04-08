# 🏫 Teacher Workspace Env

> A real-world OpenEnv RL environment simulating a teacher's daily administrative workflow across Google Classroom, Google Sheets, Gmail, and Google Calendar/Meet.

---

## Environment Description & Motivation

Teachers spend a significant portion of their day on administrative tasks: setting up courses, computing and communicating grades, managing parent communications, and scheduling meetings. This environment simulates that entire workflow as a structured RL problem, making it a strong benchmark for evaluating how well AI agents can execute multi-step, real-world productivity tasks in a Google Workspace-style setting.

The environment is fully deterministic and self-contained — no external APIs or databases are needed. All state lives in-memory as Python dicts, making it reproducible and fast.

---

## Action Space

All actions use a unified tool-invocation pattern:

**`TeacherAction`**

| Field | Type | Description |
|---|---|---|
| `tool_name` | `Literal[str]` | One of 36 available tool names (see table below) |
| `params` | `dict[str, Any]` | Keyword arguments forwarded to the tool handler |

### Available Tools

| Category | Tool | Params | Read/Write |
|---|---|---|---|
| **Google Classroom** | `list_classrooms` | — | Read |
| | `get_classroom` | `class_id` | Read |
| | `list_announcements` | `class_id` | Read |
| | `create_classroom` | `name, section, description` | Write |
| | `create_announcement` | `class_id, text` | Write |
| **Google Sheets** | `list_sheets` | — | Read |
| | `get_cells` | `sheet_name` | Read |
| | `create_sheet` | `sheet_name, headers` | Write |
| | `update_cell` | `sheet_name, cell, value` | Write |
| | `add_note` | `sheet_name, cell, note` | Write |
| | `set_formula` | `sheet_name, cell, formula` | Write |
| | `sort_range` | `sheet_name, column, ascending` | Write |
| | `filter_range` | `sheet_name, column, operator, value` | Read |
| **Gmail** | `list_inbox` | — | Read |
| | `read_mail` | `mail_id` | Read |
| | `search_mail` | `query` | Read |
| | `send_mail` | `to, subject, body` | Write |
| | `star_mail` | `mail_id` | Write |
| | `mark_important` | `mail_id` | Write |
| | `mark_spam` | `mail_id` | Write |
| | `create_label` | `name` | Write |
| | `assign_label` | `mail_id, label` | Write |
| | `categorise_mail` | `mail_id, category` | Write |
| **Calendar / Meet** | `list_events` | — | Read |
| | `get_event` | `event_id` | Read |
| | `create_event` | `title, date, time, participants, description` | Write |
| | `create_meet_event` | `title, date, time, participants, description` | Write |

---

## Observation Space

**`TeacherObservation`** — returned after every `step()` and `reset()`:

| Field | Type | Description |
|---|---|---|
| `success` | `bool` | Whether the last tool call succeeded |
| `result` | `Any \| None` | Return payload of the tool |
| `error` | `str \| None` | Error message if `success=False` |
| `classrooms` | `dict` | All classrooms keyed by `class_id` |
| `sheets` | `dict` | All spreadsheet sheets keyed by `sheet_name` |
| `inbox` | `list` | All inbox emails |
| `sent` | `list` | All sent emails (use these `mail_id`s for `assign_label`) |
| `drafts` | `list` | All email drafts |
| `calendar` | `list` | All calendar/Meet events |
| `labels` | `list[str]` | Gmail labels created so far |
| `students` | `list` | Student roster with names, emails, and IDs |
| `parents` | `list` | Parent contacts with names, emails, and `student_id` mapping |
| `step` | `int` | Current step number |
| `task_name` | `str` | Active task identifier |
| `task_prompt` | `str` | Natural-language task description shown to the agent |
| `done` | `bool` | `True` when the episode is complete |
| `reward` | `float` | Reward earned by the last action |

---

## Reward Function

The reward is shaped for partial progress and capped by deterministic graders:

- **Read-only tools** (`list_*`, `get_*`, `read_*`, `search_*`) always return `0.0`
- **Correct write actions** return a positive partial reward on the first call per episode (anti-reward-hacking: repeating the same successful action returns `0.0`)
- **Actions not required by the current task** return `−0.10` (audit penalty)
- **Repeated identical read calls** return `−0.05` (loop detection)
- **Bad params / unknown tools** return `−0.05`
- **Episode end score** is determined by a holistic grader that checks final state, normalized to `0.0–1.0`

---

## Tasks

### Task 1 — `setup_new_course` (Easy)

**Objective:** Set up a new Computer Science elective from scratch.

1. Create a classroom named `"Computer Science 101"` for `"Grade 10 - Section A"`
2. Create a gradebook sheet `"CS Gradebook"` with headers: Student Name, Student ID, Assignment 1 (%), Assignment 2 (%), Final Grade (%), Notes
3. Post a welcome announcement in the new classroom

**Expected steps:** ~3  
**Max steps:** 20

### Task 2 — `grade_and_notify` (Medium)

**Objective:** Complete end-of-week grading duties.

1. Calculate each student's Math final grade as `AVERAGE(Midterm, Assignment 1, Assignment 2)` — update column F in `Math Gradebook` (rows 2–6), rounded to 2 decimal places
2. Add a note `"Recommended for tutoring"` to column G of the student whose final grade is below 60
3. Send an individual email to each of the 5 students with their grade (subject: `"Your Math 101 Final Grade"`)
4. Star the email from Mrs. Martinez (`mail_002`) for follow-up

**Expected steps:** ~12  
**Max steps:** 30

### Task 3 — `end_of_semester` (Hard)

**Objective:** Complete all end-of-semester admin tasks.

1. Set `AVERAGE(C,D,E)` formulas in column F (F2–F6) of `Math Gradebook`
2. Sort the gradebook by column F in descending order
3. Create a Gmail label `"End of Semester"`
4. Send a grade report email to each of the 5 parents (subject: `"End of Semester Report"`)
5. Assign the `"End of Semester"` label to all 5 sent parent emails
6. Schedule a Google Meet event (`"Parent Meeting - <student_name>"`) on 2025-04-20 at 14:00 for each failing student (grade < 60), inviting the parent's email

**Expected steps:** ~18  
**Max steps:** 40

---

## Baseline Scores

Scores from two models on the standard inference script (`inference.py`):

| Task | Gemini 3 Flash Preview | Qwen 2.5 72B Instruct |
|---|---|---|
| `setup_new_course` | **1.00** ✅ (3 steps) | **1.00** ✅ (3 steps) |
| `grade_and_notify` | **1.00** ✅ (12 steps) | **0.86** ❌ (15 steps) |
| `end_of_semester` | **1.00** ✅ (18 steps) | **0.77** ❌ (17 steps) |
| **Average** | **1.00** | **0.88** |

Baseline was run with:
- `API_BASE_URL=https://router.huggingface.co/v1`
- `MODEL_NAME=gemini-3-flash-preview` / `qwen/qwen-2.5-72b-instruct`

---

## Setup & Usage

### Prerequisites

- Python 3.10+
- Docker
- `uv` (recommended) or `pip`

### Environment Variables

```bash
export API_BASE_URL="https://router.huggingface.co/v1"   # LLM API endpoint
export MODEL_NAME="gemini-3-flash-preview"            # Model identifier
export HF_TOKEN="hf_..."                              # Hugging Face / API key
```

### Install & Run Inference Locally

```bash
# Clone and enter the project
git clone https://huggingface.co/spaces/amydosomething/teacher-workspace-env
cd teacher-workspace-env

# Install dependencies
uv sync
# or: pip install -e .

# Run the baseline inference script
python inference.py
```

### Run with Docker

```bash
# Build the image
docker build -t teacher-workspace-env:latest .

# Run
docker run --rm \
  -e API_BASE_URL="https://router.huggingface.co/v1" \
  -e MODEL_NAME="gemini-3-flash-preview" \
  -e HF_TOKEN="hf_..." \
  -p 8000:8000 \
  teacher-workspace-env:latest
```

### Validate Submission

```bash
openenv validate
```

---

## Project Structure

```
teacher_workspace_env/
├── README.md                                        # This file
├── openenv.yaml                                     # OpenEnv manifest
├── Dockerfile                                       # Container image definition
├── pyproject.toml                                   # Project metadata and dependencies
├── inference.py                                     # Baseline inference script (required)
├── models.py                                        # TeacherAction & TeacherObservation models
├── client.py                                        # TeacherWorkspaceEnv client
└── server/
    ├── __init__.py
    ├── app.py                                       # FastAPI HTTP + WebSocket server
    ├── teacher_workspace_env_environment.py         # Core environment logic & graders
    └── requirements.txt
```

---

## Deployed Space

This environment is deployed at:  
`https://huggingface.co/spaces/amydosomething/teacher-workspace-env`

