#!/usr/bin/env python3
"""
inference.py
Inference Script for Teacher Workspace Environment
Run from inside teacher_workspace_env folder:
    cd teacher_workspace_env
    python inference.py
"""

import json
import os
import re
import sys
import time
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# ── Environment variables ───────────────────────────────────────────────────
API_BASE_URL     = os.getenv("API_BASE_URL", "https://router.huggingface.co/v1")
MODEL_NAME       = os.getenv("MODEL_NAME", "qwen/qwen-2.5-72b-instruct")
HF_TOKEN         = os.getenv("HF_TOKEN")

if HF_TOKEN is None:
    raise ValueError("HF_TOKEN environment variable is required")

# ── Import environment directly (no HTTP client needed) ────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from server.teacher_workspace_env_environment import TeacherWorkspaceEnvironment
from models import TeacherAction, TeacherObservation

# ── Task config ─────────────────────────────────────────────────────────────
TASKS      = ["setup_new_course", "grade_and_notify", "end_of_semester"]
MAX_STEPS  = {"setup_new_course": 20, "grade_and_notify": 30, "end_of_semester": 40}
BENCHMARK  = "teacher_workspace_env"

# ── Retry / safety config ────────────────────────────────────────────────────
MAX_RETRIES           = 5
BASE_DELAY            = 1.0
MAX_DELAY             = 30.0
MAX_CONSECUTIVE_FAILS = 5    
READ_ONLY_TOOLS = {
    "list_inbox", "list_sheets", "list_classrooms", "list_events",
    "get_cells", "read_mail", "search_mail", "get_event",
    "get_classroom", "list_announcements", "filter_range",
}


# ══════════════════════════════════════════════════════════════════════════════
# LOGGING  — mandatory stdout format
# ══════════════════════════════════════════════════════════════════════════════

def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)

def log_step(step: int, action: Optional[Dict], reward: float,
             done: bool, error: Optional[str]) -> None:
    action_str = json.dumps(action) if action else "null"
    error_str  = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} "
        f"reward={reward:.2f} done={str(done).lower()} error={error_str}",
        flush=True,
    )

def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# LLM CALL  — synchronous with exponential-backoff retry
# ══════════════════════════════════════════════════════════════════════════════

def call_llm(client: OpenAI, messages: List[Dict]) -> str:
    delay = BASE_DELAY
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=messages,
                temperature=0.1,
                max_tokens=1024,   # one action at a time — 512 is plenty
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            err = str(e).lower()
            is_transient = any(x in err for x in
                               ["rate limit", "429", "503", "502", "timeout"])
            if is_transient and attempt < MAX_RETRIES:
                time.sleep(min(delay, MAX_DELAY))
                delay *= 2
                continue
            raise
    raise RuntimeError("Exceeded max retries")


# ══════════════════════════════════════════════════════════════════════════════
# JSON PARSING  — robust extraction from LLM output
# ══════════════════════════════════════════════════════════════════════════════

def parse_action(text: str) -> Optional[Dict]:
    """
    Extract the first valid {"tool_name": ..., "params": ...} from text.
    Handles markdown fences, extra prose, and nested braces gracefully.
    """
    # Strip markdown fences
    text = re.sub(r"```(?:json)?", "", text).strip()
    text = text.replace("```", "").strip()

    # 1. Direct parse
    try:
        data = json.loads(text)
        if "tool_name" in data:
            return {"tool_name": data["tool_name"], "params": data.get("params", {})}
    except json.JSONDecodeError:
        pass

    # 2. Find outermost { ... } and parse
    start = text.find("{")
    end   = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            data = json.loads(text[start:end + 1])
            if "tool_name" in data:
                return {"tool_name": data["tool_name"], "params": data.get("params", {})}
        except json.JSONDecodeError:
            pass

    # 3. Find any {...} block containing tool_name
    for match in re.finditer(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)?\}", text, re.DOTALL):
        try:
            data = json.loads(match.group())
            if "tool_name" in data:
                return {"tool_name": data["tool_name"], "params": data.get("params", {})}
        except json.JSONDecodeError:
            continue

    return None


# ══════════════════════════════════════════════════════════════════════════════
# PROMPT HELPERS
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = """You are a teacher assistant controlling Google Workspace tools.

OUTPUT RULES — non-negotiable:
- Output EXACTLY ONE JSON object per response, nothing else.
- Format: {"tool_name": "<name>", "params": {<key>: <value>}}
- No markdown, no explanation, no prose before or after the JSON.
- ONE action per response. Never output multiple actions.

CALCULATION RULES:
- AVERAGE of n values = (n1 + n2 + n3 + nn ) / n, rounded to exactly 2 decimal places.
- Always compute step by step: first sum, then divide.

STOPPING RULES:
- Only perform actions explicitly required by the task prompt.
- Never create labels, categories, or send emails or do any action not mentioned in the task.
- Never call the same tool unecessarily if it doesn't give any output or is not going to be used for other actions.
- Once all sub-tasks are complete, do not take any further actions.
- Check COMPLETED SUB-TASKS before acting — never repeat a completed step.

AVAILABLE TOOLS:

Google Classroom (read):
  list_classrooms     {}
  get_classroom       {"class_id": str}
  list_announcements  {"class_id": str}

Google Classroom (write):
  create_classroom    {"name": str, "section": str, "description": str}
  create_announcement {"class_id": str, "text": str}

Google Sheets (read):
  list_sheets         {}
  get_cells           {"sheet_name": str}

Google Sheets (write):
  create_sheet        {"sheet_name": str, "headers": [str, ...]}
  update_cell         {"sheet_name": str, "cell": str, "value": any}
  add_note            {"sheet_name": str, "cell": str, "note": str}
  set_formula         {"sheet_name": str, "cell": str, "formula": str}
  sort_range          {"sheet_name": str, "column": str, "ascending": bool}
  filter_range        {"sheet_name": str, "column": str, "operator": str, "value": any}

Gmail (read):
  list_inbox          {}
  read_mail           {"mail_id": str}
  search_mail         {"query": str}

Gmail (write):
  send_mail           {"to": str, "subject": str, "body": str}
  star_mail           {"mail_id": str}
  mark_spam           {"mail_id": str}
  mark_important      {"mail_id": str}
  create_label        {"name": str}
  assign_label        {"mail_id": str, "label": str}
  categorise_mail     {"mail_id": str, "category": str}

Calendar / Meet (read):
  list_events         {}

Calendar / Meet (write):
  create_meet_event   {"title": str, "date": str, "time": str, "participants": [str], "description": str}
  create_event        {"title": str, "date": str, "time": str, "participants": [str], "description": str}

BEHAVIOUR RULES:
1. Complete sub-tasks in order, one action at a time.
2. Never repeat a successful action (reward > 0 means it worked).
3. Use EXACT email addresses from the students/parents lists — never invent emails.
4. set_formula format: AVERAGE(C2,D2,E2) — uppercase, comma-separated, no spaces.
5. assign_label needs a mail_id from the SENT list, not inbox. Check sent emails.
6. create_label must be called BEFORE assign_label for the same label name.
7. For failing student Meet events: use create_meet_event, include parent email in participants.
8. If an action errors, read the error carefully and fix your params before retrying.
"""


def format_sheet(sheet: Dict) -> str:
    cells = sheet.get("cells", {})
    if not cells:
        return "  (empty)\n"
    rows, cols = set(), set()
    for key in cells:
        if key and len(key) >= 2 and key[0].isalpha() and key[1:].isdigit():
            cols.add(key[0])
            rows.add(int(key[1:]))
    if not rows:
        return "  (empty)\n"
    sorted_cols = sorted(cols)
    min_row, max_row = min(rows), max(rows)
    lines = ["  " + " | ".join(f"{c:<14}" for c in sorted_cols)]
    lines.append("  " + "-" * (16 * len(sorted_cols)))
    for r in range(min_row, max_row + 1):
        vals = [str(cells.get(f"{c}{r}", ""))[:14] for c in sorted_cols]
        lines.append(f"  {r}: " + " | ".join(f"{v:<14}" for v in vals))
    return "\n".join(lines) + "\n"


def build_user_prompt(obs: TeacherObservation, step: int,
                      history: List[str], completed: List[str]) -> str:

    # Gradebook data
    sheets_str = ""
    for name, data in obs.sheets.items():
        sheets_str += f"\n--- {name} ---\n"
        sheets_str += format_sheet(data)
        notes = data.get("notes", {})
        if notes:
            sheets_str += f"  Notes: {notes}\n"
        formulas = data.get("formulas", {})
        if formulas:
            sheets_str += f"  Formulas set: {list(formulas.keys())}\n"

    # Sent emails — critical for assign_label
    sent_str = ""
    if obs.sent:
        sent_str = "\nSENT EMAILS (use these mail_ids for assign_label):\n"
        for m in obs.sent[-20:]:
            sent_str += (
                f"  mail_id={m['mail_id']}  to={m['to']}"
                f"  subject={m.get('subject','')[:40]}"
                f"  labels={m.get('labels', [])}\n"
            )

    # Inbox
    inbox_str = ""
    for m in obs.inbox:
        inbox_str += (
            f"  {m['mail_id']}  from={m.get('from_name','?')}"
            f"  starred={m.get('starred')}  spam={m.get('spam')}\n"
        )

    # Students and parents
    students_str = "\n".join(
        f"  name={s['name']}  email={s['email']}  id={s['id']}"
        for s in obs.students
    )
    parents_str = "\n".join(
        f"  name={p['name']}  email={p['email']}  student_id={p['student_id']}"
        for p in obs.parents
    )

    # Classrooms
    cls_str = ""
    for cid, c in obs.classrooms.items():
        ann_count = len(c.get("announcements", []))
        cls_str += f"  {cid}: {c['name']}  announcements={ann_count}\n"

    meet_str = ""
    for evt in obs.calendar:
        if evt.get("meet_link"):
            meet_str += (
                f"  MEET CREATED: '{evt['title']}' on {evt['date']} at {evt['time']} "
                f"participants={evt['participants']}\n"
            )

    # Last action feedback
    feedback = ""
    if obs.error:
        feedback = f"\n⚠ LAST ACTION FAILED: {obs.error}\n  Fix your params and try again.\n"
    elif obs.result:
        feedback = f"\n✓ Last result: {json.dumps(obs.result)[:200]}\n"

    return f"""TASK:
{obs.task_prompt}

STEP {step} — output ONE JSON action.

COMPLETED SUB-TASKS:
{chr(10).join(f'  ✓ {c}' for c in completed) if completed else '  (nothing yet)'}

RECENT HISTORY (last 40 steps):
{chr(10).join(history[-40:]) if history else '  (none)'}
{feedback}
CLASSROOMS:
{cls_str or '  (none)'}

GOOGLE MEET EVENTS ALREADY CREATED (do NOT recreate these):
{meet_str or '  (none)'}

GRADEBOOK DATA:
{sheets_str}

INBOX:
{inbox_str or '  (empty)'}
{sent_str}
STUDENTS (exact emails only):
{students_str}

PARENTS (exact emails only):
{parents_str}

LABELS: {obs.labels}

Respond with ONE JSON object:"""


# ══════════════════════════════════════════════════════════════════════════════
# TASK RUNNER
# ══════════════════════════════════════════════════════════════════════════════

def run_task(task_name: str) -> Dict:
    env       = TeacherWorkspaceEnvironment()
    client    = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    max_steps = MAX_STEPS[task_name]

    history:   List[str]   = []
    completed: List[str]   = []
    rewards:   List[float] = []
    steps_taken = 0
    score       = 0.0
    success     = False

    log_start(task=task_name, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset(task_name=task_name)

        consecutive_fails = 0
        consecutive_reads = 0

        for step in range(1, max_steps + 1):
            if obs.done:
                break

            # ── LLM call ──────────────────────────────────────────────────
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": build_user_prompt(
                    obs, step, history, completed)},
            ]

            try:
                raw = call_llm(client, messages)
            except Exception as e:
                err = f"LLM error: {e}"
                log_step(step, None, 0.0, False, err)
                history.append(f"Step {step}: LLM call failed — {e}")
                consecutive_fails += 1
                if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                    break
                continue

            # ── Parse ─────────────────────────────────────────────────────
            action_dict = parse_action(raw)
            if action_dict is None:
                snippet = raw[:120].replace("\n", " ")
                err = f"Parse failed: {snippet}"
                log_step(step, None, 0.0, False, err)
                history.append(f"Step {step}: parse error — output was not valid JSON")
                consecutive_fails += 1
                if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                    break
                continue

            # ── Execute ───────────────────────────────────────────────────
            try:
                action = TeacherAction(
                    tool_name=action_dict["tool_name"],
                    params=action_dict.get("params", {}),
                )
                obs    = env.step(action)
                reward = obs.reward
                done   = obs.done
                error  = obs.error if not obs.success else None
            except Exception as e:
                err = str(e)
                log_step(step, action_dict, 0.0, False, err)
                history.append(
                    f"Step {step}: {action_dict['tool_name']} → exception: {err}")
                consecutive_fails += 1
                if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                    break
                continue

            # ── Update counters ───────────────────────────────────────────
            tool = action_dict["tool_name"]

            if reward < 0 or not obs.success:
                consecutive_fails += 1
            else:
                consecutive_fails = 0   # reset on any clean success

            if tool in READ_ONLY_TOOLS:
                consecutive_reads += 1
            else:
                consecutive_reads = 0

            rewards.append(reward)
            steps_taken = step

            # ── History entry ─────────────────────────────────────────────
            params_str = json.dumps(action_dict.get("params", {}))
            result_str = json.dumps(obs.result)[:80] if obs.result else ""
            history.append(
                f"Step {step}: {tool}({params_str}) "
                f"→ reward={reward:+.2f} "
                f"{'✓ ' + result_str if obs.success else '✗ ' + str(obs.error)}"
            )

            # ── Completed tracking ────────────────────────────────────────
            if reward > 0:
                p = action_dict.get("params", {})
                tag = {
                    "create_classroom":    f"Created classroom '{p.get('name')}'",
                    "create_sheet":        f"Created sheet '{p.get('sheet_name')}'",
                    "create_announcement": f"Posted announcement in {p.get('class_id')}",
                    "update_cell":         f"Updated {p.get('cell')} in '{p.get('sheet_name')}'",
                    "set_formula":         f"Formula set in {p.get('cell')}",
                    "add_note":            f"Note added to {p.get('cell')}",
                    "sort_range":          f"Sorted '{p.get('sheet_name')}' by {p.get('column')}",
                    "send_mail":           f"Email sent to {p.get('to')}",
                    "star_mail":           f"Starred {p.get('mail_id')}",
                    "mark_spam":           f"Marked spam {p.get('mail_id')}",
                    "create_label":        f"Created label '{p.get('name')}'",
                    "assign_label":        f"Label '{p.get('label')}' → {p.get('mail_id')}",
                    "create_meet_event":   f"Meet event '{p.get('title')}'",
                    "create_event":        f"Event '{p.get('title')}'",
                }.get(tool)
                if tag and tag not in completed:
                    completed.append(tag)

            log_step(step, action_dict, reward, done, error)

            if done:
                break

            # Safety guards
            if consecutive_reads >= 4:
                history.append(
                    "WARNING: 4 consecutive reads — must take a write action next")
                consecutive_reads = 0

            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                break

        # ── Score from grader (exact holistic check) ──────────────────────
        score   = round(min(max(env.grade(), 0.0), 1.0), 2)
        success = score >= 1.0

    except Exception as e:
        print(f"[ERROR] Task '{task_name}' crashed: {e}", flush=True)
        import traceback
        traceback.print_exc()
        score   = 0.0
        success = False

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

    return {"task": task_name, "score": score, "steps": steps_taken, "success": success}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    results = []
    for task in TASKS:
        result = run_task(task)
        results.append(result)
        time.sleep(1)

    print("\n" + "=" * 50, flush=True)
    print("BASELINE RESULTS", flush=True)
    print("=" * 50, flush=True)
    for r in results:
        print(
            f"{r['task']:<25}  score={r['score']:.2f}  "
            f"steps={r['steps']}  success={r['success']}",
            flush=True,
        )
    avg = sum(r["score"] for r in results) / len(results) if results else 0.0
    print(f"\nAverage score: {avg:.2f}", flush=True)


if __name__ == "__main__":
    main()