"""
app.py
Gradio frontend for Teacher Workspace Env — runs on Hugging Face Spaces.

Place this file next to inference.py and run:
    pip install gradio
    python app.py

Set HF_TOKEN (and optionally MODEL_NAME, API_BASE_URL) as environment
variables or in a .env file — same as inference.py.
"""

import json
import os
import sys
import time
from typing import Generator

import gradio as gr

# ── Reuse everything from inference.py ───────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from inference import (
    SYSTEM_PROMPT,
    TASKS,
    MAX_STEPS,
    MAX_RETRIES,
    MAX_CONSECUTIVE_FAILS,
    READ_ONLY_TOOLS,
    API_BASE_URL,
    MODEL_NAME,
    HF_TOKEN,
    call_llm,
    parse_action,
    build_user_prompt,
)
from server.teacher_workspace_env_environment import TeacherWorkspaceEnvironment, TASKS as TASK_CONFIGS
from models import TeacherAction, TeacherObservation
from openai import OpenAI


# ── Generator version of run_task ─────────────────────────────────────────────
def run_task_stream(task_name: str) -> Generator[dict, None, None]:
    """
    Same logic as inference.run_task() but yields a state dict after
    every step so the Gradio UI can update in real time.

    Yielded dict keys:
        step, tool, params, reward, cumulative_reward, done, error,
        history, completed, obs (TeacherObservation), score, success
    """
    env    = TeacherWorkspaceEnvironment()
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    max_steps = MAX_STEPS[task_name]

    history:   list[str]   = []
    completed: list[str]   = []
    rewards:   list[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    obs = env.reset(task_name=task_name)

    consecutive_fails = 0
    consecutive_reads = 0

    for step in range(1, max_steps + 1):
        if obs.done:
            break

        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user",   "content": build_user_prompt(obs, step, history, completed)},
        ]

        try:
            raw = call_llm(client, messages)
        except Exception as e:
            err = f"LLM error: {e}"
            history.append(f"Step {step}: LLM call failed — {e}")
            consecutive_fails += 1
            yield {
                "step": step, "tool": "—", "params": {}, "reward": 0.0,
                "cumulative_reward": sum(rewards), "done": False, "error": err,
                "history": history, "completed": completed, "obs": obs,
                "score": score, "success": False,
            }
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                break
            continue

        action_dict = parse_action(raw)
        if action_dict is None:
            err = f"Parse failed: {raw[:120].replace(chr(10), ' ')}"
            history.append(f"Step {step}: parse error — output was not valid JSON")
            consecutive_fails += 1
            yield {
                "step": step, "tool": "—", "params": {}, "reward": 0.0,
                "cumulative_reward": sum(rewards), "done": False, "error": err,
                "history": history, "completed": completed, "obs": obs,
                "score": score, "success": False,
            }
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                break
            continue

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
            history.append(f"Step {step}: {action_dict['tool_name']} → exception: {err}")
            consecutive_fails += 1
            yield {
                "step": step, "tool": action_dict["tool_name"],
                "params": action_dict.get("params", {}), "reward": 0.0,
                "cumulative_reward": sum(rewards), "done": False, "error": err,
                "history": history, "completed": completed, "obs": obs,
                "score": score, "success": False,
            }
            if consecutive_fails >= MAX_CONSECUTIVE_FAILS:
                break
            continue

        tool = action_dict["tool_name"]
        if reward < 0 or not obs.success:
            consecutive_fails += 1
        else:
            consecutive_fails = 0

        if tool in READ_ONLY_TOOLS:
            consecutive_reads += 1
        else:
            consecutive_reads = 0

        rewards.append(reward)
        steps_taken = step

        params_str = json.dumps(action_dict.get("params", {}))
        result_str = json.dumps(obs.result)[:80] if obs.result else ""
        history.append(
            f"Step {step}: {tool}({params_str}) "
            f"→ reward={reward:+.2f} "
            f"{'✓ ' + result_str if obs.success else '✗ ' + str(obs.error)}"
        )

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

        if consecutive_reads >= 4:
            history.append("WARNING: 4 consecutive reads — must take a write action next")
            consecutive_reads = 0

        score = round(min(max(env.grade(), 0.01), 0.99), 2)

        yield {
            "step": step,
            "tool": tool,
            "params": action_dict.get("params", {}),
            "reward": reward,
            "cumulative_reward": sum(rewards),
            "done": done,
            "error": error,
            "history": history[:],
            "completed": completed[:],
            "obs": obs,
            "score": score,
            "success": score >= 0.99,
        }

        if done or consecutive_fails >= MAX_CONSECUTIVE_FAILS:
            break

    # Final grade
    score   = round(min(max(env.grade(), 0.01), 0.99), 2)
    success = score >= 0.99
    yield {
        "step": steps_taken, "tool": "DONE", "params": {}, "reward": 0.0,
        "cumulative_reward": sum(rewards), "done": True, "error": None,
        "history": history, "completed": completed, "obs": obs,
        "score": score, "success": success,
    }


# ── Gradio UI helpers ─────────────────────────────────────────────────────────

def _render_sheet(sheet: dict) -> str:
    """Render a single sheet as a full-width markdown table (no truncation)."""
    cells = sheet.get("cells", {})
    if not cells:
        return "_empty_"
    rows, cols = set(), set()
    for k in cells:
        if k and len(k) >= 2 and k[0].isalpha() and k[1:].isdigit():
            cols.add(k[0]); rows.add(int(k[1:]))
    if not rows:
        return "_empty_"
    sc = sorted(cols)
    lines = ["| " + " | ".join(sc) + " |", "|" + "---|" * len(sc)]
    for r in range(min(rows), max(rows) + 1):
        vals = [str(cells.get(f"{c}{r}", "")) for c in sc]
        lines.append("| " + " | ".join(vals) + " |")
    notes = sheet.get("notes", {})
    if notes:
        lines.append(f"\n**Notes:** {notes}")
    return "\n".join(lines)


def _render_sheets(sheets: dict) -> str:
    if not sheets:
        return "_No sheets yet._"
    parts = []
    for name, data in sheets.items():
        parts.append(f"### {name}")
        parts.append(_render_sheet(data))
        notes = data.get("notes", {})
        if notes:
            parts.append(f"**Notes:** {notes}")
    return "\n\n".join(parts)


def _render_inbox(inbox: list) -> str:
    if not inbox:
        return "_Inbox empty._"
    rows = ["| ID | From | Subject | Starred | Spam |",
            "|---|---|---|---|---|"]
    for m in inbox:
        starred = "⭐" if m.get("starred") else ""
        spam    = "🚫" if m.get("spam")    else ""
        rows.append(
            f"| `{m['mail_id']}` | {m.get('from_name','?')} "
            f"| {m.get('subject','')} | {starred} | {spam} |"
        )
    return "\n".join(rows)


def _render_sent(sent: list) -> str:
    if not sent:
        return "_No sent emails._"
    rows = ["| ID | To | Subject | Labels |",
            "|---|---|---|---|"]
    for m in sent[-15:]:  # last 15
        labels = ", ".join(m.get("labels", []))
        rows.append(
            f"| `{m['mail_id']}` | {m.get('to','')} "
            f"| {m.get('subject','')} | {labels} |"
        )
    return "\n".join(rows)


def _render_classrooms(classrooms: dict) -> str:
    if not classrooms:
        return "_No classrooms yet._"
    parts = []
    for cid, c in classrooms.items():
        anns = c.get("announcements", [])
        parts.append(
            f"**{c['name']}** (`{cid}`)  \n"
            f"Section: {c.get('section','')}  \n"
            f"Announcements: {len(anns)}"
        )
        for a in anns:
            parts.append(f"  - _{a['text']}_")
    return "\n\n".join(parts)


def _render_calendar(calendar: list) -> str:
    if not calendar:
        return "_No events._"
    rows = ["| Title | Date | Time | Meet |", "|---|---|---|---|"]
    for e in calendar:
        meet = "✅" if e.get("meet_link") else ""
        rows.append(f"| {e['title'][:35]} | {e['date']} | {e['time']} | {meet} |")
    return "\n".join(rows)






# ── Main Gradio streaming function ────────────────────────────────────────────

def run_ui(task_name: str):
    prompt_text = TASK_CONFIGS.get(task_name, {}).get("prompt", "")
    prompt_md   = prompt_text

    if not HF_TOKEN:
        yield (
            prompt_md,
            "❌ **HF_TOKEN not set.**", "", [], None, "", "", "", "", ""
        )
        return

    # Show prompt immediately before agent starts
    yield (
        f"### Task prompt\n\n```\n{prompt_text}\n```",
        "## Score: —",
        "_Nothing completed yet_",
        [], None, "", "", "", "", "",
    )

    step_rows   = []
    reward_data = []

    for state in run_task_stream(task_name):
        step   = state["step"]
        tool   = state["tool"]
        reward = state["reward"]
        cum    = state["cumulative_reward"]
        error  = state["error"] or ""
        obs    = state["obs"]
        score  = state["score"]
        done   = state["done"]

        status = "✓" if reward > 0 else ("✗" if reward < 0 else "→")
        step_rows.append([
            step,
            tool,
            json.dumps(state["params"]),
            f"{reward:+.2f}",
            status,
            error,
        ])

        reward_data.append([step, round(cum, 3)])
        reward_md = f"**Cumulative reward:** `{round(cum, 3)}`"

        completed_md = "\n".join(f"- ✅ {c}" for c in state["completed"]) or "_None yet_"
        emoji    = "🏆" if state["success"] else ("🔄" if not done else "⚠️")
        score_md = f"## {emoji} Score: {score:.2f}"

        sheets_md     = _render_sheets(obs.sheets)
        inbox_md      = _render_inbox(obs.inbox)
        sent_md       = _render_sent(obs.sent)
        classrooms_md = _render_classrooms(obs.classrooms)
        calendar_md   = _render_calendar(obs.calendar)

        yield (
            f"### Task prompt\n\n{prompt_text}",
            score_md,
            completed_md,
            step_rows,
            reward_md,
            sheets_md,
            inbox_md,
            sent_md,
            classrooms_md,
            calendar_md,
        )



# ── Layout ────────────────────────────────────────────────────────────────────

with gr.Blocks(title="Teacher Workspace Env — Agent Viewer") as demo:
    gr.Markdown("# 🎓 Teacher Workspace Env — Agent Live View")
    gr.Markdown(
        "Watch the LLM agent complete teacher admin tasks in real time. "
        "Select a task and click **Run**."
    )

    with gr.Row():
        task_dd  = gr.Dropdown(
            choices=TASKS,
            value=TASKS[0],
            label="Task",
            scale=3,
        )
        run_btn  = gr.Button("▶ Run", variant="primary", scale=1)
        stop_btn = gr.Button("⏹ Stop", variant="stop", scale=1)

    prompt_box = gr.Textbox(
        label="Task prompt",
        value="_Select a task and click Run to see the prompt._",
        lines=8,
        interactive=False,
    )

    with gr.Row():
        score_box = gr.Markdown("## Score: —")
        completed_box = gr.Markdown("_Completed sub-tasks will appear here_")

    gr.Markdown("---")

    # ── Step log + reward chart ───────────────────────────────────────────
    with gr.Row():
        with gr.Column(scale=3):
            gr.Markdown("### Step log")
            step_table = gr.Dataframe(
                headers=["Step", "Tool", "Params", "Reward", "Status", "Error"],
                datatype=["number", "str", "str", "str", "str", "str"],
                row_count=(1, "dynamic"),
                column_count=(6, "fixed"),
                wrap=True,
                max_height=420,
            )
        with gr.Column(scale=2):
            gr.Markdown("### Cumulative reward")
            reward_chart = gr.Markdown("_Reward will appear here_")

    gr.Markdown("---")
    gr.Markdown("## Workspace state")

    # ── Workspace panels ──────────────────────────────────────────────────
    with gr.Tabs():
        with gr.Tab("📊 Sheets"):
            sheets_panel = gr.Markdown("_Run a task to see gradebook data_")
        with gr.Tab("📥 Inbox"):
            inbox_panel = gr.Markdown("_Run a task to see inbox_")
        with gr.Tab("📤 Sent"):
            sent_panel = gr.Markdown("_Run a task to see sent mail_")
        with gr.Tab("🏫 Classrooms"):
            classrooms_panel = gr.Markdown("_Run a task to see classrooms_")
        with gr.Tab("📅 Calendar"):
            calendar_panel = gr.Markdown("_Run a task to see events_")

    # ── Wire up ───────────────────────────────────────────────────────────
    outputs = [
        prompt_box,
        score_box,
        completed_box,
        step_table,
        reward_chart,
        sheets_panel,
        inbox_panel,
        sent_panel,
        classrooms_panel,
        calendar_panel,
    ]

    run_event = run_btn.click(
        fn=run_ui,
        inputs=[task_dd],
        outputs=outputs,
    )
    stop_btn.click(fn=None, cancels=[run_event])


if __name__ == "__main__":
    demo.launch()