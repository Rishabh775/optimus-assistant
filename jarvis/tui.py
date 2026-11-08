"""Full-screen terminal UI for Optimus.

A thin frontend over the same Agent pipeline as the text/voice modes:

  - streaming answers (on_delta) rendered as Markdown
  - live route status
  - confirmation dialogs for consequential tool calls
  - proactive reminder notes from the background scheduler
  - slash commands (/help, /auto, /briefing, /reminders, /jobs, ...)

Run with: python -m jarvis --mode tui
"""
import json
import os
import threading
import time
from datetime import datetime

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, VerticalScroll
from textual.screen import ModalScreen
from textual.widgets import Button, Footer, Header, Input, Label, Markdown, Static

from . import config, scheduler

_HELP = """**Optimus TUI**

Type a question and press Enter. Responses stream in below your question.
Tool calls run quietly in the background unless they need confirmation.

**Slash commands**
- `/help` - this help
- `/auto [on|off]` - toggle confirmation mode
- `/status` - current config + pending reminders
- `/reminders` - list pending reminders
- `/briefing` - build the daily briefing now
- `/jobs` - job pipeline summary
- `/clear` - clear the transcript
- `/new` - start a fresh conversation (keeps durable memory)
- `/quit` - exit

**Keys**
- `Ctrl+Q` quit
- `Ctrl+L` clear transcript
- `Ctrl+N` new conversation
- `Ctrl+T` toggle confirmations
- `Ctrl+R` reminders
- `F1` help
"""

_TOOL_COLORS = {
    "note": "yellow",
    "error": "red",
    "reminder": "yellow",
}


class UserMessage(Static):
    def __init__(self, text: str):
        super().__init__(Text.assemble(("you\n", "bold green"), (text, "")))


class AssistantMessage(Markdown):
    def __init__(self, markdown: str = ""):
        super().__init__(markdown)


class EventMessage(Static):
    def __init__(self, text: str, kind: str = "note"):
        super().__init__(Text(text, style=_TOOL_COLORS.get(kind, "dim")))


class WelcomePanel(Static):
    def __init__(self, name: str):
        super().__init__(Text.from_markup(
            f"[bold cyan]{escape(name)}[/bold cyan] [dim]terminal UI[/dim]\n"
            "Ask anything, or type [bold]/help[/bold] for commands. "
            "Answers stream in below your question."))


class ConfirmScreen(ModalScreen[bool]):
    """Blocking tool-confirmation dialog."""

    BINDINGS = [
        Binding("y", "yes", "Yes"),
        Binding("n", "no", "No"),
        Binding("escape", "no", "No"),
    ]

    def __init__(self, reason: str, tool: str, args: dict,
                 mandatory: bool = False):
        super().__init__()
        self.reason = reason
        self.tool = tool
        self.args = args
        self.mandatory = mandatory

    def compose(self) -> ComposeResult:
        detail = json.dumps(self.args, ensure_ascii=False, default=str)
        if len(detail) > 600:
            detail = detail[:600] + " ..."
        with Container(id="confirm-dialog"):
            yield Label("Confirmation required", id="confirm-title")
            yield Static(Text(self.reason), id="confirm-reason")
            yield Static(Text(f"{self.tool} {detail}", style="dim"),
                         id="confirm-args")
            if self.mandatory:
                yield Label("This action always needs confirmation.",
                            id="confirm-mandatory")
            with Horizontal(id="confirm-buttons"):
                yield Button("Yes (y)", id="yes", variant="success")
                yield Button("No (n)", id="no", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class OptimusTUI(App):
    """Optimus chat in the terminal."""

    CSS = """
    Screen { layout: vertical; }

    Header { background: $primary-darken-2; }
    Footer { background: $panel; }

    #chat {
        height: 1fr;
        padding: 1 2;
        scrollbar-gutter: stable;
    }

    #status {
        height: 1;
        padding: 0 2;
        background: $panel;
        color: $text-muted;
    }

    #prompt { margin: 0 1 1 1; }

    WelcomePanel {
        border: round $accent;
        padding: 1 2;
        margin: 1 0 2 0;
    }

    UserMessage {
        border-left: thick $success;
        padding: 0 1;
        margin: 1 0 0 0;
        background: $boost;
    }

    AssistantMessage {
        border-left: thick $accent;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    EventMessage {
        padding: 0 1;
        margin: 0 0 0 2;
    }

    ConfirmScreen {
        align: center middle;
        background: $background 60%;
    }

    #confirm-dialog {
        width: 74;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }

    #confirm-title {
        text-style: bold;
        color: $warning;
        margin-bottom: 1;
    }

    #confirm-reason { margin-bottom: 1; }
    #confirm-args { margin-bottom: 1; }
    #confirm-mandatory { color: $warning; margin-bottom: 1; }

    #confirm-buttons {
        height: auto;
        align: right middle;
    }

    #confirm-buttons Button { margin-left: 1; }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+n", "new_session", "New"),
        Binding("ctrl+t", "toggle_auto", "Auto"),
        Binding("ctrl+r", "reminders", "Reminders"),
        Binding("f1", "help", "Help"),
    ]

    def __init__(self, agent=None):
        super().__init__()
        self._name = config.get("JARVIS_NAME")
        self.title = self._name
        self.sub_title = "terminal"
        self._provided_agent = agent
        self.agent = None
        self._busy = False
        self._route_label = ""
        self._stream_text = ""
        self._stream_widget = None
        self._delta_buf = []
        self._last_flush = 0.0

    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with VerticalScroll(id="chat"):
            yield WelcomePanel(self._name)
        yield Static("", id="status")
        yield Input(placeholder="Ask Optimus...  (/help for commands)",
                    id="prompt")
        yield Footer()

    def on_mount(self) -> None:
        if self._provided_agent is not None:
            self.agent = self._provided_agent
            self.agent.confirm = self._confirm
            self.agent.on_event = self._on_agent_event
        else:
            from .agent import Agent
            self.agent = Agent(confirm=self._confirm,
                               on_event=self._on_agent_event)
        scheduler.add_listener(self._on_reminder)
        self._set_status()
        self.query_one("#prompt", Input).focus()

    # ----------------------------------------------------------- helpers --

    def _set_status(self) -> None:
        mode = "auto" if config.get_bool("JARVIS_AUTO") else "ask"
        state = "thinking..." if self._busy else "ready"
        route = f" | {self._route_label}" if self._route_label else ""
        try:
            self.query_one("#status", Static).update(
                f"[dim]{escape(self._name)} | {state}{escape(route)} | "
                f"confirm {mode}[/dim]")
        except Exception:
            pass

    def _scroll_end(self) -> None:
        try:
            self.query_one("#chat", VerticalScroll).scroll_end(animate=False)
        except Exception:
            pass

    def _chat(self) -> VerticalScroll:
        return self.query_one("#chat", VerticalScroll)

    def _add_user(self, text: str) -> None:
        self._chat().mount(UserMessage(text))
        self.call_after_refresh(self._scroll_end)

    def _add_event(self, text: str, kind: str = "note") -> None:
        self._chat().mount(EventMessage(text, kind))
        self.call_after_refresh(self._scroll_end)

    def _add_assistant_text(self, markdown: str) -> None:
        self._chat().mount(AssistantMessage(markdown))
        self.call_after_refresh(self._scroll_end)

    def _ensure_stream_widget(self) -> AssistantMessage:
        if self._stream_widget is None:
            self._stream_widget = AssistantMessage(self._stream_text)
            self._chat().mount(self._stream_widget)
        return self._stream_widget

    def _write_stream(self, text: str) -> None:
        widget = self._ensure_stream_widget()
        if widget.is_mounted:
            widget.update(text)
        else:
            widget._initial_markdown = text

    def _reset_stream(self) -> None:
        self._stream_text = ""
        self._stream_widget = None
        self._delta_buf.clear()
        self._last_flush = time.monotonic()

    # ------------------------------------------------------ agent events --

    def _on_agent_event(self, event: dict) -> None:
        """Called from the agent worker thread."""
        try:
            self.call_from_thread(self._render_event, event)
        except Exception:
            pass

    def _render_event(self, event: dict) -> None:
        kind = event.get("type")
        if kind == "route":
            self._route_label = f"{event.get('provider')}/{event.get('model')}"
            self._set_status()

    def _on_delta(self, piece: str) -> None:
        """Streaming callback, invoked from the agent worker thread."""
        if not piece:
            return
        self._delta_buf.append(piece)
        now = time.monotonic()
        if now - self._last_flush >= 0.08:
            chunk = "".join(self._delta_buf)
            self._delta_buf.clear()
            self._last_flush = now
            try:
                self.call_from_thread(self._append_stream, chunk)
            except Exception:
                pass

    def _append_stream(self, chunk: str) -> None:
        self._stream_text += chunk
        self._write_stream(self._stream_text)
        self._scroll_end()

    # -------------------------------------------------------- run a turn --

    def _submit_question(self, text: str) -> None:
        self._add_user(text)
        self._reset_stream()
        self._route_label = ""
        self._busy = True
        self._set_status()
        self.run_worker(lambda: self._ask_blocking(text), thread=True,
                        exclusive=True, name="ask")

    def _ask_blocking(self, text: str) -> None:
        try:
            answer = self.agent.ask(text, on_delta=self._on_delta)
        except Exception as e:
            self.call_from_thread(self._finish_error, str(e))
            return
        self.call_from_thread(self._finish_answer, answer)

    def _finish_answer(self, answer: str) -> None:
        tail = "".join(self._delta_buf)
        self._delta_buf.clear()
        if tail:
            self._stream_text += tail
        final = (answer or self._stream_text or "(no response)").strip()
        self._stream_text = final
        self._write_stream(final)
        self._stream_widget = None
        self._busy = False
        self._set_status()
        self.call_after_refresh(self._scroll_end)
        try:
            self.query_one("#prompt", Input).focus()
        except Exception:
            pass

    def _finish_error(self, message: str) -> None:
        self._add_event(f"error: {message}", "error")
        self._busy = False
        self._set_status()
        try:
            self.query_one("#prompt", Input).focus()
        except Exception:
            pass

    # ------------------------------------------------------- confirmation --

    def _confirm(self, reason: str, fn_name: str, args: dict,
                 mandatory: bool = False) -> bool:
        """Blocking hook used by Agent from the worker thread."""
        done = threading.Event()
        box = {}

        def show() -> None:
            def callback(result) -> None:
                box["result"] = bool(result)
                done.set()
            try:
                self.push_screen(
                    ConfirmScreen(reason, fn_name, args, mandatory), callback)
            except Exception:
                done.set()

        try:
            self.call_from_thread(show)
        except Exception:
            return False
        timeout = max(5, config.get_int("JARVIS_APP_CONFIRM_TIMEOUT"))
        if not done.wait(timeout):
            try:
                self.call_from_thread(self._force_close_confirm)
            except Exception:
                pass
            return False
        return box.get("result", False)

    def _force_close_confirm(self) -> None:
        if isinstance(self.screen, ConfirmScreen):
            self.screen.dismiss(False)

    # ---------------------------------------------------------- reminders --

    def _on_reminder(self, reminder: dict) -> None:
        mark = "Missed reminder" if reminder.get("missed") else "Reminder"
        text = f"{mark}: {reminder.get('message', '')}"
        try:
            self.call_from_thread(self._render_reminder, text)
        except Exception:
            pass

    def _render_reminder(self, text: str) -> None:
        self._add_event(text, "reminder")
        try:
            self.notify(text, severity="warning")
        except Exception:
            pass

    # ---------------------------------------------------------- commands --

    def on_input_submitted(self, event: Input.Submitted) -> None:
        text = event.value.strip()
        event.input.value = ""
        if not text:
            return
        if text.startswith("/"):
            self._handle_command(text)
            return
        if self._busy:
            self.notify("Still working on the previous request.",
                        severity="warning")
            return
        self._submit_question(text)

    def _handle_command(self, text: str) -> None:
        cmd, _, rest = text.partition(" ")
        cmd = cmd.lower()
        rest = rest.strip()

        if cmd in ("/quit", "/exit", "/q"):
            self.exit()
        elif cmd == "/help":
            self._add_assistant_text(_HELP)
        elif cmd == "/clear":
            if self._busy:
                self.notify("Wait for the current answer first.",
                            severity="warning")
            else:
                self._clear_chat()
        elif cmd == "/new":
            if self._busy:
                self.notify("Wait for the current answer first.",
                            severity="warning")
            else:
                self._reset_conversation()
        elif cmd == "/auto":
            self._handle_auto(rest)
        elif cmd == "/status":
            self._show_status()
        elif cmd == "/reminders":
            self._show_reminders()
        elif cmd == "/briefing":
            self._start_briefing()
        elif cmd == "/jobs":
            self._show_jobs()
        else:
            self._add_event(f"unknown command {cmd} - try /help", "error")

    def _handle_auto(self, value: str) -> None:
        value = value.lower()
        if value in ("on", "1", "true", "yes"):
            enabled = True
        elif value in ("off", "0", "false", "no"):
            enabled = False
        else:
            enabled = not config.get_bool("JARVIS_AUTO")
        os.environ["JARVIS_AUTO"] = "1" if enabled else "0"
        self._set_status()
        self._add_event(
            f"confirmation mode -> {'auto' if enabled else 'ask'}", "note")

    def _clear_chat(self) -> None:
        chat = self._chat()
        chat.remove_children()
        self._stream_widget = None
        chat.mount(WelcomePanel(self._name))
        self.call_after_refresh(self._scroll_end)

    def _reset_conversation(self) -> None:
        try:
            self.agent.history = self.agent.history[:1]
        except Exception:
            pass
        self._route_label = ""
        self._clear_chat()
        self._add_event("started a new conversation", "note")

    def _show_status(self) -> None:
        mode = "auto" if config.get_bool("JARVIS_AUTO") else "ask"
        try:
            pending = len(scheduler.list_pending())
        except Exception:
            pending = 0
        mem = ""
        try:
            mem = f"{self.agent.memory.count()} entries"
        except Exception:
            pass
        lines = [
            f"**{self._name}**",
            "",
            f"- confirmations: **{mode}**",
            f"- pending reminders: **{pending}**",
            f"- conversation memory: {mem or 'n/a'}",
        ]
        self._add_assistant_text("\n".join(lines))

    def _show_reminders(self) -> None:
        try:
            rows = scheduler.list_pending()
        except Exception as e:
            self._add_event(f"could not read reminders: {e}", "error")
            return
        if not rows:
            self._add_assistant_text("No pending reminders.")
            return
        lines = ["**Pending reminders**", ""]
        for r in rows:
            when = datetime.fromtimestamp(r["due_at"]).strftime(
                "%a %d %b, %H:%M")
            repeat = f" ({r['repeat']})" if r.get("repeat") else ""
            lines.append(f"- `{r['id']}` {r['message']} - {when}{repeat}")
        self._add_assistant_text("\n".join(lines))

    def _start_briefing(self) -> None:
        if self._busy:
            self.notify("Wait for the current answer first.",
                        severity="warning")
            return
        self._busy = True
        self._set_status()
        self._add_event("building daily briefing...", "note")
        self.run_worker(self._briefing_blocking, thread=True,
                        exclusive=True, name="briefing")

    def _briefing_blocking(self) -> None:
        try:
            from .briefing import build
            text = build()
        except Exception as e:
            text = f"Briefing failed: {e}"
        self.call_from_thread(self._finish_command_text, text)

    def _show_jobs(self) -> None:
        try:
            from . import jobs
            stats = jobs.tracker.stats()
            rows = jobs.tracker.list_jobs(limit=10)
        except Exception as e:
            self._add_event(f"could not read jobs: {e}", "error")
            return
        lines = [f"**Jobs** - {stats['total']} tracked", ""]
        for status, count in stats["by_status"].items():
            if count:
                lines.append(f"- {status}: {count}")
        if rows:
            lines.extend(["", "**Top matches**", ""])
            for job in rows:
                title = job.get("title") or "(untitled)"
                company = job.get("company") or "?"
                score = job.get("score")
                score_txt = f"{score:.0f}" if isinstance(score, (int, float)) else "?"
                lines.append(f"- [{score_txt}] {title} - {company}")
        else:
            lines.append("")
            lines.append("No jobs tracked yet. Ask the agent to run a sweep.")
        self._add_assistant_text("\n".join(lines))

    def _finish_command_text(self, text: str) -> None:
        self._add_assistant_text(text or "(empty)")
        self._busy = False
        self._set_status()
        try:
            self.query_one("#prompt", Input).focus()
        except Exception:
            pass

    # ----------------------------------------------------------- actions --

    def action_quit(self) -> None:
        self.exit()

    def action_clear(self) -> None:
        self._handle_command("/clear")

    def action_new_session(self) -> None:
        self._handle_command("/new")

    def action_toggle_auto(self) -> None:
        self._handle_command("/auto")

    def action_reminders(self) -> None:
        self._handle_command("/reminders")

    def action_help(self) -> None:
        self._handle_command("/help")


def run_tui(agent=None) -> None:
    OptimusTUI(agent=agent).run()
