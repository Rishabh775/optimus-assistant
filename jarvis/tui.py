"""Soundwave terminal UI (Textual).

A voice-first frontend over the same Agent pipeline as the other modes.
Design: a quiet "instrument console" — near-black surfaces, one teal
accent, a channel rail on the left, and streaming answers as markdown.

Run with: python -m jarvis.tui_main  (or python -m jarvis --mode tui)
"""
import json
import logging
import os
import threading
import time
from datetime import datetime

from rich.markup import escape
from rich.text import Text
from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Container, Horizontal, Vertical, VerticalScroll
from textual.screen import ModalScreen
from textual.theme import Theme
from textual.widgets import Button, Input, Markdown, Static

from . import config, scheduler

log = logging.getLogger("soundwave.tui")

# Brand palette — matches the phone app (teal on near-black), not the
# stock blue/purple TUI defaults.
_SOUNDWAVE_THEME = Theme(
    name="soundwave",
    primary="#35d0ba",
    secondary="#4aa8ff",
    accent="#35d0ba",
    foreground="#e8ecf4",
    background="#0f1115",
    surface="#151922",
    panel="#1b2230",
    dark=True,
    variables={
        "text-muted": "#8b94a7",
        "warning": "#ffb454",
        "error": "#ff6b6b",
        "success": "#35d0ba",
        "boost": "#1c2230",
        "line": "#232b3a",
    },
)

_HELP = """**Soundwave**

Type a message and press Enter. Answers stream below your question. Tool
calls run quietly in the background unless they need approval.

**Commands**
- `/help` — this guide
- `/briefing` — the daily briefing now
- `/jobs` — job pipeline summary
- `/reminders` — pending reminders
- `/status` — configuration and memory
- `/auto [on|off]` — toggle confirmations
- `/voice [on|off]` — spoken replies for typed messages
- `/wake [on|off]` — the wake-word voice loop
- `/new` — fresh conversation · `/clear` — clear the transcript
- `/quit` — exit

**Keys**
`Enter` send · `F2` push-to-talk · `F3` voice · `F4` wake · `Esc` stop speech
`Ctrl+Q` quit · `Ctrl+L` clear · `Ctrl+N` new · `Ctrl+T` confirmations
`Ctrl+R` reminders · `F1` help
"""

_EVENT_STYLES = {
    "note": "dim",
    "tool": "#4aa8ff",
    "error": "#ff6b6b",
    "reminder": "#ffb454",
}


class UserMessage(Static):
    def __init__(self, text: str):
        super().__init__(Text.assemble(("you\n", "bold #4aa8ff"), (text, "")))


class AssistantMessage(Markdown):
    def __init__(self, markdown: str = ""):
        super().__init__(markdown)


class EventMessage(Static):
    def __init__(self, text: str, kind: str = "note"):
        super().__init__(Text(text, style=_EVENT_STYLES.get(kind, "dim")))


class WelcomePanel(Static):
    def __init__(self, name: str):
        super().__init__(Text.from_markup(
            f"[bold #35d0ba]{escape(name)}[/bold #35d0ba] "
            f"[dim]— your local assistant[/dim]\n\n"
            "Ask anything. Answers stream in below.\n"
            "Try [bold]/help[/bold] for commands, or say the wake word."))


class ConfirmScreen(ModalScreen[bool]):
    """Blocking approval dialog for consequential tool calls."""

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
        if len(detail) > 500:
            detail = detail[:500] + " …"
        with Container(id="confirm-dialog"):
            yield Static("Approval required", id="confirm-title")
            yield Static(self.reason, id="confirm-reason")
            yield Static(f"{self.tool}  {detail}", id="confirm-args")
            if self.mandatory:
                yield Static("This action always needs approval.",
                             id="confirm-mandatory")
            with Horizontal(id="confirm-buttons"):
                yield Button("Approve", id="yes", variant="primary")
                yield Button("Deny", id="no", variant="error")

    def on_button_pressed(self, event: Button.Pressed) -> None:
        self.dismiss(event.button.id == "yes")

    def action_yes(self) -> None:
        self.dismiss(True)

    def action_no(self) -> None:
        self.dismiss(False)


class OptimusTUI(App):
    """Soundwave's terminal frontend."""

    CSS = """
    Screen { layout: vertical; }

    #topbar {
        height: 1;
        background: $surface;
        color: $text-muted;
        padding: 0 1;
    }
    #brand { color: $accent; text-style: bold; }
    #status { width: 1fr; content-align: right middle; }

    #body { height: 1fr; }

    #rail {
        width: 22;
        background: $surface;
        border-right: solid $line;
        padding: 1;
    }
    #rail-title { color: $text-muted; margin: 0 0 1 1; }
    #rail Button {
        width: 100%;
        margin-bottom: 1;
        text-align: left;
        background: $boost;
    }
    #rail Button.-primary { background: $accent; color: #06231f; }
    #rail-rule { height: 1; border-bottom: solid $line; margin: 1 0; }
    #rail-meta { color: $text-muted; margin-top: 1; }

    #main { width: 1fr; padding: 0 1 0 2; }
    #chat {
        height: 1fr;
        padding: 1 2;
        scrollbar-gutter: stable;
    }

    WelcomePanel {
        border: round $accent;
        padding: 1 2;
        margin: 0 0 1 0;
    }

    UserMessage {
        align: right top;
        border-right: thick $secondary;
        padding: 0 1 0 4;
        margin: 1 0 1 4;
    }

    AssistantMessage {
        border-left: thick $accent;
        padding: 0 1;
        margin: 0 0 1 0;
    }

    EventMessage {
        padding: 0 1;
        margin: 0 0 0 2;
        color: $text-muted;
    }

    ConfirmScreen { align: center middle; background: $background 65%; }

    #confirm-dialog {
        width: 72;
        height: auto;
        border: thick $warning;
        background: $surface;
        padding: 1 2;
    }
    #confirm-title { text-style: bold; color: $warning; margin-bottom: 1; }
    #confirm-reason { margin-bottom: 1; }
    #confirm-args { color: $text-muted; margin-bottom: 1; }
    #confirm-mandatory { color: $warning; margin-bottom: 1; }
    #confirm-buttons { height: auto; align: right middle; }
    #confirm-buttons Button { margin-left: 1; }

    #prompt { margin: 0 1; }
    #hints {
        height: 1;
        color: $text-muted;
        padding: 0 1 1 1;
    }
    """

    BINDINGS = [
        Binding("ctrl+q", "quit", "Quit"),
        Binding("ctrl+l", "clear", "Clear"),
        Binding("ctrl+n", "new_session", "New"),
        Binding("ctrl+t", "toggle_auto", "Confirmations"),
        Binding("ctrl+r", "reminders", "Reminders"),
        Binding("f1", "help", "Help"),
        Binding("f2", "talk", "Talk"),
        Binding("f3", "toggle_voice", "Voice"),
        Binding("f4", "toggle_wake", "Wake"),
        Binding("escape", "stop_speech", "Stop speech"),
    ]

    def __init__(self, agent=None):
        super().__init__()
        self.register_theme(_SOUNDWAVE_THEME)
        self.theme = "soundwave"
        self._name = config.get("JARVIS_NAME")
        self.title = self._name
        self.sub_title = "terminal"
        self._provided_agent = agent
        self.agent = None
        self._busy = False
        self._route_label = ""
        self._stream_text = ""
        self._stream_widget = None
        self._voice_on = config.get_bool("JARVIS_TUI_VOICE")
        self._listening = False
        self._voice = None
        self._voice_turn = False
        self._wake_on = False
        self._wake_state = "off"
        self._wake_thread = None
        self._delta_buf = []
        self._last_flush = 0.0

    # ------------------------------------------------------------ compose --

    def compose(self) -> ComposeResult:
        with Horizontal(id="topbar"):
            yield Static("[bold]◉ Soundwave[/bold]", id="brand")
            yield Static("", id="status")
        with Horizontal(id="body"):
            with Vertical(id="rail"):
                yield Static("Channels", id="rail-title")
                yield Button("Briefing", id="btn-brief", variant="default")
                yield Button("Jobs", id="btn-jobs", variant="default")
                yield Button("Reminders", id="btn-reminders", variant="default")
                yield Static("", id="rail-rule")
                yield Button("Voice: off", id="btn-voice", variant="default")
                yield Button("Wake: off", id="btn-wake", variant="default")
                yield Button("New conversation", id="btn-new",
                             variant="primary")
                yield Static("", id="rail-meta")
            with Vertical(id="main"):
                with VerticalScroll(id="chat"):
                    yield WelcomePanel(self._name)
        yield Input(placeholder="Ask Soundwave…", id="prompt")
        yield Static("Enter send · F2 talk · F4 wake · F3 voice · Esc stop "
                     "· Ctrl+Q quit · /help", id="hints")

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
        if config.get_bool("JARVIS_TUI_WAKE"):
            self.action_toggle_wake()
        self.query_one("#prompt", Input).focus()

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-brief":
            self._handle_command("/briefing")
        elif bid == "btn-jobs":
            self._handle_command("/jobs")
        elif bid == "btn-reminders":
            self._handle_command("/reminders")
        elif bid == "btn-voice":
            self.action_toggle_voice()
        elif bid == "btn-wake":
            self.action_toggle_wake()
        elif bid == "btn-new":
            self._handle_command("/new")

    # ----------------------------------------------------------- helpers --

    def _set_status(self) -> None:
        mode = "auto" if config.get_bool("JARVIS_AUTO") else "ask"
        state = ("listening…" if self._listening
                 else "thinking…" if self._busy else "ready")
        parts = [state]
        if self._route_label:
            parts.append(self._route_label)
        parts.append(f"confirm {mode}")
        if self._wake_on:
            parts.append(f"wake {self._wake_state}")
        parts.append(f"voice {'on' if self._voice_on else 'off'}")
        try:
            self.query_one("#status", Static).update(" · ".join(parts))
        except Exception:
            pass
        self._refresh_buttons()

    def _refresh_buttons(self) -> None:
        labels = {
            "btn-voice": f"Voice: {'on' if self._voice_on else 'off'}",
            "btn-wake": f"Wake: {'on' if self._wake_on else 'off'}",
        }
        for bid, label in labels.items():
            try:
                self.query_one("#" + bid, Button).label = label
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
        try:
            self.call_from_thread(self._render_event, event)
        except Exception:
            pass

    def _render_event(self, event: dict) -> None:
        if event.get("type") == "route":
            self._route_label = f"{event.get('provider')}/{event.get('model')}"
            self._set_status()

    def _on_delta(self, piece: str) -> None:
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

    # ------------------------------------------------------- wake word ----

    def action_toggle_wake(self) -> None:
        if self._wake_on:
            self._wake_on = False
            self._wake_state = "off"
            self._set_status()
            self._add_event("Wake word off", "note")
            return
        if self._wake_thread and self._wake_thread.is_alive():
            return
        self._wake_on = True
        self._wake_state = "starting"
        self._set_status()
        self._wake_thread = threading.Thread(target=self._wake_loop,
                                             name="tui-wake", daemon=True)
        self._wake_thread.start()
        log.info("wake loop starting")
        self._add_event("Wake word on — say the wake phrase", "note")

    def _set_wake_state(self, state: str) -> None:
        self._wake_state = state
        self._set_status()

    def _wake_loop(self) -> None:
        try:
            from .wake import wait_for_wake, listen_after_wake
            from . import sounds
            from .__main__ import _greet, _is_stop, _is_echo
            from .voice import listen, speak
        except Exception as e:
            log.warning("wake unavailable: %s", e)
            self.call_from_thread(self._add_event,
                                  f"wake unavailable: {e}", "error")
            self._wake_on = False
            self.call_from_thread(self._set_wake_state, "off")
            return
        phrase = (config.get("JARVIS_WAKE_WORDS") or "hey_optimus") \
            .replace("_", " ").split(",")[0]
        # Pre-load the command STT model off the critical path so the first
        # spoken command transcribes instantly (and load errors surface now,
        # rather than silently during a voice turn).
        try:
            from .voice import get_stt
            threading.Thread(
                target=get_stt,
                args=(os.getenv("JARVIS_COMMAND_MODEL")
                      or config.get("JARVIS_WHISPER_MODEL"),),
                name="tui-stt-warm", daemon=True).start()
        except Exception:
            pass
        self.call_from_thread(self._set_wake_state, f"armed ({phrase})")
        while self._wake_on:
            try:
                woke = False
                for _ in wait_for_wake():
                    woke = True
                    break
                if not woke or not self._wake_on:
                    continue
                log.info("wake word detected")
                self.call_from_thread(self._set_wake_state, "heard wake word")
                sounds.ack()
                time.sleep(0.25)
                self.call_from_thread(self._set_wake_state, "greeting")
                _greet()
                self.call_from_thread(self._set_wake_state, "listening")
                q = listen_after_wake()
                if not q:
                    log.info("false wake (no speech)")
                    self.call_from_thread(
                        self._add_event,
                        "Didn't catch that — say your request again", "note")
                    time.sleep(2)
                    self.call_from_thread(self._set_wake_state,
                                          f"armed ({phrase})")
                    continue
                log.info("transcript: %s", q[:200])
                spoken = self._voice_turn_sync(q, speak)
                if config.get_bool("JARVIS_FOLLOWUP"):
                    spoken = self._voice_followups(spoken, listen, _is_stop,
                                                   _is_echo, speak)
                self.call_from_thread(self._set_wake_state,
                                      f"armed ({phrase})")
            except Exception as e:
                log.exception("wake loop error")
                self.call_from_thread(self._add_event,
                                      f"wake error: {e}", "error")
                time.sleep(2)
        log.info("wake loop stopped")
        self.call_from_thread(self._set_wake_state, "off")

    def _voice_turn_sync(self, question: str, speak) -> str:
        self.call_from_thread(self._begin_voice_turn, question)
        try:
            answer = self.agent.ask(
                question,
                on_delta=lambda piece: self.call_from_thread(
                    self._on_delta, piece))
        except Exception as e:
            answer = f"error: {e}"
        self.call_from_thread(self._finish_answer, answer)
        if answer and not answer.startswith("error:"):
            try:
                speak(answer)
            except Exception:
                pass
        return answer

    def _voice_followups(self, last_answer: str, listen, is_stop, is_echo,
                         speak) -> str:
        window = config.get_float("JARVIS_FOLLOWUP_SECONDS")
        spoken = last_answer
        while self._wake_on:
            self.call_from_thread(self._set_wake_state,
                                  f"follow-up ({int(window)}s)")
            q = listen(timeout=window, phrase_limit=30)
            if not q or is_stop(q):
                self.call_from_thread(self._set_wake_state, "armed")
                return spoken
            if is_echo(q, spoken):
                continue
            spoken = self._voice_turn_sync(q, speak)
        return spoken

    # ------------------------------------------------------------- voice --

    def _voice_module(self):
        if self._voice is None:
            from . import voice
            self._voice = voice
        return self._voice

    def _speak(self, text: str) -> None:
        if not self._voice_on or not (text or "").strip():
            return
        try:
            self._voice_module().speak(text, blocking=False)
        except Exception as e:
            log.warning("voice output unavailable: %s", e)
            self._add_event(f"voice output unavailable: {e}", "error")

    def _stop_speaking(self) -> None:
        try:
            self._voice_module().stop_tts()
        except Exception:
            pass

    def action_toggle_voice(self) -> None:
        self._voice_on = not self._voice_on
        self._set_status()
        self._add_event(f"Spoken replies {'on' if self._voice_on else 'off'}",
                        "note")

    def action_talk(self) -> None:
        if self._listening:
            return
        if self._busy:
            self.notify("Wait for the current answer first.",
                        severity="warning")
            return
        self._listening = True
        self._set_status()
        self._add_event("Listening… speak now", "note")
        self.run_worker(self._listen_blocking, thread=True,
                        exclusive=False, name="listen")

    def _listen_blocking(self) -> None:
        try:
            text = self._voice_module().listen(timeout=8, phrase_limit=30)
        except Exception as e:
            text, err = "", str(e)
        else:
            err = ""
        self.call_from_thread(self._after_listen, text, err)

    def _after_listen(self, text: str, err: str = "") -> None:
        self._listening = False
        self._set_status()
        if err:
            log.warning("voice input unavailable: %s", err)
            self._add_event(f"voice input unavailable: {err}", "error")
            return
        text = (text or "").strip()
        if not text:
            self.notify("No speech detected.", severity="warning")
            return
        self._add_event(f"heard: {text}", "note")
        self._submit_question(text)

    def _begin_voice_turn(self, question: str) -> None:
        self._voice_turn = True
        self._add_user(question)
        self._reset_stream()
        self._route_label = ""
        self._busy = True
        self._set_status()

    def action_stop_speech(self) -> None:
        self._stop_speaking()

    # --------------------------------------------------------- run a turn --

    def _submit_question(self, text: str) -> None:
        self._stop_speaking()
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
        voice_turn = self._voice_turn
        self._voice_turn = False
        self._busy = False
        self._set_status()
        if not voice_turn:
            self._speak(final)
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

    # ------------------------------------------------------ confirmation --

    def _confirm(self, reason: str, fn_name: str, args: dict,
                 mandatory: bool = False) -> bool:
        done = threading.Event()
        box = {}

        def show() -> None:
            def callback(result) -> None:
                box["result"] = bool(result)
                done.set()
            try:
                self.push_screen(ConfirmScreen(reason, fn_name, args,
                                               mandatory), callback)
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

    # --------------------------------------------------------- reminders --

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

    # --------------------------------------------------------- commands ---

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
        elif cmd == "/wake":
            if rest.lower() in ("on", "1", "true", "yes"):
                if not self._wake_on:
                    self.action_toggle_wake()
            elif rest.lower() in ("off", "0", "false", "no"):
                if self._wake_on:
                    self.action_toggle_wake()
            else:
                self.action_toggle_wake()
        elif cmd == "/voice":
            if rest.lower() in ("on", "1", "true", "yes"):
                self._voice_on = True
            elif rest.lower() in ("off", "0", "false", "no"):
                self._voice_on = False
            else:
                self._voice_on = not self._voice_on
            self._set_status()
            self._add_event(
                f"Spoken replies {'on' if self._voice_on else 'off'}", "note")
        else:
            self._add_event(f"Unknown command {cmd} — try /help", "error")

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
            f"Confirmation mode -> {'auto' if enabled else 'ask'}", "note")

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
        self._add_event("Started a new conversation", "note")

    def _show_status(self) -> None:
        mode = "auto" if config.get_bool("JARVIS_AUTO") else "ask"
        try:
            pending = len(scheduler.list_pending())
        except Exception:
            pending = 0
        try:
            mem = f"{self.agent.memory.count()} entries"
        except Exception:
            mem = "n/a"
        lines = [
            f"**{self._name}**", "",
            f"- confirmations: **{mode}**",
            f"- pending reminders: **{pending}**",
            f"- conversation memory: {mem}",
        ]
        self._add_assistant_text("\n".join(lines))

    def _show_reminders(self) -> None:
        try:
            rows = scheduler.list_pending()
        except Exception as e:
            self._add_event(f"Could not read reminders: {e}", "error")
            return
        if not rows:
            self._add_assistant_text("No pending reminders.")
            return
        lines = ["**Pending reminders**", ""]
        for r in rows:
            when = datetime.fromtimestamp(r["due_at"]).strftime(
                "%a %d %b, %H:%M")
            repeat = f" ({r['repeat']})" if r.get("repeat") else ""
            lines.append(f"- `{r['id']}` {r['message']} — {when}{repeat}")
        self._add_assistant_text("\n".join(lines))

    def _start_briefing(self) -> None:
        if self._busy:
            self.notify("Wait for the current answer first.",
                        severity="warning")
            return
        self._busy = True
        self._set_status()
        self._add_event("Building the daily briefing…", "note")
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
            self._add_event(f"Could not read jobs: {e}", "error")
            return
        lines = [f"**Jobs** — {stats['total']} tracked", ""]
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
                lines.append(f"- [{score_txt}] {title} — {company}")
        else:
            lines.extend(["", "No jobs yet. Ask Soundwave to run a sweep."])
        self._add_assistant_text("\n".join(lines))

    def _finish_command_text(self, text: str) -> None:
        self._add_assistant_text(text or "(empty)")
        self._speak(text)
        self._busy = False
        self._set_status()
        try:
            self.query_one("#prompt", Input).focus()
        except Exception:
            pass

    # ---------------------------------------------------------- actions ---

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


def _setup_tui_logging() -> None:
    """TUI logs go to a file (stdout is the Textual screen)."""
    from pathlib import Path
    raw = os.environ.get("JARVIS_TUI_LOG") or \
        str(Path(__file__).resolve().parents[1] / "data" / "logs" / "tui.log")
    path = Path(raw).expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
    fh = logging.FileHandler(path)
    fh.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root.addHandler(fh)
    root.setLevel(logging.INFO)
    log.info("TUI started (log=%s)", path)


def run_tui(agent=None) -> None:
    _setup_tui_logging()
    OptimusTUI(agent=agent).run()
