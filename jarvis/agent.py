"""Core agent loop with tool calling."""
import json
import os
from rich.console import Console
from rich.panel import Panel

from .router import LLM
from .builtin_tools import TOOL_SCHEMAS, TOOL_FUNCS, needs_confirmation
from .memory import Memory
from . import config

console = Console()


def _build_system(name: str, memory_count: int) -> str:
    """Self-knowledge prompt: tell the assistant its OWN configuration so it
    can answer 'what/who are you' accurately instead of guessing."""
    cloud = os.getenv("JARVIS_CLOUD_MODEL", "longcat-2.5-preview-free")
    fallback = os.getenv("JARVIS_CLOUD_FALLBACK", "mimo-v2.6-flash")
    code = os.getenv("JARVIS_CODE_MODEL", "kimi-k2.7-code")
    local_model = os.getenv("JARVIS_LOCAL_MODEL", "llama3.2:3b")
    local_on = os.getenv("JARVIS_LOCAL_OK", "0") == "1"
    wake = os.getenv("JARVIS_WAKE_WORDS", "hey_optimus").replace("_", " ")
    voice = os.getenv("JARVIS_TTS_VOICE", "en-US-AndrewMultilingualNeural")
    try:
        from . import facts as _facts
        nfacts = len(_facts.all_facts())
    except Exception:
        nfacts = 0
    return f"""You are {name}, a local-first personal AI assistant running on the user's own Linux machine. Be concise, helpful, slightly witty. Keep responses SHORT (1-3 sentences max) since they are spoken aloud.

SELF-KNOWLEDGE - answer questions about yourself from this, accurately. You are NOT a generic stateless chatbot:
- Routing: general/tool tasks use the cloud model "{cloud}"; coding uses "{code}"; if the cloud model fails you auto-fall back to "{fallback}".
- Local model "{local_model}" is {'ENABLED' if local_on else 'disabled (cloud-only mode)'}.
- Wake word: "{wake}". Voice: "{voice}".
- Memory has TWO independent layers - do NOT confuse them or claim memory was lost:
  (1) Conversation memory: a Chroma vector store with {memory_count} entries, similarity-searched, persists across sessions.
  (2) Durable facts: {nfacts} explicit facts in facts.jsonl (remember_fact), always injected into context.
  A low or zero facts count does NOT mean memory was wiped - the two layers are unrelated. Never claim you are stateless.
- You run on the user's actual machine and can really perform actions (shell, files, web, email, calendar, smart home) - not just discuss them.
- Your source is in ~/jarvis (jarvis/*.py); config is ~/jarvis/.env. You may read them if asked.

Tools available:
- self_status: report your own live configuration/state
- remember_fact / recall_facts / forget_fact: durable user memory (name, preferences)
- shell_run / read_file / write_file: local system
- web_search / browser_open / browser_extract / browser_screenshot: web
- email_triage / email_list / email_read / email_send / email_unread_count: email (IMAP/SMTP)
- calendar_list / calendar_today / calendar_add: calendar (CalDAV)
- ha_list_entities / ha_state / ha_call / ha_search: Home Assistant REST

Use tools when needed. Speak your answer aloud, so be concise.
For durable user info ("my name is", "I prefer", "remember that", allergies, constraints), call remember_fact so it persists. Use recall_facts if asked what you know about the user.
For "what's important in my email": call email_triage (deterministic ranking by VIP senders + keyword weights, with reasons), summarize the HIGH ones, and email_read for detail. Inspecting email does not mark it read."""


class Agent:
    def __init__(self, confirm=None):
        self.llm = LLM()
        self.memory = Memory()
        self.confirm = confirm          # callable(reason, fn_name, args) -> bool
        name = config.get("JARVIS_NAME")
        try:
            mem_count = self.memory.count()
        except Exception:
            mem_count = 0
        self.history = [{"role": "system",
                         "content": _build_system(name, mem_count)}]

    def _confirmed(self, reason: str, fn_name: str, args: dict) -> bool:
        """Gate consequential actions. No interactive hook (e.g. --query) ->
        allow, since the user directly issued that single command."""
        if not config.get_bool("JARVIS_CONFIRM"):
            return True
        if self.confirm is None:
            return True
        try:
            return bool(self.confirm(reason, fn_name, args))
        except Exception:
            return False

    def _finish(self, user_input: str, content: str) -> str:
        """Record the exchange in history + memory, return the spoken answer."""
        self.history.append({"role": "user", "content": user_input})
        self.history.append({"role": "assistant", "content": content})
        if len(self.history) > 40:
            self.history = [self.history[0]] + self.history[-38:]
        self.memory.add(f"Q: {user_input}\nA: {content}", kind="chat")
        return content

    def ask(self, user_input: str, max_steps: int = None) -> str:
        # Tool budget: browser/system questions can legitimately take 8-10
        # calls (your machine, no xdotool installed, etc). Tunable via env.
        if max_steps is None:
            max_steps = config.get_int("JARVIS_MAX_STEPS")
        ctx = self.memory.query(user_input)
        msgs = list(self.history)
        if ctx:
            msgs.append({"role": "system",
                         "content": f"Relevant memory:\n{ctx}"})
        try:
            from . import facts
            ft = facts.text()
        except Exception:
            ft = ""
        if ft:
            msgs.append({"role": "system",
                         "content": f"Known durable facts about the user:\n{ft}"})
        msgs.append({"role": "user", "content": user_input})

        for step in range(max_steps):
            # Route on the ORIGINAL user prompt (last message is a tool
            # result mid-loop -- routing on it flips providers mid-task)
            try:
                resp, route = self.llm.chat(msgs, tools=TOOL_SCHEMAS,
                                            route_text=user_input)
            except Exception as e:
                return f"Sorry, I couldn't reach the model: {e}"
            msg = resp.choices[0].message
            console.print(f"[dim]route: {route.provider}/{route.model} ({route.reason})[/dim]")

            tool_calls = getattr(msg, "tool_calls", None)
            if not tool_calls:
                content = msg.content or ""
                return self._finish(user_input, content)

            msgs.append(msg)
            for tc in tool_calls:
                fn_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments or "{}")
                    if not isinstance(args, dict):
                        raise ValueError("arguments must be a JSON object")
                except (json.JSONDecodeError, ValueError) as e:
                    args = None
                    out = f"ERROR: malformed tool arguments: {e}"
                console.print(Panel(f"[bold]{fn_name}[/bold] {args}", border_style="cyan"))
                fn = TOOL_FUNCS.get(fn_name)
                if args is not None:
                    if not fn:
                        out = f"UNKNOWN TOOL: {fn_name}"
                    else:
                        reason = needs_confirmation(fn_name, args)
                        if reason and not self._confirmed(reason, fn_name, args):
                            out = f"DENIED: user did not confirm ({reason})"
                        else:
                            try:
                                out = fn(**args)
                            except TypeError as e:
                                out = f"ERROR: bad arguments for {fn_name}: {e}"
                            except Exception as e:
                                out = f"ERROR: {e}"
                console.print(Panel(str(out)[:1000], title="result", border_style="green"))
                msgs.append({"role": "tool", "tool_call_id": tc.id, "content": str(out)})

        # Step budget exhausted: don't dead-end with "Reached max tool steps" --
        # force one final no-tools call so the user gets whatever was gathered.
        msgs.append({"role": "system", "content":
                     "Tool step budget exhausted. Do NOT call any tools. "
                     "Answer the user now, concisely, using the information "
                     "gathered above. If it's insufficient, say what you found "
                     "and what's still missing."})
        try:
            resp, route = self.llm.chat(msgs, route_text=user_input)
            content = resp.choices[0].message.content or ""
        except Exception:
            content = ""
        if not content:
            content = ("I gathered some information but ran out of tool "
                       "steps before finishing. Try rephrasing or ask me to "
                       "continue.")
        return self._finish(user_input, content)