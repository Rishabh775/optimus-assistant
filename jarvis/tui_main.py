"""Entrypoint for the Optimus TUI.

Kept separate from ``jarvis.__main__`` so a lean TUI launch does not drag in
the voice stack (sounddevice, faster-whisper, wake-word models) just to show
a terminal chat. Run with ``python -m jarvis.tui_main``.
"""
import os
import sys
import warnings
from pathlib import Path


def _load_env() -> None:
    from dotenv import load_dotenv
    candidates = [Path.cwd() / ".env", Path.home() / "jarvis" / ".env"]
    for path in candidates:
        if path.is_file():
            load_dotenv(path, override=False)


def _check() -> None:
    from jarvis import config
    from jarvis.agent import Agent
    from jarvis.tui import OptimusTUI
    agent = Agent(confirm=None)
    print(f"ok: {config.get('JARVIS_NAME')} | memory={agent.memory.count()} "
          f"| {OptimusTUI.__name__}")


def main() -> None:
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
    os.environ.setdefault("HF_HUB_DISABLE_TELEMETRY", "1")
    _load_env()

    if "--check" in sys.argv:
        _check()
        return

    from jarvis import briefing, jobs, push, scheduler
    scheduler.start()
    push.install()
    briefing.install()
    jobs.install()

    from jarvis.tui import run_tui
    run_tui()


if __name__ == "__main__":
    main()
