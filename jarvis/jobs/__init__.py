"""Jobs skill: discovery, matching, tracking (application engine next).

Pattern for future skills:
  * keep the subsystem in its own package (state, provider adapters, tests)
  * expose a small TOOL_SCHEMAS / TOOL_FUNCS surface for the agent
  * register scheduled work in install() via scheduler.add_daily
  * never let background work take irreversible actions without a gate
"""
from . import digest, discovery, match, profile, resume, tracker  # noqa: F401
from .digest import install, sweep, top  # noqa: F401
from .tools import TOOL_FUNCS, TOOL_SCHEMAS  # noqa: F401

__all__ = ["digest", "discovery", "match", "profile", "resume", "tracker",
           "install", "sweep", "top", "TOOL_SCHEMAS", "TOOL_FUNCS"]
