"""Event dataclass and format helpers used by the pipeline generator.

CLI driver prints via terminal_format().
FastAPI driver yields via sse_format() inside a StreamingResponse.
"""

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Event:
    kind: str        # "status" | "chunk" | "stage" | "hotspot" | "report" |
                     # "file" | "job_status" | "log_line" | "error" | "done"
    data: Any = ""
    session_id: str = ""


# ── SSE wire format ───────────────────────────────────────────────────────────

def sse_format(event: Event) -> str:
    """Render an Event as a Server-Sent Events string."""
    payload = event.data if isinstance(event.data, str) else json.dumps(event.data)
    return f"event: {event.kind}\ndata: {payload}\n\n"


# ── Terminal format ───────────────────────────────────────────────────────────

# Map event kinds to a prefix shown on the CLI
_PREFIX = {
    "stage":      "[try-ozaki] ",
    "status":     "[try-ozaki] ",
    "chunk":      "",           # Claude Code output — no prefix, printed raw
    "hotspot":    "  ",
    "job_status": "[try-ozaki] ",
    "log_line":   "  │ ",
    "report":     "",
    "file":       "[try-ozaki] wrote: ",
    "error":      "[try-ozaki] ERROR: ",
    "done":       "",
    "session_id": "",           # not printed on CLI
}


def terminal_format(event: Event) -> str | None:
    """Render an Event for stdout. Returns None for events that are silent on CLI."""
    if event.kind in ("session_id", "done"):
        return None
    prefix = _PREFIX.get(event.kind, "[try-ozaki] ")
    data = event.data if isinstance(event.data, str) else json.dumps(event.data)
    if not data:
        return None
    return f"{prefix}{data}"
