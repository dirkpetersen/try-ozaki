"""Invoke the Claude Code CLI as a subprocess and stream its output as Events.

Lifted from ~/codecheck/app.py and adapted for try-ozaki's Event model.

Key design decisions:
- No SDK fallback: if `claude` isn't found we raise immediately with a clear message.
- Full environment inheritance: _build_env() copies os.environ and prepends
  ~/bin and ~/.local/bin so that wrapper scripts (e.g. ~/bin/claude) are found
  even when the process was started by systemd or cron.
- Recursive-call guard: returns None from get_claude_bin() when CLAUDECODE env
  var is set (i.e. we're already inside a Claude Code session).
"""

import asyncio
import json
import os
import shutil
from pathlib import Path

from .events import Event

CLAUDE_CLI_READ_TIMEOUT_SECS = 300   # 5 min idle → kill


def get_claude_bin() -> str | None:
    """Return path to the claude CLI binary, or None if inside Claude Code.

    Search order:
      1. ~/bin/claude          (wrapper scripts live here)
      2. ~/.local/bin/claude   (standard pipx / user install)
      3. shutil.which("claude") (anything else on PATH)
    Returns None when CLAUDECODE env var is set to avoid recursive invocations.
    """
    if os.environ.get("CLAUDECODE"):
        return None
    for d in ["bin", ".local/bin"]:
        candidate = Path.home() / d / "claude"
        if candidate.is_file():
            return str(candidate)
    return shutil.which("claude")


def require_claude_bin() -> str:
    """Return claude binary path or raise RuntimeError with a clear message."""
    path = get_claude_bin()
    if not path:
        raise RuntimeError(
            "claude CLI not found. Install Claude Code and ensure it is on PATH "
            "or at ~/bin/claude. See https://claude.ai/code for installation."
        )
    return path


def _build_env() -> dict[str, str]:
    """Return a copy of the current environment with ~/bin and ~/.local/bin prepended.

    All dotenv-loaded variables (loaded at CLI startup) are already in os.environ
    and therefore included. Every subprocess in try-ozaki should use this so that
    claude, runai, gh, and aws all see the full inherited environment.
    """
    env = os.environ.copy()
    home = Path.home()
    path_parts = env.get("PATH", "").split(":")
    for p in [str(home / "bin"), str(home / ".local" / "bin")]:
        if p not in path_parts:
            env["PATH"] = p + ":" + env.get("PATH", "")
    return env


async def _drain_to_buffer(stream, buf: bytearray, max_bytes: int = 65536) -> None:
    """Read a pipe into a bounded buffer concurrently so it never fills and blocks."""
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            return
        if len(buf) < max_bytes:
            buf.extend(chunk[: max_bytes - len(buf)])


def _parse_cli_event(event: dict) -> list[Event]:
    """Translate one stream-json event dict into zero or more Events."""
    results: list[Event] = []
    etype = event.get("type")

    if etype == "assistant":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") == "text" and block.get("text"):
                results.append(Event("chunk", block["text"]))
            elif block.get("type") == "tool_use":
                tool = block.get("name", "")
                inp = block.get("input", {})
                detail = (
                    inp.get("file_path")
                    or inp.get("command")
                    or inp.get("pattern")
                    or inp.get("query")
                    or ""
                )
                label = f"[{tool}] {detail}\n" if detail else f"[{tool}]\n"
                results.append(Event("chunk", label))

    elif etype == "user":
        for block in event.get("message", {}).get("content", []):
            if block.get("type") != "tool_result":
                continue
            raw = block.get("content", "") or block.get("output", "")
            if isinstance(raw, list):
                raw = "\n".join(
                    b.get("text", "") for b in raw if b.get("type") == "text"
                )
            if raw and isinstance(raw, str):
                lines = raw.strip().splitlines()
                preview = lines[0][:160] if lines else ""
                suffix = f" …({len(lines)} lines)" if len(lines) > 1 else ""
                results.append(Event("chunk", f"  ↳ {preview}{suffix}\n"))

    elif etype == "result":
        result_text = event.get("result", "")
        if result_text and isinstance(result_text, str):
            results.append(Event("report", result_text))

    return results


async def stream_claude_cli(
    prompt: str,
    cwd: str | Path,
    model: str = "sonnet",
) :
    """Run claude CLI on a directory and yield Events as they arrive.

    Yields Event objects — callers decide whether to forward as SSE or print.
    Raises RuntimeError if the claude binary is not found.
    """
    claude_bin = require_claude_bin()
    cmd = [
        claude_bin,
        "-p", prompt,
        "--model", model,
        "--output-format", "stream-json",
        "--verbose",
        "--dangerously-skip-permissions",
    ]

    proc = await asyncio.create_subprocess_exec(
        *cmd,
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        cwd=str(cwd),
        env=_build_env(),
    )

    stderr_buf = bytearray()
    stderr_task = asyncio.create_task(_drain_to_buffer(proc.stderr, stderr_buf))

    try:
        while True:
            try:
                line = await asyncio.wait_for(
                    proc.stdout.readline(),
                    timeout=CLAUDE_CLI_READ_TIMEOUT_SECS,
                )
            except asyncio.TimeoutError:
                proc.kill()
                yield Event(
                    "error",
                    f"Claude CLI timed out after "
                    f"{CLAUDE_CLI_READ_TIMEOUT_SECS // 60} minutes of silence.",
                )
                return

            if not line:
                break
            text = line.decode("utf-8", errors="replace").strip()
            if not text:
                continue
            try:
                event_dict = json.loads(text)
            except json.JSONDecodeError:
                continue
            for ev in _parse_cli_event(event_dict):
                yield ev

        await proc.wait()
        await stderr_task

        if proc.returncode not in (0, None):
            stderr = bytes(stderr_buf).decode("utf-8", errors="replace")
            if proc.returncode == 143:
                yield Event(
                    "error",
                    "Claude CLI was killed (SIGTERM) — likely a memory limit. "
                    "Try a smaller repository or raise MemoryMax on the service unit.",
                )
            else:
                yield Event(
                    "error",
                    f"Claude CLI exited with code {proc.returncode}: {stderr[:500]}",
                )
        else:
            yield Event("done", "")

    finally:
        stderr_task.cancel()
