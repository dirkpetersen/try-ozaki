"""try-ozaki FastAPI web application.

Mirrors the codecheck architecture:
  - Session isolation: each job gets a temp dir, cleaned up after 2h
  - Streaming SSE output for all pipeline stages
  - Accepts GitHub URL (clones to temp dir) or local path
  - No authentication (deployment is behind appmotel's network boundary)

Run:
    uvicorn try_ozaki.app:app --host 0.0.0.0 --port 8000
"""

import asyncio
import shutil
import subprocess
import tempfile
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .events import Event, sse_format
from .pipeline import run_pipeline

SESSION_TTL_SECS = 2 * 60 * 60   # 2 hours

# In-memory session map: session_id → {tmp_dir, created}
_sessions: dict[str, dict] = {}

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(title="try-ozaki")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


def _cleanup_sessions() -> None:
    now = time.time()
    expired = [sid for sid, s in _sessions.items()
               if now - s.get("created", 0) > SESSION_TTL_SECS]
    for sid in expired:
        tmp = _sessions.pop(sid, {}).get("tmp_dir")
        if tmp:
            shutil.rmtree(tmp, ignore_errors=True)


def _clone_repo(url: str, dest: Path) -> tuple[bool, str]:
    """Clone a GitHub repo into dest. Returns (ok, error_message)."""
    result = subprocess.run(
        ["git", "clone", "--depth=1", url, str(dest)],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        return False, result.stderr[:500]
    return True, ""


@app.get("/", response_class=HTMLResponse)
async def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/version")
async def version():
    from . import __version__
    return {"version": __version__}


@app.get("/evaluate")
async def evaluate(
    request: Request,
    source: str = "",
    project: str = "",
    tolerance: float = 1e-6,
    image: str = "",
    mode: str = "full",          # "full" | "no_submit" | "dry_run"
):
    """Stream the try-ozaki pipeline as Server-Sent Events.

    ?source=<github-url-or-local-path>
    """
    _cleanup_sessions()

    session_id = uuid.uuid4().hex[:8]
    tmp_dir = Path(tempfile.mkdtemp(prefix=f"try-ozaki-web-{session_id}-"))
    _sessions[session_id] = {"tmp_dir": str(tmp_dir), "created": time.time()}

    async def generate():
        try:
            # Resolve source path
            if source.startswith("http://") or source.startswith("https://"):
                yield sse_format(Event("status", "Cloning repository...", session_id))
                repo_dir = tmp_dir / "repo"
                ok, err = await asyncio.get_event_loop().run_in_executor(
                    None, _clone_repo, source, repo_dir,
                )
                if not ok:
                    yield sse_format(Event("error", f"Git clone failed: {err}"))
                    yield sse_format(Event("done", ""))
                    return
                source_path = repo_dir
            else:
                source_path = Path(source)
                if not source_path.is_dir():
                    yield sse_format(Event("error", f"Not a directory: {source}"))
                    yield sse_format(Event("done", ""))
                    return

            async for event in run_pipeline(
                source_path=source_path,
                project=project or None,
                image=image or None,
                tolerance=tolerance,
                no_submit=(mode == "no_submit"),
                dry_run=(mode == "dry_run"),
                session_id=session_id,
            ):
                yield sse_format(event)
                # respect client disconnect
                if await request.is_disconnected():
                    return

        except Exception as e:
            yield sse_format(Event("error", str(e)))
            yield sse_format(Event("done", ""))
        finally:
            # Session cleanup deferred to TTL expiry so /results/<id> still works
            pass

    return StreamingResponse(generate(), media_type="text/event-stream")
