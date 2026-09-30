"""HTTP front end for the lecture pipeline (Phase 7).

Wraps :func:`main.run_pipeline` in a small FastAPI application and serves a
static dashboard shell from ``static/``. The pipeline itself is unchanged and
still perfectly usable from a terminal -- this module adds a way to *start* a
run and *watch* it, not a second implementation of it.

Endpoints
---------
``POST /api/record``
    Accepts ``{"course_code": ..., "language": ...}``, returns immediately, and
    runs the pipeline in the background.
``GET /api/status``
    Returns the current :data:`APP_STATE` snapshot.
``GET /``
    The dashboard shell.

Run it with::

    uvicorn server:app --reload

Design notes
------------
**Why the response is immediate.** A lecture run is tens of minutes of blocking
work. Holding the HTTP request open for that long would sit on a worker, trip
every proxy timeout between here and the browser, and leave the operator with a
spinner and no way to ask what happened. So the request only *starts* the run.

**Why the callback is the source of truth for state.** ``run_pipeline`` already
reports its own progress; this module does not re-derive progress by watching
files or parsing stdout. The pipeline is asked, and its answer is stored. That
keeps the server from needing a second model of the pipeline that could drift
out of sync with the first.

**Why a lock.** :data:`APP_STATE` is mutated from the background thread and
read from request handlers on the event loop. A dict assignment is atomic under
CPython's GIL, but a read-then-copy of several keys is not, so a status poll
could observe a half-updated state -- e.g. ``status`` already ``"complete"``
while ``message`` still describes transcription. The lock makes each snapshot
internally consistent.
"""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Callable

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from main import (
    STATE_COMPLETE,
    STATE_ERROR,
    STATE_RECORDING,
    STATE_SYNTHESIZING,
    STATE_TRANSCRIBING,
    PipelineError,
    _sanitize_course_code,
    resolve_device_index,
    run_pipeline,
)
from transcriber import DEFAULT_LANGUAGE

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------

# The three keys are fixed by the dashboard's contract, so they are written in
# place rather than replaced: a handler that copied the dict and swapped it in
# would race with the poll loop holding a reference to the original.
APP_STATE: dict[str, Any] = {
    "status": "idle",
    "course_code": None,
    "message": "",
}

# Guards every read and write of APP_STATE. Held only for the handful of
# statements that touch the dict, never across a pipeline call.
_state_lock = threading.Lock()

# Held for the whole duration of a run so a second POST cannot start a second
# pipeline. Separate from _state_lock on purpose: this one is held for minutes,
# that one for microseconds, and merging them would make every status poll block
# on the recording.
_run_lock = threading.Lock()
_run_active = False

# What each state means in a sentence a human can act on. The wire format stays
# the bare token; only the presentation is added.
_STATE_MESSAGES: dict[str, str] = {
    "idle": "Ready. Submit a course code to begin a recording.",
    STATE_RECORDING: "Recording loopback audio. Press Ctrl+C in the terminal to stop.",
    STATE_TRANSCRIBING: "Transcribing and diarizing. This is the longest phase.",
    STATE_SYNTHESIZING: "Synthesizing study notes from the transcript and slides.",
    STATE_COMPLETE: "Pipeline complete. Study notes written to disk.",
    STATE_ERROR: "Pipeline failed. See the terminal for detail.",
}


def _set_state(status: str, message: str = "", course_code: Any = ...) -> None:
    """Update :data:`APP_STATE` atomically.

    Args:
        status: The new machine-readable status token.
        message: Human-readable detail. Defaults to the token's canned text.
        course_code: New course code, or ``...`` to leave the current value
            untouched. A sentinel rather than ``None`` because ``None`` is itself
            a meaningful value here -- it is how a finished run reports "no
            course" -- so it cannot double as "not supplied".
    """
    with _state_lock:
        APP_STATE["status"] = status
        APP_STATE["message"] = message or _STATE_MESSAGES.get(status, status)
        if course_code is not ...:
            APP_STATE["course_code"] = course_code


def get_state() -> dict[str, Any]:
    """Return a consistent copy of :data:`APP_STATE`.

    A copy, not the live dict: handing the original to a response encoder would
    let the JSON serialiser iterate it while the background thread mutates it.
    """
    with _state_lock:
        return dict(APP_STATE)


# ---------------------------------------------------------------------------
# Application
# ---------------------------------------------------------------------------

app = FastAPI(
    title="Academic Capture Engine",
    description="Record, transcribe and summarise a lecture.",
    version="1.0.0",
)


class RecordRequest(BaseModel):
    """Body of ``POST /api/record``.

    ``course_code`` is length-bounded because it is interpolated straight into
    an output directory name; an unbounded string is a filesystem DoS at best.
    It is sanitised again in :func:`_run_in_background` -- the browser is not a
    trusted caller, and this is the boundary where that stops being true.
    """

    course_code: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Course unit code, e.g. CRM_1106.",
    )
    language: str = Field(
        default=DEFAULT_LANGUAGE,
        min_length=2,
        max_length=2,
        description="ISO-639-1 spoken-language code, e.g. en or sw.",
    )


def _make_state_callback() -> Callable[[str], None]:
    """Build the sink that mirrors pipeline states into :data:`APP_STATE`.

    A closure rather than a bare function so the course code rides along with
    the states. Without it a poll arriving between two states would show the
    new status attached to the *previous* lecture's code.

    The terminal states get extra handling: ``complete`` records the notes
    path, and ``error`` must set a message the operator can act on, which
    ``run_pipeline`` has already printed but does not pass through the callback
    (the callback receives a bare token, not the exception).
    """

    def on_state(state: str) -> None:
        if state == STATE_COMPLETE:
            _set_state(STATE_COMPLETE)
        elif state == STATE_ERROR:
            _set_state(STATE_ERROR)
        else:
            _set_state(state)

    return on_state


def _run_in_background(course_code: str, language: str) -> None:
    """Execute the pipeline, recording its outcome into :data:`APP_STATE`.

    Runs on Starlette's threadpool because it is a plain synchronous function:
    ``BackgroundTasks`` hands sync callables to ``run_in_threadpool``, so the
    event loop stays free to answer status polls for the whole run. An
    ``async def`` here would be the opposite of what is wanted -- it would be
    awaited on the loop and block every other request for the entire lecture.

    Every exit path releases ``_run_lock``; a run that raised while holding it
    would leave the server permanently refusing new requests.
    """
    global _run_active
    try:
        notes_path = run_pipeline(
            course_code,
            language,
            _make_state_callback(),
            # No terminal to prompt on, so both interactive decisions are made
            # here instead. Refusing to overwrite is the safe default: a
            # browser has no way to ask the operator.
            device_id=resolve_device_index(),
            confirm_overwrite=False,
        )
    except PipelineError as exc:
        # Expected failure, already reported to the terminal by run_pipeline.
        _set_state(STATE_ERROR, f"{STATE_ERROR.title()}: {exc}")
    except Exception as exc:  # noqa: BLE001 - the last line of defence
        # run_pipeline is not exhaustive, and a background task has no caller to
        # propagate to. Anything reaching here would otherwise leave the
        # dashboard on "transcribing" forever with no way to retry.
        _set_state(STATE_ERROR, f"Unexpected failure: {exc!r}")
        raise
    else:
        _set_state(
            STATE_COMPLETE,
            f"Pipeline complete. Study notes written to {notes_path}.",
        )
    finally:
        with _run_lock:
            _run_active = False


@app.get("/api/status")
def read_status() -> dict[str, Any]:
    """Return the current pipeline state.

    Polled by the dashboard every two seconds. Deliberately a bare dict with no
    envelope: the payload is three keys, and a wrapper would be one more shape
    for the client to unwrap for no benefit.
    """
    return get_state()


@app.post("/api/record")
def start_recording(payload: RecordRequest, background: BackgroundTasks) -> dict[str, Any]:
    """Start a pipeline run and return immediately.

    Args:
        payload: Course code and language.
        background: Starlette's post-response task runner.

    Returns:
        An acknowledgement carrying the accepted course code and status. The
        run is already in flight by the time this is serialised.

    Raises:
        HTTPException: 409 if a run is already in progress. Starting a second
            pipeline would open a second capture stream on the same loopback
            device, and two concurrent runs would interleave writes to
            ``raw_audio.wav`` and race on the same output directory.
    """
    global _run_active
    with _run_lock:
        if _run_active:
            raise HTTPException(
                status_code=409,
                detail=(
                    "A recording is already in progress. Wait for it to finish "
                    "before starting another."
                ),
            )
        _run_active = True

    course_code = _sanitize_course_code(payload.course_code.strip())
    if not course_code:
        # Reachable when the code was nothing but path-unsafe characters, e.g.
        # "/" or "...". The lock is already held, so release it before bailing.
        with _run_lock:
            _run_active = False
        raise HTTPException(
            status_code=422,
            detail="Course code contained no usable characters.",
        )

    _set_state(STATE_RECORDING, course_code=course_code)
    background.add_task(_run_in_background, course_code, payload.language.lower())
    return {"accepted": True, "course_code": course_code, "status": STATE_RECORDING}


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe.

    Separate from ``/api/status`` so a monitor can distinguish "the server is
    up" from "a lecture is running" -- the status endpoint's payload changes
    with the pipeline, this one never does.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static dashboard
# ---------------------------------------------------------------------------
# Registered LAST, and the ordering is load-bearing rather than incidental.
# Starlette matches routes in registration order, and a mount at "/" matches
# every path. Registering it first would shadow /api/status and /api/record
# with a 404 from the static handler, producing a dashboard that loads and then
# silently never updates.

STATIC_DIR = Path(__file__).resolve().parent / "static"


@app.get("/")
def index() -> FileResponse:
    """Serve the dashboard shell.

    An explicit route rather than relying on ``StaticFiles(html=True)``, so that
    ``/`` is served with a definite 200 and the file path is visible in the
    source instead of being implied by mount configuration.
    """
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(
            status_code=500,
            detail=(
                f"Dashboard shell missing at {index_file}. Create static/index.html."
            ),
        )
    return FileResponse(index_file)


if STATIC_DIR.is_dir():
    app.mount("/", StaticFiles(directory=str(STATIC_DIR)), name="static")
