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
``GET /api/stream``
    A Server-Sent Events stream of every state change.
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

**Why SSE and not polling.** The first version had the browser ask
``GET /api/status`` every two seconds. That is a request every two seconds per
open tab, forever, to be told things that change roughly five times per lecture.
Worse, it is *late by up to two seconds* precisely when the operator wants to
know what is happening. Pushing state changes means zero requests while idle
and no lag while busy.

**Why a lock.** :data:`APP_STATE` is mutated from the background thread and
read from request handlers on the event loop. A dict assignment is atomic under
CPython's GIL, but a read-then-copy of several keys is not, so a snapshot could
observe a half-updated state -- e.g. ``status`` already ``"complete"`` while
``message`` still describes transcription. The lock makes each snapshot
internally consistent.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import BackgroundTasks, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
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
# would race with anything still holding a reference to the original.
APP_STATE: dict[str, Any] = {
    "status": "idle",
    "course_code": None,
    "message": "",
}

# Guards every read and write of APP_STATE. Held only for the handful of
# statements that touch the dict, never across a pipeline call, and never while
# publishing -- the broadcast is fire-and-forget, so there is no reason to hold a
# lock across it.
_state_lock = threading.Lock()

# Held for the whole duration of a run so a second POST cannot start a second
# pipeline. Separate from _state_lock on purpose: this one is held for minutes,
# that one for microseconds, and merging them would make every state write block
# behind the recording it is reporting on.
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

# ---------------------------------------------------------------------------
# Broadcasting
# ---------------------------------------------------------------------------

SSE_MEDIA_TYPE = "text/event-stream"

# An SSE connection with no traffic for a couple of minutes is a connection a
# proxy in the middle has decided to close, usually silently. A comment frame
# costs one line and keeps the pipe demonstrably alive; per the SSE spec it is
# ignored by the client, so EventSource never sees it.
SSE_KEEPALIVE_SECONDS = 15.0

# Per-subscriber backlog. A lecture emits about five state changes, so this is
# generous -- the bound exists to cap memory for a client that has stopped
# reading, not to hold a replay log.
_QUEUE_SIZE = 16

# Pushed to every subscriber on shutdown so the stream generators wake up and
# exit instead of being abandoned mid-``await``.
_SHUTDOWN = object()


def _sse_frame(payload: dict[str, Any]) -> str:
    """Render one SSE ``data:`` frame.

    Compact separators: a state frame is 120 bytes either way, but there is no
    reason to make the parser's job harder.
    """
    return f"data: {json.dumps(payload, separators=(',', ':'))}\n\n"


class Broadcaster:
    """Fan state changes out to every connected SSE client.

    The awkward part of this is that the two sides live on different threads.
    :meth:`publish_threadsafe` is called from the pipeline worker (a
    ``threading`` threadpool worker started by ``BackgroundTasks``) while
    :meth:`subscribe` and :meth:`_deliver` run on the event loop, inside the
    stream generator.

    ``asyncio.Queue`` is not thread-safe, and worse than merely racy: calling
    ``put_nowait`` from a foreign thread updates the queue's internal deque but
    never signals the waiter, so a connected client would simply never be woken
    -- the dashboard would sit on "Recording..." and then silently jump
    straight to "complete". Every cross-thread hop therefore goes through
    ``loop.call_soon_threadsafe``, which is the one API designed for exactly
    this and is safe even when the loop is in another thread.

    The subscriber set is only ever touched on the event loop (``subscribe``,
    ``_deliver`` and ``unsubscribe`` are all called from the generator), so it
    needs no lock of its own.
    """

    def __init__(self) -> None:
        self._subscribers: set[asyncio.Queue[Any]] = set()
        self._loop: asyncio.AbstractEventLoop | None = None

    def bind_loop(self, loop: asyncio.AbstractEventLoop) -> None:
        """Remember the loop that :meth:`publish_threadsafe` must hand work to.

        Captured in the app's ``lifespan``, which is one of the few places a
        running loop is guaranteed to be reachable. The worker thread has none:
        ``asyncio.get_running_loop()`` raises ``RuntimeError`` there, so the loop
        has to be stashed here and used by reference.
        """
        self._loop = loop

    @property
    def subscriber_count(self) -> int:
        """How many clients are currently attached.

        Loop-affine: read it from a coroutine, not from a worker thread.
        """
        return len(self._subscribers)

    def subscribe(self) -> asyncio.Queue[Any]:
        """Register a new client and return the queue it should read from.

        Must be called on the event loop.
        """
        queue: asyncio.Queue[Any] = asyncio.Queue(maxsize=_QUEUE_SIZE)
        self._subscribers.add(queue)
        return queue

    def unsubscribe(self, queue: asyncio.Queue[Any]) -> None:
        """Detach a client. Idempotent, and must be called on the event loop.

        Reached via the generator's ``finally``, so it runs on normal
        completion, on client disconnect, and on cancellation alike. Skipping it
        would leak the queue for the life of the process.
        """
        self._subscribers.discard(queue)

    def publish_threadsafe(self, payload: dict[str, Any]) -> None:
        """Broadcast a state snapshot, callable from any thread.

        A no-op before the loop is bound or after it closes, which is the correct
        behaviour: it means "nobody is listening", not "drop the run". The run's
        own outcome is still recorded in :data:`APP_STATE` either way.
        """
        loop = self._loop
        if loop is None or loop.is_closed():
            return
        try:
            loop.call_soon_threadsafe(self._deliver, payload)
        except RuntimeError:
            # The loop was closed between the check above and the call. Shutdown
            # is already underway, so there is nothing left to notify.
            pass

    def _deliver(self, payload: dict[str, Any]) -> None:
        """Hand one payload to every subscriber. Runs on the event loop."""
        for queue in list(self._subscribers):
            if queue.full():
                # Coalesce: drop the backlog and keep only the newest state.
                # Better that a briefly-slow client skips a transitional state
                # than that it falls permanently behind and reports a stale
                # phase for the rest of the lecture.
                while not queue.empty():
                    queue.get_nowait()
            queue.put_nowait(payload)

    def shutdown(self) -> None:
        """Wake every generator so it can exit cleanly. Runs on the event loop."""
        for queue in list(self._subscribers):
            queue.put_nowait(_SHUTDOWN)
        self._subscribers.clear()


_broadcaster = Broadcaster()


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------


def _set_state(status: str, message: str = "", course_code: Any = ...) -> None:
    """Update :data:`APP_STATE` and push the new snapshot to every subscriber.

    Args:
        status: The new machine-readable status token.
        message: Human-readable detail. Defaults to the token's canned text.
        course_code: New course code, or ``...`` to leave the current value
            untouched. A sentinel rather than ``None`` because ``None`` is itself
            a meaningful value here -- it is how a finished run reports "no
            course" -- so it cannot double as "not supplied".

    The snapshot is copied *and scheduled* while the lock is held, which is what
    keeps the stream in order. Writing under the lock and scheduling after
    releasing it leaves a window in which two publishers interleave: the request
    thread can write ``recording``, the pipeline thread can write and broadcast
    ``transcribing``, and only then does the first one get around to broadcasting
    ``recording`` -- so a client would watch a lecture go back a phase. One lock
    around write-copy-schedule collapses that window to nothing.

    Holding the lock across ``publish_threadsafe`` costs nothing: it appends to a
    deque and pokes the loop's self-pipe, neither of which blocks. The earlier
    version released the lock first on the theory that publishing might block on
    a slow subscriber; it cannot, because the actual delivery happens later on
    the event loop, and the arrangement only bought an ordering bug.
    """
    with _state_lock:
        APP_STATE["status"] = status
        APP_STATE["message"] = message or _STATE_MESSAGES.get(status, status)
        if course_code is not ...:
            APP_STATE["course_code"] = course_code
        # Copied here rather than via get_state(), which would re-enter the
        # non-reentrant lock and deadlock.
        snapshot = dict(APP_STATE)
        _broadcaster.publish_threadsafe(snapshot)


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


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Bind the broadcaster to this process's event loop.

    ``on_event`` would do the same job but is deprecated in FastAPI 0.142, and
    the whole broadcaster depends on holding a live loop reference, so the
    modern entry point is worth the few lines.
    """
    _broadcaster.bind_loop(asyncio.get_running_loop())
    try:
        yield
    finally:
        # Without this, uvicorn's shutdown would wait on stream generators that
        # are parked in `await queue.get()` with no pending work to complete
        # them.
        _broadcaster.shutdown()


app = FastAPI(
    title="Academic Capture Engine",
    description="Record, transcribe and summarise a lecture.",
    version="1.0.0",
    lifespan=lifespan,
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
    the states. Without it a subscriber connecting between two states would show
    the new status attached to the *previous* lecture's code.

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
    event loop stays free to keep pushing updates for the whole run. An
    ``async def`` here would be the opposite of what is wanted -- it would be
    awaited on the loop and block every connected client for the entire lecture.

    Every state change reaches subscribers through :func:`_set_state`, which
    marshals onto the event loop itself, so this thread never has to know a loop
    exists.

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
        # propagate to. Anything reaching here would otherwise leave every
        # connected dashboard on "transcribing" forever with no way to retry.
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


# ---------------------------------------------------------------------------
# Server-Sent Events
# ---------------------------------------------------------------------------


async def _event_stream() -> AsyncIterator[str]:
    """Yield SSE frames for every state change until told to stop.

    Subscribing happens *before* the first snapshot is read. That ordering is
    deliberate and asymmetric: a state change landing in the gap would
    otherwise be missed entirely, whereas the worst case of the reverse race is
    that a client is sent the same state twice, which renders identically. Missing
    an update is a wrong dashboard; a duplicate is a harmless no-op.
    """
    queue = _broadcaster.subscribe()
    try:
        # A client that connects mid-lecture learns the current state at once
        # rather than waiting for the next change, which could be minutes away.
        yield _sse_frame(get_state())

        while True:
            try:
                payload = await asyncio.wait_for(
                    queue.get(), timeout=SSE_KEEPALIVE_SECONDS
                )
            except asyncio.TimeoutError:
                yield ": keepalive\n\n"
                continue

            if payload is _SHUTDOWN:
                break

            yield _sse_frame(payload)
    finally:
        # Reached on normal shutdown, on client disconnect, and on cancellation
        # from Starlette tearing down the response -- all three, because
        # GeneratorExit and CancelledError both unwind through `finally`.
        _broadcaster.unsubscribe(queue)


@app.get("/api/stream")
def stream_events() -> StreamingResponse:
    """Subscribe to state changes as a Server-Sent Events stream.

    Returns:
        An endless ``text/event-stream`` response. It is deliberately never
        closed by the server: the client hangs up, or the process shuts down.

    Replaces the old polled ``GET /api/status``. Nothing polls any more, and the
    one-shot read is gone with it -- :data:`APP_STATE` is still readable in
    Python for anything that needs a synchronous snapshot, but a client that
    wants to know the state should subscribe and be handed the current value
    immediately as its first frame.
    """
    return StreamingResponse(
        _event_stream(),
        media_type=SSE_MEDIA_TYPE,
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            # Tells nginx (and anything else in front) not to buffer this
            # response. A proxy that batches the frames defeats the entire
            # point -- the client would get both states at once, on a delay,
            # which is exactly the behaviour SSE exists to remove.
            "X-Accel-Buffering": "no",
        },
    )


# ---------------------------------------------------------------------------
# Run control
# ---------------------------------------------------------------------------


@app.post("/api/record")
def start_recording(payload: RecordRequest, background: BackgroundTasks) -> dict[str, Any]:
    """Start a pipeline run and return immediately.

    Args:
        payload: Course code and language.
        background: Starlette's post-response task runner.

    Returns:
        An acknowledgement carrying the accepted course code and status. The
        run is already in flight by the time this is serialised, and any
        connected dashboard has already been pushed the ``recording`` state --
        the call to :func:`_set_state` below broadcasts before the response is
        even built.

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

    Separate from the state stream so a monitor can distinguish "the server is
    up" from "a lecture is running" -- ``/api/stream`` is a hanging connection
    and useless as a probe, while this always answers immediately and its
    payload never changes with the pipeline.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Static dashboard
# ---------------------------------------------------------------------------
# Registered LAST, and the ordering is load-bearing rather than incidental.
# Starlette matches routes in registration order, and a mount at "/" matches
# every path. Registering it first would shadow /api/stream and /api/record
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
