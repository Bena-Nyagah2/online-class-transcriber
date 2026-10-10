"""HTTP front end for the lecture pipeline (Phase 7).

Wraps :func:`main.run_pipeline` in a small FastAPI application and serves a
static dashboard shell from ``static/``. The pipeline itself is unchanged and
still perfectly usable from a terminal -- this module adds a way to *start* a
run and *watch* it, not a second implementation of it.

Endpoints
---------
``POST /api/record``
    Accepts ``{"course_code": ..., "domain": ..., "language": ...}``, returns
    immediately, and runs the pipeline in the background. ``domain`` picks the
    synthesis profile (default ``criminology``).
``POST /api/stop``
    Ends the recording phase and lets the pipeline proceed to transcription.
``GET /api/stream``
    A Server-Sent Events stream of every state change.
``GET /api/windows``
    Enumerates the titles of the on-screen windows available as slide-capture
    targets, returning ``{"windows": [...]}``.
``GET /api/sessions/{session_id}``
    Fetches one recorded session: its identity fields, the raw study-notes
    Markdown (``markdown_text``), and ``/media`` URLs for the audio and slides
    when those files still exist on disk. Answers 404 for an unknown id.
``/media/<path>``
    Mount of ``output/`` so the playback UI can load raw ``.wav`` and ``.jpg``
    artifacts directly from the browser instead of proxying bytes through the
    API.
``GET /health``
    Liveness probe; returns ``{"status": "ok"}``. The non-``z`` spelling is
    what browser extensions, container runtimes, and proxies probe by default,
    so serving it stops a steady drip of 404s in the terminal.
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

**Why stopping is an Event and not a signal.** The CLI ends a recording with
Ctrl+C, which raises ``KeyboardInterrupt`` in whichever thread owns the
terminal. A browser has no terminal and cannot send a signal, so the recording
loop is instead handed a ``threading.Event`` that ``run_status_loop`` watches.
Setting it ends the loop and drops the pipeline into the same hand-off path a
Ctrl+C takes, including flushing the audio to disk. ``run_pipeline`` stays the
single implementation of "stop recording": this module only supplies the signal.

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
import shutil
import threading
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Callable

from fastapi import BackgroundTasks, Depends, FastAPI, HTTPException
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from database import get_db
from main import (
    STATE_COMPLETE,
    STATE_ERROR,
    STATE_RECORDING,
    STATE_SYNTHESIZING,
    STATE_TRANSCRIBING,
    WAV_FILENAME,
    PipelineError,
    _sanitize_course_code,
    create_output_dir,
    resolve_device_index,
    run_pipeline,
)
from models import Lecture
from transcriber import DEFAULT_LANGUAGE
from vision import get_active_windows

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

# The "end the recording phase" signal for the run in flight, or None when no
# run is recording. Set by POST /api/record and consumed by POST /api/stop.
#
# Created in the request handler rather than inside the background task, and
# that is not a style preference. BackgroundTasks do not start until the
# response has been returned, so an event minted inside the task would leave a
# window in which /api/stop correctly refuses a stop that the operator could
# legitimately have requested. Minting it here means it exists before the 200
# does.
#
# Reading and writing a single module global needs no lock of its own (the
# assignment is atomic under the GIL), and the one genuinely two-field
# question -- "is a recording actually in progress?" -- is answered by the
# status token, which _set_state does lock. See _request_stop for how the two
# are combined and why the residual window is harmless.
_stop_event: threading.Event | None = None

# What each state means in a sentence a human can act on. The wire format stays
# the bare token; only the presentation is added.
_STATE_MESSAGES: dict[str, str] = {
    "idle": "Ready. Submit a course code to begin a recording.",
    STATE_RECORDING: "Recording loopback audio. Press Finish Class when the lecture ends.",
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


# ---------------------------------------------------------------------------
# Static media
# ---------------------------------------------------------------------------

# Root of the artifacts the /media mount serves. ``output`` is relative to the
# process working directory, matching how the pipeline writes it
# (main.create_output_dir). Kept as a module constant so ``_media_url`` below
# builds URLs against the exact same root the server serves from, and so the
# tests can redirect it to a scratch directory instead of the real one.
MEDIA_ROOT = Path("output")

# The directory must exist before the mount or Starlette raises at startup,
# and a fresh clone has no output/ yet. The mkdir is idempotent, so this is a
# one-time guard, not a reset of anything the pipeline wrote.
Path(MEDIA_ROOT).mkdir(exist_ok=True)

app.mount("/media", StaticFiles(directory=str(MEDIA_ROOT)), name="media")


class RecordRequest(BaseModel):
    """Body of ``POST /api/record``.

    ``course_code`` is length-bounded because it is interpolated straight into
    an output directory name; an unbounded string is a filesystem DoS at best.
    It is sanitised again in :func:`_run_in_background` -- the browser is not a
    trusted caller, and this is the boundary where that stops being true.

    ``domain`` selects which synthesis profile from ``summarizer.PROFILES``
    shapes the notes. It is validated only for length here on purpose: an
    unknown key degrades to the generic profile inside the synthesizer rather
    than failing a run that has already captured the lecture, so rejecting it
    at the boundary would buy no safety and cost the operator their session.

    ``capture_mode`` and ``target_window`` choose what the vision daemon
    records: the whole primary monitor, or one named window. They are forwarded
    verbatim to :func:`main.run_pipeline`, which reduces them to a single
    ``window_title`` (``None`` = full screen) before constructing the catcher.
    """

    course_code: str = Field(
        ...,
        min_length=1,
        max_length=64,
        description="Course unit code, e.g. COURSE_101.",
    )
    domain: str = Field(
        default="criminology",
        min_length=1,
        max_length=32,
        description=(
            "Domain profile key selecting the synthesis prompts: "
            "criminology, software, or general. Defaults to criminology; "
            "unknown keys fall back to the general profile."
        ),
    )
    language: str = Field(
        default=DEFAULT_LANGUAGE,
        min_length=2,
        max_length=2,
        description="ISO-639-1 spoken-language code, e.g. en or sw.",
    )
    resume: bool = Field(
        default=False,
        description=(
            "Continue an interrupted run instead of starting a new one. When "
            "true the existing-take overwrite guard is bypassed: the audio "
            "already on disk is the lecture being recovered, so capture is "
            "skipped and the pipeline restarts at the first unfinished stage."
        ),
    )
    capture_mode: str = Field(
        default="fullscreen",
        description=(
            'Capture source: "fullscreen" grabs the primary monitor; '
            '"window" restricts capture to the on-screen window named by '
            '"target_window". Any other value is treated as fullscreen.'
        ),
    )
    target_window: str | None = Field(
        default=None,
        description=(
            "Title of the window to capture when capture_mode is 'window'. "
            "Taken from the /api/windows picker; matching is a case-insensitive "
            "substring, so a stable prefix also works. Ignored otherwise."
        ),
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


def _run_in_background(
    course_code: str,
    language: str,
    stop_event: threading.Event,
    resume: bool = False,
    skipping_capture: bool = False,
    domain: str = "criminology",
    capture_mode: str = "fullscreen",
    target_window: str | None = None,
) -> None:
    """Execute the pipeline, recording its outcome into :data:`APP_STATE`.

    Runs on Starlette's threadpool because it is a plain synchronous function:
    ``BackgroundTasks`` hands sync callables to ``run_in_threadpool``, so the
    event loop stays free to keep pushing updates for the whole run. An
    ``async def`` here would be the opposite of what is wanted -- it would be
    awaited on the loop and block every connected client for the entire lecture.

    Every state change reaches subscribers through :func:`_set_state`, which
    marshals onto the event loop itself, so this thread never has to know a loop
    exists.

    Args:
        course_code: Sanitised course unit code.
        language: Two-letter spoken-language code.
        stop_event: The event ``POST /api/stop`` sets to end the recording
            phase. Passed in rather than read from the module global so that
            this run can only ever be stopped by the request that started it --
            a stale event from a previous run cannot be re-armed into a new one.
        resume: Forwarded verbatim to ``run_pipeline``. It is the caller's
            opt-in to trust the checkpoints and the existing ``raw_audio.wav``,
            and dropping it on the floor here would make the handler's 409
            bypass a silent clobber instead of a recovery.
        skipping_capture: Whether the handler has already established that the
            run will resume from audio on disk. Only used to decide whether a
            capture device is needed; ``resume`` remains the pipeline's own
            authority on whether to read checkpoints.
        domain: Domain profile key forwarded verbatim to ``run_pipeline`` and
            on to ``generate_notes``, where it selects the system prompt and
            slide-extraction prompt. Normalised to lower case by the handler
            so ``"Software"`` from a hand-rolled client still matches the
            ``"software"`` profile key.
        capture_mode: Capture source, ``"fullscreen"`` or ``"window"``.
            Forwarded verbatim to ``run_pipeline``; ``"fullscreen"`` (or any
            unrecognised value) records the primary monitor.
        target_window: Window title for the vision daemon when
            ``capture_mode == "window"``, otherwise ignored. Forwarded
            verbatim so the decision happens in one place, inside
            ``run_pipeline``.

    Every exit path releases ``_run_lock``; a run that raised while holding it
    would leave the server permanently refusing new requests. The same applies
    to the stop event: leaving a *set* event published would make the next
    recording end the instant it started.

    ``confirm_overwrite`` is ``True`` and that is not a reversal of the safe
    default it looks like. The decision was already made, in the request
    handler, which refuses outright when a previous take exists -- so by the
    time this runs there is nothing left to clobber. Passing ``False`` here
    instead (as this originally did) made ``run_pipeline`` abort on *every*
    request, existing file or not, so ``/api/record`` could never begin a
    lecture at all.
    """
    global _run_active, _stop_event
    try:
        # A device index is only meaningful when capture will run. Resolving one
        # on a resume would abort recovery on the machine most likely to need
        # it -- the one whose capture device is broken -- even though
        # ``run_pipeline`` never consults ``device_id`` once it resumes from
        # audio already on disk. The placeholder is ``0`` and not ``None``
        # because ``None`` means "prompt": a browser has no terminal to answer
        # with, so a race that did make the resumed run capture would hang the
        # worker forever instead of failing loudly.
        device_id = 0 if skipping_capture else resolve_device_index()
        notes_path = run_pipeline(
            course_code,
            language,
            _make_state_callback(),
            # No terminal to prompt on, so both interactive decisions are made
            # here instead. The device is auto-selected (unless capture is
            # skipped) and the overwrite has already been vetted by the handler.
            device_id=device_id,
            confirm_overwrite=True,
            stop_event=stop_event,
            resume=resume,
            domain=domain,
            capture_mode=capture_mode,
            target_window=target_window,
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
        # Cleared, never reset: a fresh Event is minted per run, so an aborted
        # run's stop signal cannot leak into the next one and instantly end a
        # recording the operator just started. The identity check keeps a slow
        # teardown from clearing the event of a run that has already replaced it.
        if _stop_event is stop_event:
            _stop_event = None


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
# Window discovery
# ---------------------------------------------------------------------------


@app.get("/api/windows")
def list_active_windows() -> dict[str, list[str]]:
    """Enumerate the slide-capture targets for the dashboard's window picker.

    Returns the deduplicated titles of every currently visible, non-minimised
    top-level window via :func:`vision.get_active_windows`. Only titles cross
    the boundary -- window handles are not stable across sessions and process
    ids mean nothing to the operator choosing what to capture -- and the value
    feeds ``RecordRequest.target_window`` verbatim.

    Enumeration is best-effort: a failure (no display, no window-server
    access) yields an empty list rather than a 500, so a client can simply
    hide the picker when there is nothing to pick.

    Returns:
        ``{"windows": [title, ...]}`` sorted in enumeration order.
    """
    try:
        windows = get_active_windows()
    except Exception:  # noqa: BLE001 - degrading is the contract, see above
        windows = []
    return {"windows": windows}


# ---------------------------------------------------------------------------
# Run control
# ---------------------------------------------------------------------------


@app.post("/api/record")
def start_recording(payload: RecordRequest, background: BackgroundTasks) -> dict[str, Any]:
    """Start a pipeline run and return immediately.

    Args:
        payload: Course code, domain profile, language and the optional
            resume opt-in.
        background: Starlette's post-response task runner.

    Returns:
        An acknowledgement carrying the accepted course code and status. The
        run is already in flight by the time this is serialised, and any
        connected dashboard has already been pushed the run's opening state --
        the call to :func:`_set_state` below broadcasts before the response is
        even built. The status is ``recording`` for a normal run and
        ``transcribing`` for a resume that skips capture; see the note on the
        branch below for why it is not a guess.

    Raises:
        HTTPException: 409 if a run is already in progress, or if this course
            already has a recording from an earlier take today. Starting a
            second pipeline would open a second capture stream on the same
            loopback device, and two concurrent runs would interleave writes to
            ``raw_audio.wav`` and race on the same output directory. The
            overwrite case is refused rather than confirmed because a browser
            has no way to ask the operator, and an existing ``raw_audio.wav``
            is a previous lecture's only copy. A request carrying
            ``resume=true`` is the one exception: see below.
    """
    global _run_active, _stop_event
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

    # Refuse a clobber *only when there is something to clobber*. Checking
    # here rather than delegating to run_pipeline's confirm_overwrite is what
    # makes the rule legible: a caller gets 409 with a reason, instead of a run
    # that starts, records nothing, and reports an opaque "overwrite declined"
    # failure. The output directory is derived rather than created twice --
    # create_output_dir is an idempotent mkdir, and run_pipeline will call it
    # again itself.
    output_dir = create_output_dir(course_code)
    existing_take = output_dir / WAV_FILENAME
    has_take = existing_take.exists()

    # A resume request is the one caller that *wants* the existing take: the
    # audio already on disk is the lecture being recovered, so refusing it would
    # make recovery impossible. The bypass is safe only because the flag is
    # threaded all the way into run_pipeline, which skips capture whenever the
    # WAV is present and therefore never opens it for writing. If the flag were
    # dropped between here and there, this would turn a guard into a clobber --
    # which is why the wiring is tested rather than assumed.
    if has_take and not payload.resume:
        with _run_lock:
            _run_active = False
        raise HTTPException(
            status_code=409,
            detail=(
                f"A recording for {course_code} already exists at "
                f"{existing_take.name} from an earlier take today, and it is "
                "the only copy of that lecture. Move or rename the output "
                "directory before recording a new take, or re-submit with "
                "resume=true to continue it."
            ),
        )

    # Whether the run will actually open the microphone. Mirrors the condition
    # run_pipeline uses to skip capture (a real, non-empty WAV plus the opt-in),
    # so a device is only resolved for a run that will consult one, and only
    # "recording" is announced for a run that will capture.
    #
    # The size is read defensively: this runs while the run slot is already
    # held, so a FileNotFoundError escaping here would answer 500 *and* strand
    # the slot, wedging every later request behind a 409 for a run that never
    # existed. A WAV that vanished between the two checks has no audio to
    # resume from, which is exactly what recording from scratch does about it.
    skipping_capture = False
    if payload.resume and has_take:
        try:
            skipping_capture = existing_take.stat().st_size > 0
        except OSError:
            skipping_capture = False

    # Published before the opening state on purpose. The reverse order would
    # admit a window where a client is told it is recording but /api/stop finds
    # no event to set, so a legitimate Finish Class press would be answered with
    # 400.
    stop_event = threading.Event()
    _stop_event = stop_event

    if skipping_capture:
        # No microphone is opened and no capture phase will run, so announcing
        # "recording" would be a lie the dashboard acts on -- it would offer
        # "Finish Class" for a phase that never begins. The run resumes into the
        # AI stages, so it is reported as transcribing until the pipeline emits
        # the phase it actually enters (which may be synthesizing, when the
        # diarization checkpoint is already complete). The stream therefore
        # skips "recording" entirely, and the dashboard -- which derives its
        # stage tracker from whichever token arrives -- needs no special case
        # for the fast-forward.
        _set_state(
            STATE_TRANSCRIBING,
            "Resuming interrupted class: audio already captured. Picking up "
            "at the first unfinished stage...",
            course_code=course_code,
        )
        status = STATE_TRANSCRIBING
    else:
        _set_state(STATE_RECORDING, course_code=course_code)
        status = STATE_RECORDING

    background.add_task(
        _run_in_background,
        course_code,
        payload.language.lower(),
        stop_event,
        payload.resume,
        skipping_capture,
        payload.domain.lower(),
        # Normalised here so a hand-rolled client sending "Window", or a title
        # with stray whitespace, still lands on the intended capture target.
        payload.capture_mode.lower(),
        payload.target_window.strip() if payload.target_window else None,
    )
    return {"accepted": True, "course_code": course_code, "status": status}


def _request_stop() -> threading.Event:
    """Set the recording phase's stop event and return it.

    Raises:
        HTTPException: 400 if no recording is in progress.

    **Why the status token is the gate and not the event.** The event alone
    cannot answer "is a recording happening?", because it is also briefly set
    while the run is past recording: the pipeline only reports ``transcribing``
    *after* the audio has been flushed to disk, which takes real time. Gating on
    the status means a stop is refused the instant the recording phase is
    genuinely over, instead of silently setting an event nobody will read again.

    The two reads are not atomic, and the residual window is harmless in only
    one direction. The event is always published before the state says
    ``recording``, so "recording but no event" is unreachable. The other order --
    event published, state not yet ``recording`` -- answers 400, which is
    defensible: the recording phase genuinely has not started, and the operator's
    click arrived within microseconds of the Start Capture that caused it.
    """
    if get_state()["status"] != STATE_RECORDING:
        raise HTTPException(
            status_code=400,
            detail=(
                "No recording is in progress. There is nothing to finish; the "
                "recording phase is already over or never started."
            ),
        )
    event = _stop_event
    if event is None:
        # Unreachable while the status is "recording" (see the module-level
        # note on _stop_event), but the 400 is the right answer if a future
        # change ever breaks that ordering.
        raise HTTPException(
            status_code=400,
            detail="No recording is in progress. Nothing to finish.",
        )
    # Idempotent by nature: setting an already-set event is a no-op, so a
    # double-click cannot fail or corrupt anything. The 200 below is a
    # confirmation of intent, not a promise that the phase ended instantly --
    # the audio still has to be flushed, and the client learns that from the
    # stream.
    event.set()
    return event


@app.post("/api/stop")
def stop_recording() -> dict[str, Any]:
    """End the recording phase; transcription begins from the captured audio.

    This is the web equivalent of the CLI's first Ctrl+C, and it deliberately
    does the same thing rather than aborting the run: the recorded audio is
    flushed to ``raw_audio.wav`` and the pipeline continues into transcription
    and synthesis. Stopping a capture is the normal end of a lecture, not a
    failure, so the run is not abandoned.

    Idempotent while the recording phase lasts. The event is a level-triggered
    flag, so pressing the button twice is harmless; what changes is that once
    the phase is over the endpoint answers 400, which is how a client learns
    that "Finish Class" is no longer the right verb.

    Returns:
        An acknowledgement. The client should wait for the next state change on
        ``/api/stream`` rather than treating this as "the run is finished" --
        the flush and slide-capture teardown happen after the 200.

    Raises:
        HTTPException: 400 if no recording is in progress.
    """
    event = _request_stop()
    return {
        "stopped": True,
        "status": STATE_RECORDING,
        "message": (
            "Stop requested. Finishing the recording, then transcribing. "
            "The audio is saved to raw_audio.wav before transcription begins."
        ),
        # Echoed so a client can correlate its request with this run without
        # inventing its own correlation scheme.
        "course_code": get_state()["course_code"],
        "event": id(event),
    }


@app.get("/healthz")
def healthz() -> dict[str, str]:
    """Liveness probe.

    Separate from the state stream so a monitor can distinguish "the server is
    up" from "a lecture is running" -- ``/api/stream`` is a hanging connection
    and useless as a probe, while this always answers immediately and its
    payload never changes with the pipeline.
    """
    return {"status": "ok"}


@app.get("/health")
def health() -> dict[str, str]:
    """Conventional liveness alias for :func:`healthz`.

    The ``z`` in ``/healthz`` is Google's convention, not the one every tool
    reaches for first: browser extensions, uptime checkers, and container
    healthchecks probe ``/health`` by default, and every miss lands in the
    server log as a 404. Answering both spellings is cheaper than teaching
    every client about this one, and the two are deliberately identical so a
    monitor cannot get a different answer depending on which it picked.
    """
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Session playback
# ---------------------------------------------------------------------------


def _media_url(stored_path: str | None) -> str | None:
    """Map a stored artifact path onto the ``/media`` mount, or ``None``.

    The ledger stores artifact paths relative to the process working directory
    as ``output/<SESSION_DIR>/<artifact>``, and the ``/media`` mount serves
    exactly that root, so the URL is the stored path with the root replaced by
    ``/media``. Anything that resolves outside the root -- an absolute path
    somewhere else, or a session folder the operator moved -- is not
    serviceable by the mount and is reported as ``None`` rather than a URL the
    browser would 404 on.
    """
    if not stored_path:
        return None
    try:
        relative = Path(stored_path).resolve().relative_to(MEDIA_ROOT.resolve())
    except ValueError:
        return None
    # as_posix() keeps the URL on forward slashes even on Windows.
    return "/media/" + relative.as_posix()


@app.get("/api/sessions/{session_id}")
def get_session(session_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Fetch one recorded session and the URLs of its on-disk artifacts.

    The playback UI asks for a specific lecture, so this route reads the
    ledger row by primary key and answers 404 when it does not exist. Every
    artifact URL is presence-checked against the filesystem first -- a row
    can outlive its files (a moved ``output/``, or a machine that synced the
    database but never had the session) -- so the payload never advertises a
    URL the browser would just 404 on. When the notes file exists its raw
    Markdown is returned as ``markdown_text``, so the UI can render the
    document without a second round trip.

    Args:
        session_id: Primary key of the ``Lecture`` row to load.
        db: Dependency-injected session over the ledger; overridden with a
            throwaway in-memory engine in tests.

    Returns:
        The row's identity fields, ``audio_url`` and ``slides_url_prefix``
        pointing at the ``/media`` mount when the corresponding files exist
        (``None`` otherwise), and the raw ``markdown_text`` of the notes.

    Raises:
        HTTPException: 404 when no session has this id.
    """
    lecture = db.get(Lecture, session_id)
    if lecture is None:
        raise HTTPException(
            status_code=404,
            detail=f"No recorded session with id {session_id}.",
        )

    # The audio, slides and notes live in the same session folder, so the
    # folder is derived from whichever path the row has rather than stored a
    # fourth time where it could drift out of sync with the files.
    if lecture.md_path:
        anchor = Path(lecture.md_path)
    elif lecture.audio_path:
        anchor = Path(lecture.audio_path)
    else:
        anchor = None
    session_dir = anchor.parent if anchor is not None else None

    markdown_text = None
    if lecture.md_path:
        notes_file = Path(lecture.md_path)
        if notes_file.is_file():
            try:
                markdown_text = notes_file.read_text(encoding="utf-8")
            except OSError:
                # Present but unreadable is the same answer as absent: there is
                # no document to render, and no grounds for a 500.
                markdown_text = None

    audio_url = None
    if lecture.audio_path:
        wav = Path(lecture.audio_path)
        if wav.is_file():
            audio_url = _media_url(lecture.audio_path)

    slides_url_prefix = None
    if session_dir is not None:
        slides_dir = session_dir / "slides"
        if slides_dir.is_dir():
            base = _media_url(str(slides_dir))
            if base is not None:
                slides_url_prefix = base + "/"

    return {
        "id": lecture.id,
        "course_code": lecture.course_code,
        "domain": lecture.domain,
        "session_date": lecture.session_date,
        "audio_url": audio_url,
        "slides_url_prefix": slides_url_prefix,
        "markdown_text": markdown_text,
    }


@app.get("/api/sessions")
def list_sessions(db: Session = Depends(get_db)) -> list[dict[str, Any]]:
    """Return all recorded sessions, newest first.

    Lightweight ledger view for the Library UI: only the fields needed to
    render a list (identity, timing, domain, sync state) are returned. No
    artifact bodies, no filesystem probes. Ordering is by ``created_at``
    descending so the most recent recording appears at the top.
    """
    sessions = (
        db.query(Lecture)
        .order_by(Lecture.created_at.desc(), Lecture.id.desc())
        .all()
    )
    return [
        {
            "id": session.id,
            "course_code": session.course_code,
            "domain": session.domain,
            "session_date": session.session_date,
            "sync_status": session.sync_status,
            "created_at": session.created_at.isoformat()
            if session.created_at is not None
            else None,
        }
        for session in sessions
    ]


@app.delete("/api/sessions/{session_id}")
def delete_session(session_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Delete a recorded session: remove its on-disk artifacts and the ledger row.

    The on-disk directory is derived from md_path or audio_path. If it exists,
    it is recursively removed. The database row is deleted and committed.
    Returns 404 if the session does not exist.
    """
    lecture = db.get(Lecture, session_id)
    if lecture is None:
        raise HTTPException(
            status_code=404,
            detail=f"No recorded session with id {session_id}.",
        )

    if lecture.md_path:
        anchor = Path(lecture.md_path)
    elif lecture.audio_path:
        anchor = Path(lecture.audio_path)
    else:
        anchor = None
    session_dir = anchor.parent if anchor is not None else None

    if session_dir is not None and session_dir.exists():
        shutil.rmtree(session_dir, ignore_errors=True)

    db.delete(lecture)
    db.commit()
    return {"status": "deleted", "id": session_id}


@app.post("/api/sessions/{session_id}/sync")
def retry_session_sync(session_id: int, db: Session = Depends(get_db)) -> dict[str, Any]:
    """Retry uploading the session's markdown to Google Drive.

    Returns 404 if the session does not exist. Returns 400 if md_path is missing
    or unreadable. Attempts the upload via services.gdrive.sync_markdown and
    updates sync_status to SUCCESS or FAILED accordingly.
    """
    lecture = db.get(Lecture, session_id)
    if lecture is None:
        raise HTTPException(
            status_code=404,
            detail=f"No recorded session with id {session_id}.",
        )

    if not lecture.md_path:
        raise HTTPException(
            status_code=400,
            detail="Cannot sync: md_path is missing.",
        )

    notes_file = Path(lecture.md_path)
    if not notes_file.is_file():
        raise HTTPException(
            status_code=400,
            detail="Cannot sync: markdown file does not exist on disk.",
        )

    try:
        md_text = notes_file.read_text(encoding="utf-8")
    except OSError:
        raise HTTPException(
            status_code=400,
            detail="Cannot sync: markdown file could not be read.",
        )

    try:
        from services.gdrive import sync_markdown
    except Exception as exc:  # noqa: BLE001 - degrade to FAILED
        lecture.sync_status = "FAILED"
        db.commit()
        return {"id": session_id, "sync_status": lecture.sync_status}

    try:
        synced = sync_markdown(lecture.course_code, lecture.session_date, md_text)
    except Exception:  # noqa: BLE001
        synced = False

    lecture.sync_status = "SUCCESS" if synced else "FAILED"
    db.commit()
    return {"id": session_id, "sync_status": lecture.sync_status}


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
