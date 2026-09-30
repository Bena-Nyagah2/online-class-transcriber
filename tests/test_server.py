"""Behavioural tests for the Phase 7 HTTP front end (``server.py``).

Scope
-----
The server's job is narrow and mostly about *plumbing correctness* rather than
pipeline behaviour, which ``test_main.py`` already owns. What is genuinely
worth testing here:

1. **The API contract** -- shape and status codes of the endpoints, including
   the validation and the 409 guard.
2. **State mirroring** -- that ``APP_STATE`` tracks the pipeline's own
   callbacks, and that a failure cannot leave it stuck mid-run.
3. **The push path** -- that a state change actually reaches a connected
   client, over a real event loop, from a real worker thread.
4. **The boundary decisions** -- course-code sanitisation, and the two
   non-interactive defaults that keep a browser-initiated run from blocking on
   a prompt nobody can answer.

No audio hardware, no network, and no real pipeline run. ``run_pipeline`` is
replaced by a stub that emits a state sequence; the ASGI app is driven in
process through Starlette's ``TestClient``.

The ``TestClient`` is built *inside* each test rather than in a module-level
fixture, because ``APP_STATE`` is global and shared. A session-scoped client
would leak state between tests and make ordering significant -- the exact
coupling these tests are meant to be free of.

Why the stream tests bypass ``TestClient``
------------------------------------------
``GET /api/stream`` is an endless response by design, and Starlette's
``TestClient`` cannot consume one: it runs the ASGI app to completion and
buffers the whole body before returning, so ``client.stream(...)`` blocks
forever on a body that never ends. (Confirmed by direct experiment, not
assumed -- the hang is at the ``with client.stream(...)`` line itself.)

So the stream's *logic* is tested by driving ``server._event_stream()``
directly under ``asyncio.run``, which is the same generator object the route
returns and therefore the real thing rather than a stand-in. Its *HTTP
contract* -- media type, headers, route registration -- is tested by calling
the route function directly and inspecting the returned ``StreamingResponse``.
Together those cover everything a real client would exercise.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time

import pytest

pytest.importorskip("fastapi", reason="server.py needs the web extra installed")
pytest.importorskip("httpx", reason="TestClient is built on httpx")

from fastapi.testclient import TestClient  # noqa: E402
from starlette.responses import StreamingResponse  # noqa: E402

import server  # noqa: E402
import main as m  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state(tmp_path, monkeypatch):
    """Reset the global state and run guard around every test.

    ``APP_STATE`` and ``_run_active`` are module-level and mutated by the
    background thread, so without this a test that starts a run would make the
    next one fail with a 409 for reasons that have nothing to do with the code
    under test.

    The broadcaster is reset too, and that is not tidiness. ``asyncio.run()``
    closes the loop it was given, so a test that binds one leaves the module
    pointing at a dead loop; the next test to publish would silently take the
    ``is_closed()`` early return and assert nothing. The subscriber set is
    cleared for the same reason a leaked queue would let one test's assertions
    pass on another's client. Both are private because there is no public reset
    -- and that is fine, since resetting module-global test state is exactly
    what this fixture is for.

    ``_stop_event`` is reset for a different and sharper reason: it is a
    level-triggered flag, so one test leaving it *set* would make the next
    test's recording end instantly. That is precisely the leak the endpoint is
    written to avoid, and a fixture that did not clear it would hide the bug
    rather than catch it.

    ``create_output_dir`` is redirected into a tmp dir because the request
    handler now calls it to find out where a run *would* write, so that it can
    refuse to clobber an earlier take. That mkdir is a real side effect: without
    this, every test that POSTs ``/api/record`` would leave an empty dated
    directory in the operator's real ``output/``.
    """
    server._run_active = False
    server._stop_event = None
    server.APP_STATE.update({"status": "idle", "course_code": None, "message": ""})
    monkeypatch.setattr(server, "create_output_dir", lambda code: tmp_path)
    yield
    server._run_active = False
    server._stop_event = None
    server.APP_STATE.update({"status": "idle", "course_code": None, "message": ""})
    server._broadcaster._subscribers.clear()
    server._broadcaster._loop = None


@pytest.fixture
def client():
    """A ``TestClient`` over the real app, with the pipeline stubbed.

    Only ``run_pipeline`` and ``resolve_device_index`` are replaced. Routing,
    validation, the state machinery, and the static mount are all the genuine
    article -- stubbing those would leave nothing to test.
    """
    def _stub(course_code, language, state_callback=None, **kwargs):
        calls.append(
            {
                "course_code": course_code,
                "language": language,
                "state_callback": state_callback,
                "kwargs": kwargs,
            }
        )
        for state in ("recording", "transcribing", "synthesizing", "complete"):
            state_callback(state)
        return f"output/{course_code}_2026-09-30/{course_code}_2026-09-30.md"

    calls: list[dict] = []
    real = server.run_pipeline
    real_resolve = server.resolve_device_index
    server.run_pipeline = _stub
    server.resolve_device_index = lambda *a, **k: 0
    try:
        with TestClient(server.app) as test_client:
            test_client.calls = calls
            yield test_client
    finally:
        server.run_pipeline = real
        server.resolve_device_index = real_resolve


def _start(client, code="CRM_1106", language="en"):
    """POST /api/record and return the response.

    NOTE ``TestClient`` runs the app to completion *inside* this call, which
    includes Starlette's background tasks. So a run that blocks will block this
    call too -- anything testing concurrency has to drive the request from a
    separate thread. :func:`_start_in_thread` does that.
    """
    return client.post("/api/record", json={"course_code": code, "language": language})


def _start_in_thread(client, code="CRM_1106", language="en"):
    """POST /api/record from a worker thread; return ``(response_box, slot)``.

    Needed because a ``TestClient`` call is synchronous through to the end of
    the background task. To observe a run *while it is in flight* the request
    itself has to be in flight too, which means another thread.

    ``slot`` is a two-slot list: the response lands in ``slot[0]`` and any
    exception in ``slot[1]``, so the caller can join and then inspect either.
    """
    slot: list = [None, None]

    def _worker() -> None:
        try:
            slot[0] = _start(client, code, language)
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            slot[1] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread, slot


def _state():
    """Read the state exactly as a connected dashboard receives it.

    The SSE first frame is built from :func:`server.get_state`, so for every
    assertion about "what a client would see", this is that function -- going
    through the stream would test JSON encoding twice over.
    """
    return server.get_state()


def _payload(frame: str) -> dict:
    """Parse one SSE ``data:`` frame into its JSON object."""
    assert frame.startswith("data: "), f"not a data frame: {frame!r}"
    assert frame.endswith("\n\n"), f"frame is not SSE-terminated: {frame!r}"
    return json.loads(frame[len("data: ") : -2])


async def _take(gen, count: int, timeout: float = 5.0) -> list[str]:
    """Pull ``count`` frames off an async generator, failing loudly on a stall.

    The timeout is not paranoia. The generator's normal mode is to sit in
    ``await queue.get()`` forever, so a broadcast that never arrives looks
    exactly like one that is working -- without a deadline these tests would
    hang instead of failing, and a hanging test is a much worse thing to
    hand someone than a red one.
    """
    frames = []
    for _ in range(count):
        frames.append(await asyncio.wait_for(anext(gen), timeout=timeout))
    return frames


# ---------------------------------------------------------------------------
# 1. API contract
# ---------------------------------------------------------------------------


def test_state_starts_idle(client):
    """A fresh server reports idle with no course and no message drift."""
    body = _state()
    assert body["status"] == "idle"
    assert body["course_code"] is None


def test_state_returns_exactly_the_three_contract_keys(client):
    """The payload shape is the dashboard's contract; extra keys leak internals.

    The frontend reads ``status``/``course_code``/``message`` and nothing else,
    so a key added for debugging would be a key with no defined behaviour on
    the client side. Now doubly load-bearing: the payload is serialised
    straight into an SSE frame on every single state change.
    """
    assert set(_state()) == {
        "status",
        "course_code",
        "message",
    }


def test_record_returns_immediately_and_accepted(client):
    """The response acknowledges the start; it does not wait for the run.

    A lecture is tens of minutes. Blocking the request for that would exhaust a
    worker and trip every proxy timeout in between, so the contract is
    "started, not finished" and the client polls for the rest.
    """
    response = _start(client)
    assert response.status_code == 200
    body = response.json()
    assert body["accepted"] is True
    assert body["course_code"] == "CRM_1106"
    assert body["status"] == "recording"


def test_record_starts_the_pipeline_with_the_submitted_arguments(client):
    """Course code and language must reach ``run_pipeline`` unchanged.

    Routing guard, mirroring the CLI's language test. A dropped or
    re-derived code here is invisible until someone transcribes a Swahili
    lecture into English text, which Whisper does not flag as an error.
    """
    _start(client, "CRM_1106", "sw")
    assert client.calls[0]["course_code"] == "CRM_1106"
    assert client.calls[0]["language"] == "sw"


def test_language_is_lower_cased_before_the_decoder(client):
    """ISO-639-1 codes are lower-case; a pasted ``EN`` must still work.

    Whisper rejects a capitalised code, and a browser input has no reason to
    normalise what the operator typed.
    """
    _start(client, "CRM_1106", "EN")
    assert client.calls[0]["language"] == "en"


def test_language_defaults_to_english_when_omitted(client):
    """The field is optional, matching the CLI's default."""
    response = client.post("/api/record", json={"course_code": "CRM_1106"})
    assert response.status_code == 200
    assert client.calls[0]["language"] == m.DEFAULT_LANGUAGE


def test_empty_course_code_is_rejected(client):
    """An empty code cannot name an output directory."""
    assert _start(client, "").status_code == 422


def test_overlong_language_is_rejected(client):
    """A three-letter code is a paste accident, not an ISO-639-1 code.

    Caught at the boundary so the operator gets a 422 pointing at the field
    rather than a decoder error minutes into a recording.
    """
    assert _start(client, "CRM_1106", "eng").status_code == 422


def test_second_concurrent_run_is_refused(client):
    """Two live pipelines would fight over one capture device and one directory.

    The 409 matters more than it looks: the alternative is not a harmless
    duplicate but two recorders opening the same loopback stream and two runs
    writing ``raw_audio.wav`` in the same folder.
    """
    hold, release = threading.Event(), threading.Event()

    def _blocking(course_code, language, state_callback=None, **kwargs):
        state_callback("recording")
        hold.set()
        release.wait(timeout=10)
        state_callback("complete")
        return "notes.md"

    real = server.run_pipeline
    server.run_pipeline = _blocking
    try:
        first_thread, first_slot = _start_in_thread(client, "CRM_1106")
        assert hold.wait(timeout=10), "first run never started"
        # The slot is held, so a second request must be refused.
        second = _start(client, "CRM_1107")
        release.set()
        first_thread.join(timeout=10)
    finally:
        release.set()
        server.run_pipeline = real

    assert first_slot[1] is None, f"first run raised {first_slot[1]!r}"
    assert first_slot[0].status_code == 200
    assert second.status_code == 409
    assert "already in progress" in second.json()["detail"].lower()


def test_run_slot_is_released_after_a_successful_run(client):
    """A finished run must not leave the server refusing new requests."""
    _start(client)
    assert server._run_active is False
    assert _start(client, "CRM_1107").status_code == 200


def test_run_slot_is_released_even_when_the_pipeline_raises(client):
    """A crashing run must not wedge the server permanently.

    This is the failure mode that makes a lock dangerous: if the release were
    only on the success path, one unhandled exception would make every later
    recording attempt fail with 409, with nothing in the logs to explain it.
    """

    def _boom(course_code, language, state_callback=None, **kwargs):
        raise RuntimeError("engine exploded")

    real = server.run_pipeline
    server.run_pipeline = _boom
    try:
        # The exception surfaces to the caller: Starlette runs the background
        # task inside the request. What matters is the state afterwards.
        with pytest.raises(RuntimeError, match="engine exploded"):
            _start(client)
    finally:
        server.run_pipeline = real

    assert server._run_active is False
    assert _start(client, "CRM_1107").status_code == 200


# ---------------------------------------------------------------------------
# 2. State mirroring
# ---------------------------------------------------------------------------


def test_state_follows_the_pipeline_callbacks(client):
    """The server must report what the pipeline reports, not guess.

    The dashboard's whole accuracy rests on this: progress is read from the
    pipeline's own callback rather than inferred from elapsed time, so the two
    cannot drift apart as the pipeline changes.
    """
    _start(client)
    body = _state()
    assert body["status"] == "complete"
    assert body["course_code"] == "CRM_1106"
    assert "complete" in body["message"].lower()


def test_running_state_is_visible_before_the_run_finishes(client):
    """A dashboard must not sit on "idle" while a lecture is being recorded."""

    def _slow(course_code, language, state_callback=None, **kwargs):
        state_callback("recording")
        hold.set()
        release.wait(timeout=10)
        state_callback("complete")
        return "notes.md"

    hold, release = threading.Event(), threading.Event()
    real = server.run_pipeline
    server.run_pipeline = _slow
    try:
        thread, slot = _start_in_thread(client, "CRM_1106")
        assert hold.wait(timeout=10), "run never reached the recording state"
        body = _state()
        assert body["status"] == "recording"
        assert body["course_code"] == "CRM_1106"
        release.set()
        thread.join(timeout=10)
    finally:
        release.set()
        server.run_pipeline = real

    assert slot[1] is None, f"run raised {slot[1]!r}"


def test_failure_is_reported_as_error_not_left_mid_run(client):
    """A failed run must not leave the dashboard claiming to be transcribing.

    The engine surfaces failures as a ``PipelineError``; if the server let that
    escape without updating state, the UI would show a live spinner for a run
    that died minutes ago -- the most misleading possible outcome, because it
    looks like progress.
    """

    def _fails(course_code, language, state_callback=None, **kwargs):
        state_callback("transcribing")
        raise m.PipelineError("synthesis skipped: no transcript")

    real = server.run_pipeline
    server.run_pipeline = _fails
    try:
        _start(client)
    finally:
        server.run_pipeline = real

    body = _state()
    assert body["status"] == "error"
    assert "no transcript" in body["message"]


def test_unexpected_exception_is_reported_as_error(client):
    """A non-PipelineError must still land in state.

    ``run_pipeline`` is not exhaustive. Anything else reaching the server is a
    bug, but the dashboard's job is to stop showing false progress even so --
    the traceback is the developer's problem, the stale spinner is the
    operator's.
    """

    def _boom(course_code, language, state_callback=None, **kwargs):
        state_callback("synthesizing")
        raise RuntimeError("something unforeseen")

    real = server.run_pipeline
    server.run_pipeline = _boom
    try:
        with pytest.raises(RuntimeError):
            _start(client)
    finally:
        server.run_pipeline = real

    assert _state()["status"] == "error"


def test_status_snapshot_is_a_copy_not_the_live_dict(client):
    """Callers must not be handed a mutable reference to module state.

    ``get_state()`` returning the live dict would let a caller -- or a serialiser
    iterating it while the background thread writes -- observe or corrupt shared
    state, and the copy is what makes each poll internally consistent.
    """
    snapshot = server.get_state()
    snapshot["status"] = "tampered"
    assert server.APP_STATE["status"] != "tampered"


def test_every_reachable_state_has_a_canned_message(client):
    """No state may render as an empty message.

    The frontend prints ``message`` verbatim, so a missing entry shows the
    operator a blank card with a status badge and nothing to act on. The
    terminal states matter most: "what now?" is the whole question a failed run
    raises.
    """
    for state in (
        "idle",
        m.STATE_RECORDING,
        m.STATE_TRANSCRIBING,
        m.STATE_SYNTHESIZING,
        m.STATE_COMPLETE,
        m.STATE_ERROR,
    ):
        assert state in server._STATE_MESSAGES, f"{state} has no message"
        assert server._STATE_MESSAGES[state].strip()


def test_missing_loopback_device_does_not_wedge_the_dashboard(client):
    """A no-device machine must show "error", not stay on "recording" forever.

    Regression guard with a specific history. ``resolve_device_index`` used to
    raise ``SystemExit``, which derives from ``BaseException`` and therefore
    sailed past ``except Exception`` in ``_run_in_background``: the run lock was
    released on the way out, but the state was never updated. The POST returned
    200, and the operator was left staring at a live "Recording in progress..."
    that no longer corresponded to any process. A machine with no loopback
    device is exactly the machine that hits this, which is why it is tested
    rather than assumed.
    """
    def _no_device(*a, **k):
        raise m.PipelineError("no loopback audio devices found")

    real = server.resolve_device_index
    server.resolve_device_index = _no_device
    try:
        response = _start(client)
    finally:
        server.resolve_device_index = real

    # The request is still accepted -- the device check happens in the
    # background, after the response.
    assert response.status_code == 200
    body = _state()
    assert body["status"] == "error", "dashboard left claiming to be recording"
    assert "loopback" in body["message"].lower()
    # And the server is still usable.
    assert server._run_active is False


# ---------------------------------------------------------------------------
# 3. The HTTP boundary
# ---------------------------------------------------------------------------


def test_path_traversal_in_the_course_code_is_neutralised(client):
    """A course code from a browser must not escape the output directory.

    The code is interpolated straight into a path, so ``../../..`` would let a
    request write outside ``output/``. This is the untrusted-input boundary, and
    the sanitiser is the only thing standing between the API and the filesystem.
    """
    _start(client, "../../../etc")
    assert "/" not in client.calls[0]["course_code"]
    assert ".." not in client.calls[0]["course_code"].replace(".._", "")


def test_a_code_of_only_whitespace_is_rejected(client):
    """A course code that sanitises to nothing cannot name an output directory.

    Note the input is whitespace, not slashes. ``"///"`` sanitises to ``"_"`` --
    a *valid* name, because the sanitiser collapses unsafe runs to an
    underscore -- so a slash-only code is accepted and lands in a real directory.
    Whitespace is the case that genuinely reaches zero characters: it survives
    pydantic's ``min_length=1`` check and is only stripped afterwards, so the
    server has to catch it. Left unchecked it would create ``output/_<date>``,
    a directory shared by every such request, silently mixing lectures.
    """
    assert _start(client, " ").status_code == 422
    assert _start(client, "\t \n").status_code == 422


def test_the_run_slot_is_released_when_a_bad_code_is_rejected(client):
    """A rejected request must not leave the server permanently busy.

    The run slot is claimed before the code is validated, so the rejection path
    has to hand it back explicitly or every later request would 409 against a
    run that never started. The symptom would be a server that works exactly
    once and then refuses forever, with nothing in the logs to explain it.
    """
    _start(client, " ")
    assert server._run_active is False
    assert _start(client, "CRM_1106").status_code == 200


def test_pipeline_never_prompts_for_a_device(client):
    """The background call must carry an explicit device index.

    A browser-initiated run has no terminal. If ``device_id`` were left as
    ``None`` the pipeline would call ``select_device`` and block on an
    ``input()`` that can never be answered -- the request would hang rather than
    fail, which is far harder to diagnose.
    """
    _start(client)
    assert client.calls[0]["kwargs"]["device_id"] == 0


def test_pipeline_never_prompts_before_overwriting(client):
    """Overwrite confirmation must be pre-answered, not asked.

    Same reason as the device: a browser cannot ask, and the question would
    otherwise block the run on an ``input()`` nobody can answer.

    The answer is ``True``. This reads as a reversal of the safe default, and
    it is not, because the refusal now lives in the handler -- which rejects a
    clobber with a 409 naming the file, before any run starts. See
    ``test_a_run_is_refused_rather_than_silently_overwriting_a_previous_take``
    for the half that actually protects the earlier lecture.
    """
    _start(client)
    assert client.calls[0]["kwargs"]["confirm_overwrite"] is True


def test_a_record_request_actually_starts_a_lecture(client):
    """A run must get past the pre-flight checks and reach its recording phase.

    Written as a regression test for a real bug this suite's own stub was
    hiding. ``/api/record`` passed ``confirm_overwrite=False``, which
    ``run_pipeline`` reads as "refuse" -- unconditionally, whether or not a
    previous take exists. Every browser-initiated run therefore printed
    "Aborted -- nothing was recorded" and went straight to ``error``, and the
    stop mechanism had nothing to stop.

    It survived because every test here replaces ``run_pipeline`` with a stub
    that ignores the flag, so the suite asserted only that the right value was
    *passed*, never that it was the right value. This one calls the real
    ``run_pipeline`` and checks the run actually gets going.
    """

    def _blocking(course_code, language, state_callback=None, **kwargs):
        state_callback(m.STATE_RECORDING)
        assert kwargs["stop_event"].wait(timeout=10), "the stop event never arrived"
        state_callback(m.STATE_COMPLETE)
        return "notes.md"

    real = server.run_pipeline
    server.run_pipeline = _blocking
    try:
        thread, slot = _start_in_thread(client, "CRM_1106")
        # Wait for the state to leave "idle", not while it is already
        # "recording": _start_in_thread returns the instant the thread is
        # spawned, which can be before the handler has run at all, so a loop
        # that polls for the value it wants would read the initial "idle" and
        # exit on its first pass.
        observed = "idle"
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            observed = server.get_state()["status"]
            if observed != "idle" or slot[1] is not None:
                break
            time.sleep(0.02)
        client.post("/api/stop")
        thread.join(timeout=10)
    finally:
        server.run_pipeline = real

    assert slot[1] is None, f"the run raised {slot[1]!r}"
    assert observed == m.STATE_RECORDING, (
        f"the run never reached recording (status={observed!r}); it is being "
        "refused by a pre-flight check"
    )


def test_a_run_is_refused_rather_than_silently_overwriting_a_previous_take(
    client, tmp_path
):
    """The protection the previous test gives up, asserted directly.

    An existing ``raw_audio.wav`` is the only copy of an earlier lecture, and a
    browser cannot ask whether to replace it. The handler must refuse with a
    409 that names the file, and must not start a run that would overwrite it.

    The ``clean_state`` fixture points ``create_output_dir`` at this same
    ``tmp_path``, so the WAV planted here is exactly the one the handler looks
    for.
    """
    (tmp_path / "raw_audio.wav").write_bytes(b"RIFF" + b"\0" * 200)

    response = _start(client, "CRM_1106")

    assert response.status_code == 409
    detail = response.json()["detail"]
    assert "raw_audio.wav" in detail, detail
    assert "already exists" in detail, detail
    # The run slot must be released, or the refusal would wedge the server.
    assert server._run_active is False
    assert server.get_state()["status"] == "idle", "a refused run changed the state"


def test_a_rejected_overwrite_does_not_arm_the_stop_event(client, tmp_path):
    """A refused request must leave nothing behind for the next one.

    Both the run slot and the stop event are armed by ``/api/record``. If a
    refusal released the first but not the second, the next run would inherit
    an event and end the instant it began.
    """
    (tmp_path / "raw_audio.wav").write_bytes(b"RIFF" + b"\0" * 200)

    assert _start(client, "CRM_1106").status_code == 409
    assert server._stop_event is None


def test_a_rejected_overwrite_does_not_start_a_pipeline(client, tmp_path):
    """The 409 has to come from the handler, before any work is scheduled.

    If the check happened inside the background task instead, the request would
    answer 200 and the refusal would surface minutes later as a state change --
    which is the shape of bug this whole pair of tests exists to prevent.
    """
    (tmp_path / "raw_audio.wav").write_bytes(b"RIFF" + b"\0" * 200)

    _start(client, "CRM_1106")

    assert client.calls == [], "a pipeline ran despite the refusal"


# ---------------------------------------------------------------------------
# 4. Static shell
# ---------------------------------------------------------------------------


def test_root_serves_the_dashboard(client):
    """``/`` must return the HTML shell, not a 404."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_api_routes_are_registered_before_the_root_static_mount(client):
    """The ``/`` mount must not shadow the API.

    Load-bearing ordering, not a style choice: Starlette matches routes in
    registration order and a mount at ``/`` matches every path, so registering
    it before the endpoints would make ``/api/stream`` return a static 404. The
    symptom is a dashboard that loads and then silently never updates -- which
    looks exactly like a broken backend.

    Asserted structurally rather than over HTTP because the stream never
    completes a response body (see the module docstring); what matters is the
    *order*, and the order is directly readable from ``app.routes``.
    """
    from starlette.routing import Mount

    routes = server.app.routes
    # Matched by name rather than by path: Starlette rewrites a mount at "/" to
    # "/{path}", so testing for a literal "/" finds nothing.
    mount_index = next(
        i
        for i, r in enumerate(routes)
        if isinstance(r, Mount) and r.name == "static"
    )
    api = {
        r.path for r in routes[:mount_index] if getattr(r, "path", "").startswith("/api")
    }
    assert api == {"/api/record", "/api/stream", "/api/stop"}

    # And the two routes that *can* complete a response still answer normally.
    assert client.get("/healthz").status_code == 200
    assert client.get("/").status_code == 200


def test_healthz_is_independent_of_pipeline_state(client):
    """Liveness must not change when a lecture is running.

    A monitor polling a payload that flips to "transcribing" cannot tell "the
    server is down" from "the server is busy", and will page for the second one.
    The stream is no substitute: it is a hanging connection, useless as a probe.
    """
    assert client.get("/healthz").json() == {"status": "ok"}
    _start(client)
    assert client.get("/healthz").json() == {"status": "ok"}


def test_dashboard_streams_instead_of_polling(client):
    """The shell must consume the endpoint the server actually exposes.

    Checked against the shipped HTML rather than a copy, so a rename on either
    side fails here instead of leaving a UI that polls a 404 forever. Asserts the
    negative too: a leftover ``setInterval`` would mean the dashboard is *also*
    still polling, and the two together are a defect neither check alone
    would catch.
    """
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "/api/stream" in html
    assert "EventSource" in html
    assert "setInterval" not in html
    assert "/api/status" not in html


def test_dashboard_reports_a_lost_connection(client):
    """A dropped stream must be visible, not silently leave a stale card.

    ``EventSource`` reconnects on its own, which is exactly why this needs an
    explicit test: the recovery is invisible by design, so without an indicator
    the operator sees an unchanged card and has no way to tell a live lecture
    from a dashboard that stopped hearing about one minutes ago.
    """
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "es.onerror" in html
    assert "EventSource.CLOSED" in html

    # The subtle half. EventSource retries transport failures on its own, so
    # calling close() on a transient error would cancel the browser's own
    # reconnection and turn a self-healing blip into a permanent outage. The
    # close() is allowed only on the CLOSED branch.
    handler = html.split("es.onerror = function () {", 1)[1]
    handler = handler.split("\n    };", 1)[0]
    closed_branch, transient_branch = handler.split("} else {", 1)
    assert "es.readyState === EventSource.CLOSED" in closed_branch
    assert "close()" in closed_branch
    assert "close()" not in transient_branch


def test_the_control_deck_posts_to_both_endpoints(client):
    """The two buttons must hit the two endpoints the server actually exposes.

    Same rationale as the stream check above, and it covers the button that
    starts a lecture as well as the one that ends it -- a rename on either side
    would otherwise leave a control that fails only once pressed, mid-lecture.
    """
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "/api/record" in html
    assert "/api/stop" in html


def test_the_stop_button_cannot_submit_the_form(client):
    """It lives inside the form, so a missing ``type`` would fire a *start*.

    A submit-typed button in this position raises the form's submit handler,
    which POSTs ``/api/record``. The operator pressing "Finish Class" would
    then get a 409 -- and, worse, the run they are trying to end would be
    reported as already in progress, which is a maddening thing to be told
    while holding the button that was meant to stop it.
    """
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    button = html.split('id="stop-button"', 1)[1].split(">", 1)[0]
    assert 'type="button"' in button, f"stop button tag is: {button.strip()!r}"


def test_the_dashboard_collects_a_course_code_and_a_language(client):
    """Both fields the pipeline needs, with the course code bounded.

    ``maxlength`` is not cosmetic: the value is interpolated into an output
    directory name, and the server's own bound is 64. An unbounded input is a
    filesystem DoS at best.
    """
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert 'placeholder="e.g., CRM_3102"' in html
    assert 'maxlength="64"' in html

    # A <select> rather than free text, so an unsupported code cannot be typed.
    # Scanned *backwards* from the id to the opening angle bracket: splitting
    # forward from 'id="language"' would only ever see the attributes that come
    # after it, and the tag name is always before, so the naive version of this
    # assertion passes on an <input> and fails on a real <select>.
    anchor = html.index('id="language"')
    tag = html[:anchor][html[:anchor].rindex("<"):]
    assert tag.startswith("<select"), f"language control is: {tag!r}"
    # And it must actually offer choices, or it is a text box wearing a costume.
    options = html.split("<select", 1)[1].split("</select>", 1)[0]
    assert options.count("<option") >= 2, "the language select has no options"


# ---------------------------------------------------------------------------
# 5. The push path (Server-Sent Events)
# ---------------------------------------------------------------------------
# The tests above prove the state machine is right. These prove the state
# actually *travels* -- a separate question, and the one with a real failure
# mode: `asyncio.Queue` is not thread-safe, so a broadcast published from the
# pipeline worker by the obvious means looks fine in development and silently
# never wakes a connected browser in production.


def test_stream_endpoint_returns_an_event_stream(client):
    """The HTTP contract, checked without consuming the endless body.

    ``TestClient`` blocks forever on a response that never ends, so the route
    function is called directly and the object it returns inspected. That object
    *is* what Starlette hands the ASGI layer, so this is the real contract and
    not a stand-in.
    """
    response = server.stream_events()
    assert isinstance(response, StreamingResponse)
    assert response.media_type == "text/event-stream"

    headers = {k.lower(): v for k, v in response.headers.items()}
    assert headers["cache-control"] == "no-cache"
    # Without this a reverse proxy batches frames, which reinstates exactly the
    # delay SSE exists to remove -- while the app still looks perfect in dev,
    # because nothing sits in front of it there.
    assert headers["x-accel-buffering"] == "no"


def test_status_endpoint_is_gone(client):
    """The polled endpoint was removed, and its absence is a 404, not a hang.

    Worth pinning explicitly: a dashboard still polling ``/api/status`` would now
    get a fast, plausible-looking 404 forever, and the only symptom would be a
    status card frozen on "Idle".
    """
    assert client.get("/api/status").status_code == 404


def test_frames_are_well_formed_sse(client):
    """Every frame must parse as ``data: <json>`` terminated by a blank line.

    The blank line is not decoration: it is the frame delimiter the browser
    splits on. Omit it and ``EventSource`` buffers the event forever, so the
    dashboard would show exactly one state and then nothing at all.
    """
    async def scenario():
        server._broadcaster.bind_loop(asyncio.get_running_loop())
        gen = server._event_stream()
        try:
            await anext(gen)
            server._set_state(m.STATE_TRANSCRIBING, course_code="CRM_1106")
            return (await _take(gen, 1))[0]
        finally:
            await gen.aclose()

    frame = asyncio.run(scenario())
    assert frame.startswith("data: ")
    assert frame.endswith("\n\n")
    assert _payload(frame) == {
        "status": m.STATE_TRANSCRIBING,
        "course_code": "CRM_1106",
        "message": server._STATE_MESSAGES[m.STATE_TRANSCRIBING],
    }


def test_a_new_client_is_told_the_current_state_immediately(client):
    """A client connecting mid-lecture learns the state at once.

    Without the leading snapshot, a dashboard opened during transcription would
    sit on the markup's hardcoded "Idle" until the *next* transition -- which for
    a two-hour lecture can be most of an hour. The cost is one extra frame per
    connection, which is nothing next to that.
    """
    async def scenario():
        server._broadcaster.bind_loop(asyncio.get_running_loop())
        server._set_state(m.STATE_SYNTHESIZING, course_code="CRM_3102")
        gen = server._event_stream()
        try:
            return (await _take(gen, 1))[0]
        finally:
            await gen.aclose()

    payload = _payload(asyncio.run(scenario()))
    assert payload["status"] == m.STATE_SYNTHESIZING
    assert payload["course_code"] == "CRM_3102"


def test_state_changes_reach_a_connected_client_from_a_worker_thread(client):
    """The load-bearing test: a worker-thread publish wakes a parked client.

    This is the entire reason for the ``call_soon_threadsafe`` indirection. The
    publishing thread has no running event loop -- it is a ``BackgroundTasks``
    threadpool worker, and ``asyncio.get_running_loop()`` raises there. Calling
    ``queue.put_nowait()`` straight from it would still update the queue's
    internal deque, but the waiter is only signalled through ``call_soon``, and
    ``call_soon`` from a foreign thread does *not* wake the loop's selector, so
    the frame sits in the ready queue until something else wakes the loop.

    Two details make this test actually able to see that, and both were arrived
    at by watching it fail to:

    * **Park the loop first.** With the worker racing the loop, the stranded
      callback often ran while the loop was still spinning, and a mutation
      swapping ``call_soon_threadsafe`` for a direct call sailed through.
    * **Assert latency, not arrival.** ``asyncio.wait_for`` installs its own
      timer, and that timer wakes the loop -- which then flushes the very ready
      queue the mutation was relying on staying untouched. So the mutant *does*
      eventually deliver, roughly ``timeout`` seconds late. A pass/fail on
      "did the frame arrive" therefore cannot distinguish a correct
      implementation from a broken one; measuring how long it took can, and it
      matches production, where the loop is idle in ``epoll()`` for minutes
      between state changes.
    """
    states = (m.STATE_TRANSCRIBING, m.STATE_SYNTHESIZING, m.STATE_COMPLETE)

    async def scenario():
        server._broadcaster.bind_loop(asyncio.get_running_loop())
        gen = server._event_stream()
        await anext(gen)  # drain the snapshot; the subscription is now live

        # Prove the loop has genuinely parked before anything is published.
        # A timer cannot fire until the loop has sat in its selector, so this is
        # a receipt rather than a hope. (An earlier version used a blocking
        # `Event.wait()` here, which froze the loop thread instead of parking
        # it -- so the worker published into a *blocked* loop, the loop resumed
        # and drained the stranded callbacks, and the mutant sailed through.)
        parked = asyncio.Event()
        asyncio.get_running_loop().call_later(0.2, parked.set)
        await parked.wait()

        clock: dict[str, float] = {}

        def worker():
            # Long enough that the loop is certainly back in select() by now.
            time.sleep(0.15)
            clock["t0"] = time.perf_counter()
            for state in states:
                server._set_state(state, course_code="CRM_1106")

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
        try:
            # Generous on purpose: the timeout is a deadlock backstop, not the
            # assertion. Arriving late still arrives, and must be measured.
            frames = await _take(gen, len(states), timeout=6.0)
            clock["elapsed"] = time.perf_counter() - clock["t0"]
            return frames, clock["elapsed"]
        finally:
            await gen.aclose()
            thread.join(timeout=5)

    frames, elapsed = asyncio.run(scenario())
    assert [_payload(f)["status"] for f in frames] == list(states)
    assert all(_payload(f)["course_code"] == "CRM_1106" for f in frames)
    # A correct cross-thread hop wakes the selector, so this is sub-millisecond.
    # A bare put_nowait from a worker thread leaves the frames waiting for
    # whatever timer fires next, which is what the timeout above is.
    assert elapsed < 1.0, (
        f"frames took {elapsed:.2f}s to reach the client; a cross-thread "
        f"put_nowait only lands when something else happens to wake the loop"
    )


def test_every_connected_client_receives_the_same_broadcast(client):
    """Two dashboards, two identical streams.

    The fan-out is a loop over the subscriber set, so a failure while queueing to
    one departed client must not stop delivery to the rest -- otherwise a
    browser tab that vanished mid-lecture silently blinds every other tab.
    """
    async def scenario():
        server._broadcaster.bind_loop(asyncio.get_running_loop())
        first, second = server._event_stream(), server._event_stream()
        await anext(first)
        await anext(second)
        try:
            server._set_state(m.STATE_SYNTHESIZING, course_code="CRM_1106")
            await asyncio.sleep(0)  # let the scheduled delivery land
            return (
                await asyncio.wait_for(anext(first), timeout=5.0),
                await asyncio.wait_for(anext(second), timeout=5.0),
            )
        finally:
            await first.aclose()
            await second.aclose()

    first_frame, second_frame = asyncio.run(scenario())
    assert _payload(first_frame) == _payload(second_frame)
    assert _payload(first_frame)["status"] == m.STATE_SYNTHESIZING


def test_closing_a_stream_unsubscribes_it(client):
    """A departed client must not leak.

    Reached via the generator's ``finally``, so it runs on normal completion and
    on the ``GeneratorExit`` raised by ``aclose()`` alike. Missing it would grow
    the subscriber set without bound for the life of the process, and every
    publish would keep building payloads for a client that is never reading.
    """
    async def scenario():
        server._broadcaster.bind_loop(asyncio.get_running_loop())
        gen = server._event_stream()
        await anext(gen)
        peak = server._broadcaster.subscriber_count
        await gen.aclose()
        return peak, server._broadcaster.subscriber_count

    peak, after = asyncio.run(scenario())
    assert peak == 1
    assert after == 0


def test_a_slow_client_keeps_the_newest_state_not_a_backlog(client):
    """A client that stops reading must not accumulate a replay log.

    Bounded per-subscriber queue, oldest dropped when it fills. A lecture emits a
    handful of states, so this is not about volume; it is about a client that
    stalls for a minute coming back to report a phase from before the stall. Only
    the newest state is still true.
    """
    flood = server._QUEUE_SIZE + 5

    async def scenario():
        server._broadcaster.bind_loop(asyncio.get_running_loop())
        queue = server._broadcaster.subscribe()
        try:
            for i in range(flood):
                server._broadcaster.publish_threadsafe({"n": i})
            await asyncio.sleep(0)  # drain the scheduled deliveries
            seen = []
            while not queue.empty():
                seen.append(queue.get_nowait()["n"])
            return seen
        finally:
            server._broadcaster.unsubscribe(queue)

    seen = asyncio.run(scenario())
    assert len(seen) <= server._QUEUE_SIZE
    assert seen[-1] == flood - 1, "newest state was not the one that survived"


def test_an_idle_stream_emits_a_keepalive(client, monkeypatch):
    """A connection with no traffic must still say something.

    Proxies close idle connections, usually silently, and the client's next
    symptom is an ``onerror`` on a dashboard that looks perfectly fine. A comment
    frame is invisible to ``EventSource`` -- it does not fire ``onmessage`` -- so
    this costs the UI nothing.
    """
    monkeypatch.setattr(server, "SSE_KEEPALIVE_SECONDS", 0.05)

    async def scenario():
        server._broadcaster.bind_loop(asyncio.get_running_loop())
        gen = server._event_stream()
        try:
            return await _take(gen, 2)
        finally:
            await gen.aclose()

    snapshot, keepalive = asyncio.run(scenario())
    assert _payload(snapshot)["status"] == "idle"
    assert keepalive == ": keepalive\n\n"
    assert not keepalive.startswith("data:"), "keepalive must not look like state"


def test_shutdown_releases_every_parked_stream(client):
    """A client must not be left waiting on a queue nobody will ever fill.

    Without the shutdown sentinel the generators sit in ``await queue.get()``
    with no pending work, and uvicorn's graceful shutdown waits on them. The
    deadline here is what makes a missing sentinel fail red instead of hanging
    the suite.
    """
    async def scenario():
        server._broadcaster.bind_loop(asyncio.get_running_loop())
        gen = server._event_stream()
        await anext(gen)
        parked = asyncio.create_task(anext(gen))
        await asyncio.sleep(0)  # let it reach queue.get()
        try:
            server._broadcaster.shutdown()
            await asyncio.wait_for(parked, timeout=5.0)
        except StopAsyncIteration:
            return "exited"
        return "still-open"

    assert asyncio.run(scenario()) == "exited"


def test_publishing_with_no_loop_bound_is_harmless(client):
    """A run started before any client connects must not blow up.

    The POST publishes ``recording`` synchronously, and in a deployment with no
    dashboard open nothing ever binds a loop at all. "Nobody is listening" has
    to mean a no-op, never an exception that aborts the run itself.
    """
    server._broadcaster._loop = None
    server._set_state(m.STATE_RECORDING, course_code="CRM_1106")
    assert _state()["status"] == m.STATE_RECORDING


def test_publishing_after_the_loop_closes_is_harmless(client):
    """The final state of a shutting-down server must still be recorded.

    The reverse race of the previous test: a background task finishing while
    uvicorn is tearing down. ``publish_threadsafe`` sees ``is_closed()`` and
    returns -- and the state write, which is the durable part, has already
    happened either way.
    """
    holder = {}

    async def bind_and_close():
        holder["loop"] = asyncio.get_running_loop()
        server._broadcaster.bind_loop(holder["loop"])

    asyncio.run(bind_and_close())  # the loop is now closed
    assert holder["loop"].is_closed()

    server._set_state(m.STATE_COMPLETE)
    assert _state()["status"] == m.STATE_COMPLETE


def test_post_publishes_the_whole_run_sequence_to_the_broadcaster(client):
    """A real POST must push every state the pipeline reports, in order.

    The end-to-end version of the push-path claim: the ``TestClient`` runs the
    app to completion, so the background task really does fire and really does
    publish through the loop bound by the app's lifespan. Consecutive duplicates
    are collapsed before asserting, because the handler sets ``recording``
    itself and the pipeline then announces ``recording`` again -- both are
    legitimate, and only the order and the absence of gaps is the contract.
    """
    published: list[dict] = []
    real_deliver = server._broadcaster._deliver

    def _capture(payload):
        published.append(payload)
        real_deliver(payload)

    server._broadcaster._deliver = _capture
    try:
        _start(client, "CRM_1106", "sw")
    finally:
        server._broadcaster._deliver = real_deliver

    seen = [p["status"] for p in published]
    collapsed = [s for i, s in enumerate(seen) if i == 0 or s != seen[i - 1]]
    assert collapsed == [
        m.STATE_RECORDING,
        m.STATE_TRANSCRIBING,
        m.STATE_SYNTHESIZING,
        m.STATE_COMPLETE,
    ]
    # Every published payload is a complete, self-consistent snapshot, not the
    # bare token -- a client that received only a status would have to guess.
    for payload in published:
        assert set(payload) == {"status", "course_code", "message"}
        assert payload["course_code"] == "CRM_1106"
        assert payload["message"]


def test_publishing_happens_while_the_state_lock_is_held(client):
    """A state snapshot must be handed to the broadcaster before the lock drops.

    Mutation testing turned this into a real bug rather than a hypothetical one.
    ``_set_state`` originally wrote under ``_state_lock``, copied, released, and
    only then called ``publish_threadsafe`` -- which left a window in which the
    request thread could write and broadcast ``recording``, the pipeline thread
    could write and broadcast ``transcribing``, and the first thread could then
    get around to broadcasting ``recording``. A client would watch a lecture go
    *back* a phase, with nothing in the logs to explain it.

    The fix was to schedule inside the critical section, which is free:
    ``call_soon_threadsafe`` appends to a deque and pokes the self-pipe, neither
    of which blocks. The justification that it might block on a slow subscriber
    was simply wrong -- delivery happens later, on the event loop.

    Asserted by probing the lock at the moment of publication: if publishing
    happens inside the critical section the lock is held and a non-blocking
    acquire fails; if it happens after, the lock is free and the acquire
    succeeds. (The first attempt at this test held the lock from the test thread
    and checked that a publisher stayed blocked -- which passes either way,
    because the lock also guards the *write*, so it proved nothing.)
    """
    observed: dict[str, bool] = {}
    real_publish = server._broadcaster.publish_threadsafe

    def _probe(payload):
        acquired = server._state_lock.acquire(blocking=False)
        if acquired:
            server._state_lock.release()
            observed["lock_held"] = False
        else:
            observed["lock_held"] = True
        real_publish(payload)

    server._broadcaster.publish_threadsafe = _probe
    try:
        server._broadcaster.bind_loop(asyncio.new_event_loop())
        server._set_state(m.STATE_RECORDING, course_code="CRM_1106")
    finally:
        server._broadcaster.publish_threadsafe = real_publish
        server._broadcaster._loop = None

    assert observed.get("lock_held") is True, (
        "the frame was scheduled after the lock was dropped, so two publishers "
        "can interleave and a client can see a phase arrive out of order"
    )


# ---------------------------------------------------------------------------
# 5. Stopping a recording
# ---------------------------------------------------------------------------
# The CLI ends a recording with Ctrl+C, which raises KeyboardInterrupt on
# whichever thread owns the terminal. A browser owns no terminal and cannot
# send a signal, so the same outcome is driven by a threading.Event that
# run_status_loop watches.
#
# What is worth pinning down here is not the status code -- it is that the
# stop is *delivered to the engine* and that a stopped run carries on into
# transcription. An endpoint that answered 200 while the pipeline sat in its
# capture loop forever would pass every status-code assertion and still be
# useless, so several of these tests assert the run's fate rather than the
# response.


def test_stop_is_refused_when_no_run_has_started(client):
    """400, not 500 and not a silent success.

    There is no event to set, and pretending otherwise would leave the client
    believing a capture had been ended when none existed.
    """
    response = client.post("/api/stop")

    assert response.status_code == 400
    assert "no recording is in progress" in response.json()["detail"].lower()


def test_stop_is_refused_after_the_run_finished(client):
    """A stale dashboard must not be able to stop a run that already completed."""
    _start(client)

    response = client.post("/api/stop")

    assert response.status_code == 400


def test_stop_sets_the_event_and_answers_200(client):
    """The core contract: a live recording is stopped, and told so."""
    recording = threading.Event()
    seen: dict = {}

    def _stub(course_code, language, state_callback=None, **kwargs):
        seen["stop_event"] = kwargs["stop_event"]
        state_callback(m.STATE_RECORDING)
        recording.set()
        assert kwargs["stop_event"].wait(timeout=10), "the stop never arrived"
        return "notes.md"

    real = server.run_pipeline
    server.run_pipeline = _stub
    try:
        thread, slot = _start_in_thread(client, "CRM_1106")
        assert recording.wait(timeout=10), "the run never started recording"
        response = client.post("/api/stop")
        thread.join(timeout=10)
    finally:
        server.run_pipeline = real

    assert slot[1] is None, f"the run raised {slot[1]!r}"
    assert response.status_code == 200
    assert response.json()["stopped"] is True
    # The endpoint must set the *same* object the pipeline is watching, not a
    # copy or a fresh event. A copy would be set and never observed, and the
    # recording would run to the end of time while the dashboard reported
    # success.
    assert seen["stop_event"].is_set()


def test_stopping_lets_the_pipeline_proceed_to_transcription(client):
    """The web stop is the CLI's first Ctrl+C, not an abort.

    Asserts the whole point of the feature rather than just the HTTP 200: a
    stopped run continues into ``transcribing`` and ``complete``. The recording
    phase is the normal end of a lecture, so the captured audio must still be
    transcribed and synthesised.
    """
    sequence: list[str] = []
    recording = threading.Event()

    def _stub(course_code, language, state_callback=None, **kwargs):
        sequence.append(m.STATE_RECORDING)
        state_callback(m.STATE_RECORDING)
        recording.set()
        assert kwargs["stop_event"].wait(timeout=10), "the stop never arrived"
        for state in (m.STATE_TRANSCRIBING, m.STATE_COMPLETE):
            sequence.append(state)
            state_callback(state)
        return "notes.md"

    real = server.run_pipeline
    server.run_pipeline = _stub
    try:
        thread, slot = _start_in_thread(client, "CRM_1106")
        assert recording.wait(timeout=10), "the run never started recording"
        response = client.post("/api/stop")
        thread.join(timeout=10)
    finally:
        server.run_pipeline = real

    assert slot[1] is None, f"the run raised {slot[1]!r}"
    assert response.status_code == 200
    assert sequence == [m.STATE_RECORDING, m.STATE_TRANSCRIBING, m.STATE_COMPLETE]
    assert _state()["status"] == m.STATE_COMPLETE
    assert server._run_active is False


def test_stop_is_refused_once_the_recording_phase_is_over(client):
    """400 as soon as the phase is past, so the client can stop showing the button.

    The pipeline reports ``transcribing`` only *after* the audio is flushed, so
    there is a real interval in which the run is alive but recording is over.
    Gating on the status token (rather than on the event merely existing) is
    what makes "Finish Class" mean "finish the class" instead of "set a flag
    nobody reads any more".
    """
    started = threading.Event()
    release = threading.Event()

    def _stub(course_code, language, state_callback=None, **kwargs):
        state_callback(m.STATE_RECORDING)
        state_callback(m.STATE_TRANSCRIBING)
        started.set()
        release.wait(timeout=10)
        state_callback(m.STATE_COMPLETE)
        return "notes.md"

    real = server.run_pipeline
    server.run_pipeline = _stub
    try:
        thread, _ = _start_in_thread(client, "CRM_1106")
        assert started.wait(timeout=10), "the run never reached transcription"
        response = client.post("/api/stop")
        release.set()
        thread.join(timeout=10)
    finally:
        release.set()
        server.run_pipeline = real

    assert response.status_code == 400
    assert _state()["status"] == m.STATE_COMPLETE


def test_stop_is_idempotent_while_the_recording_phase_lasts(client):
    """A double-click on Finish Class must not error or corrupt the run.

    The event is a level-triggered flag, so setting it twice is a no-op. The
    button is not disabled while a stop is in flight -- the pipeline still has
    to flush the WAV and stop the slide catcher before it reports the next
    state, and that takes real time during which the operator has no other way
    to tell the app the second click was not a second request. The stub below
    holds the run inside the recording phase after the first stop precisely to
    keep the second press in that window; a stub that finished instantly would
    be answering a different question (and would correctly get a 400).
    """
    recording = threading.Event()
    flushing = threading.Event()
    release = threading.Event()

    def _stub(course_code, language, state_callback=None, **kwargs):
        state_callback(m.STATE_RECORDING)
        recording.set()
        assert kwargs["stop_event"].wait(timeout=10), "the stop never arrived"
        flushing.set()
        release.wait(timeout=10)
        state_callback(m.STATE_COMPLETE)
        return "notes.md"

    real = server.run_pipeline
    server.run_pipeline = _stub
    try:
        thread, slot = _start_in_thread(client, "CRM_1106")
        assert recording.wait(timeout=10), "the run never started recording"
        first = client.post("/api/stop")
        assert flushing.wait(timeout=10), "the run never began flushing"
        second = client.post("/api/stop")
        release.set()
        thread.join(timeout=10)
    finally:
        release.set()
        server.run_pipeline = real

    assert slot[1] is None, f"the run raised {slot[1]!r}"
    assert first.status_code == 200
    assert second.status_code == 200
    assert first.json()["event"] == second.json()["event"], (
        "the two stops targeted different events, so only one of them was real"
    )


def test_the_stop_event_is_published_before_the_start_request_returns(client):
    """A stop must be answerable the instant the start request returns.

    ``BackgroundTasks`` do not begin until the response has been returned, so
    an event minted inside the task would leave a window in which ``/api/stop``
    finds nothing to set and answers 400 to a Finish Class press that is
    perfectly legitimate. Probed at the top of the stub -- the first thing the
    background task executes -- so it fails if the endpoint ever moves the mint
    back inside the task.
    """
    observed: dict = {}

    def _stub(course_code, language, state_callback=None, **kwargs):
        observed["already_published"] = server._stop_event is kwargs["stop_event"]
        state_callback(m.STATE_RECORDING)
        return "notes.md"

    real = server.run_pipeline
    server.run_pipeline = _stub
    try:
        _start(client, "CRM_1106")
    finally:
        server.run_pipeline = real

    assert observed.get("already_published") is True, (
        "the stop event did not exist until the background task started, so a "
        "stop pressed in that window would be refused"
    )


def test_the_stop_event_is_cleared_when_the_run_ends(client):
    """A finished run must not leave a set event pointing at the next one.

    Setting the event once and never clearing it would make every subsequent
    recording end the instant it began, which is a far nastier failure than a
    button that does nothing -- and one that only shows up on the *second* run.
    """
    assert server._stop_event is None

    _start(client, "CRM_1106")

    assert server._stop_event is None


def test_a_previous_runs_stop_event_cannot_end_the_next_recording(client):
    """Regression test for the level-triggered-flag leak.

    Stop a run, then start another and check its event is unset and distinct.
    Asserting the flag is ``None`` afterwards is not quite the same thing: the
    real risk is a *surviving* event that the next run adopts, and that is only
    observable by looking at the second run's own event object.
    """
    events: list = []

    def _stub(course_code, language, state_callback=None, **kwargs):
        events.append(kwargs["stop_event"])
        state_callback(m.STATE_RECORDING)
        return "notes.md"

    real = server.run_pipeline
    server.run_pipeline = _stub
    try:
        _start(client, "CRM_1106")
        assert client.post("/api/stop").status_code == 400  # already finished
        _start(client, "CRM_1107")
    finally:
        server.run_pipeline = real

    assert len(events) == 2
    assert events[0] is not events[1], "both runs shared one stop event"
    assert not events[1].is_set(), (
        "the second run inherited a set stop event and would have ended the "
        "instant it started recording"
    )


def test_the_stop_endpoint_publishes_no_state_change(client):
    """Stopping is an instruction, not a state transition.

    A frame published here would be a ``recording`` snapshot arriving *after*
    the operator asked to stop -- so a client rendering frames in order would
    show the lecture going back to recording, the same backwards-phase defect
    the ``_set_state`` lock fix exists to prevent. The client learns the phase
    ended from the next real transition instead.
    """
    failures: list[str] = []

    async def _check() -> None:
        loop = asyncio.get_running_loop()
        server._broadcaster.bind_loop(loop)
        queue = server._broadcaster.subscribe()
        try:
            server._stop_event = threading.Event()
            # Reaching "recording" necessarily publishes a frame -- the gate in
            # _request_stop reads that state. It is the *stop* that must add
            # nothing, so the setup frame is drained and only what arrives
            # afterwards is interesting. Two turns of the loop: one for
            # call_soon_threadsafe to hand the work over, one for _deliver to run.
            server._set_state(m.STATE_RECORDING, course_code="CRM_1106")
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            while not queue.empty():
                queue.get_nowait()

            server.stop_recording()
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            if not queue.empty():
                failures.append(
                    f"the stop endpoint published {queue.qsize()} frame(s)"
                )
        finally:
            server._broadcaster.unsubscribe(queue)
            server._stop_event = None

    asyncio.run(_check())

    assert not failures, failures[0]
