"""Behavioural tests for the Phase 7 HTTP front end (``server.py``).

Scope
-----
The server's job is narrow and mostly about *plumbing correctness* rather than
pipeline behaviour, which ``test_main.py`` already owns. What is genuinely
worth testing here:

1. **The API contract** -- shape and status codes of the two endpoints,
   including the validation and the 409 guard.
2. **State mirroring** -- that ``APP_STATE`` tracks the pipeline's own
   callbacks, and that a failure cannot leave it stuck mid-run.
3. **The boundary decisions** -- course-code sanitisation, and the two
   non-interactive defaults that keep a browser-initiated run from blocking on
   a prompt nobody can answer.

No audio hardware, no network, and no real pipeline run. ``run_pipeline`` is
replaced by a stub that emits a state sequence; the ASGI app is driven in
process through Starlette's ``TestClient``.

The ``TestClient`` is built *inside* each test rather than in a module-level
fixture, because ``APP_STATE`` is global and shared. A session-scoped client
would leak state between tests and make ordering significant -- the exact
coupling these tests are meant to be free of.
"""

from __future__ import annotations

import pytest

pytest.importorskip("fastapi", reason="server.py needs the web extra installed")
pytest.importorskip("httpx", reason="TestClient is built on httpx")

from fastapi.testclient import TestClient  # noqa: E402

import server  # noqa: E402
import main as m  # noqa: E402


@pytest.fixture(autouse=True)
def clean_state():
    """Reset the global state and run guard around every test.

    ``APP_STATE`` and ``_run_active`` are module-level and mutated by the
    background thread, so without this a test that starts a run would make the
    next one fail with a 409 for reasons that have nothing to do with the code
    under test.
    """
    server._run_active = False
    server.APP_STATE.update({"status": "idle", "course_code": None, "message": ""})
    yield
    server._run_active = False
    server.APP_STATE.update({"status": "idle", "course_code": None, "message": ""})


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
    import threading

    slot: list = [None, None]

    def _worker() -> None:
        try:
            slot[0] = _start(client, code, language)
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            slot[1] = exc

    thread = threading.Thread(target=_worker, daemon=True)
    thread.start()
    return thread, slot


# ---------------------------------------------------------------------------
# 1. API contract
# ---------------------------------------------------------------------------


def test_status_starts_idle(client):
    """A fresh server reports idle with no course and no message drift."""
    body = client.get("/api/status").json()
    assert body["status"] == "idle"
    assert body["course_code"] is None


def test_status_returns_exactly_the_three_contract_keys(client):
    """The payload shape is the dashboard's contract; extra keys leak internals.

    The frontend reads ``status``/``course_code``/``message`` and nothing else,
    so a key added for debugging would be a key with no defined behaviour on
    the client side.
    """
    assert set(client.get("/api/status").json()) == {
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
    import threading

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
    body = client.get("/api/status").json()
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

    import threading

    hold, release = threading.Event(), threading.Event()
    real = server.run_pipeline
    server.run_pipeline = _slow
    try:
        thread, slot = _start_in_thread(client, "CRM_1106")
        assert hold.wait(timeout=10), "run never reached the recording state"
        body = client.get("/api/status").json()
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

    body = client.get("/api/status").json()
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

    assert client.get("/api/status").json()["status"] == "error"


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
    body = client.get("/api/status").json()
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

    Same reason as the device. More pointedly: a browser cannot ask, and the
    default taken here is to refuse, because an existing ``raw_audio.wav`` is a
    previous lecture's only copy.
    """
    _start(client)
    assert client.calls[0]["kwargs"]["confirm_overwrite"] is False


# ---------------------------------------------------------------------------
# 4. Static shell
# ---------------------------------------------------------------------------


def test_root_serves_the_dashboard(client):
    """``/`` must return the HTML shell, not a 404."""
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]


def test_api_routes_survive_the_root_static_mount(client):
    """The ``/`` mount must not shadow the API.

    Load-bearing ordering, not a style choice: Starlette matches routes in
    registration order and a mount at ``/`` matches every path, so registering
    it before the endpoints would make ``/api/status`` return a static 404. The
    symptom is a dashboard that loads and then silently never updates -- which
    looks exactly like a broken backend.
    """
    assert client.get("/api/status").status_code == 200
    assert client.get("/healthz").status_code == 200


def test_healthz_is_independent_of_pipeline_state(client):
    """Liveness must not change when a lecture is running.

    A monitor polling a payload that flips to "transcribing" cannot tell "the
    server is down" from "the server is busy", and will page for the second one.
    """
    assert client.get("/healthz").json() == {"status": "ok"}
    _start(client)
    assert client.get("/healthz").json() == {"status": "ok"}


def test_dashboard_polls_the_status_endpoint(client):
    """The shell must poll the endpoint the server actually exposes.

    Checked against the shipped HTML rather than a copy, so a rename on either
    side fails here instead of leaving a UI that polls a 404 forever.
    """
    html = (server.STATIC_DIR / "index.html").read_text(encoding="utf-8")
    assert "/api/status" in html
    assert "setInterval" in html
