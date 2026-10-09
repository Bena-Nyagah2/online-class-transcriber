"""Behavioural tests for the Phase 6 ``main.py`` orchestration layer.

Scope
-----
``main.py`` is the only module the other four engines never test. These tests
cover the two things that changed when ``rich`` telemetry was introduced:

1. **Console routing** -- every operator message goes through ``console`` /
   ``console_err``, the live spinner is genuinely a spinner, and the stage rules
   delineate the pipeline.
2. **Stream separation** -- genuine errors must still reach *stderr* after the
   move away from ``print(..., file=sys.stderr)``.

No audio hardware is touched. ``AudioRecorder``, ``SlideCatcher`` and both AI
engines are replaced with fakes, and ``main()`` is driven end to end.

Console isolation
-----------------
Every test redirects ``main.console`` and ``main.console_err`` to an
``io.StringIO`` via a fixture, so no markup, no ANSI, and no spinner frames ever
reach the real terminal and pytest output stays completely clean. The
underlying ``rich`` behaviour (animation, soft-wrap, cursor restore) is still
genuinely exercised -- only the destination changes.
"""

from __future__ import annotations

import io
import os
import signal
import subprocess
import sys
import threading
import time
import types
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent

import pytest
from rich.console import Console

import main as m
from transcriber import DEFAULT_LANGUAGE

# Spinner glyphs rich cycles through for the "dots" spinner. Seeing more than
# one proves the live display actually repainted rather than printing once.
DOTS_FRAMES = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"


# ---------------------------------------------------------------------------
# Console isolation
# ---------------------------------------------------------------------------


@pytest.fixture
def consoles():
    """Redirect both module-level consoles into in-memory buffers.

    ``force_terminal`` is off by default so assertions read plain text. Tests
    that specifically exercise terminal behaviour build their own console and
    patch over this one.
    """
    out, err = io.StringIO(), io.StringIO()
    with mock.patch.object(m, "console", Console(file=out, force_terminal=False)), \
         mock.patch.object(m, "console_err", Console(file=err, force_terminal=False)):
        yield out, err


@pytest.fixture
def term_console():
    """A console that believes it is a TTY, for animation assertions."""

    def _make(**kwargs):
        buf = io.StringIO()
        opts = dict(force_terminal=True, width=100, legacy_windows=False)
        opts.update(kwargs)
        console = Console(file=buf, **opts)
        return console, buf

    return _make


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class FakeRecorder:
    """Stand-in for ``AudioRecorder``: one device, instant capture window."""

    def __init__(self, emergency_backup_path=None, **kwargs):
        self.emergency_backup_path = emergency_backup_path
        # ``is_recording.is_set()`` returning False ends ``run_status_loop``
        # on its first poll, i.e. the "device dropped on its own" branch.
        self.is_recording = types.SimpleNamespace(is_set=lambda: False)

    def get_loopback_devices(self):
        return [types.SimpleNamespace(name="Speakers (Realtek)", channels=2)]

    def start_recording(self, device_id):
        return None

    def stop_recording(self, wav_path: Path) -> None:
        wav_path.write_bytes(b"RIFF" + b"\0" * 200)


class EmptyDeviceRecorder(FakeRecorder):
    """``get_loopback_devices`` returns nothing -- the fatal pre-flight case."""

    def get_loopback_devices(self):
        return []


class ExplodingRecorder(FakeRecorder):
    """Fails at capture start, to exercise the ``(RuntimeError, OSError)`` branch."""

    def start_recording(self, device_id):
        raise OSError("loopback device vanished mid-capture")


class SilentRecorder(FakeRecorder):
    """Records nothing -- the capture produced an empty WAV.

    Needed for the "no audio captured" branch: the base fake writes 204 bytes
    on ``stop_recording``, which would satisfy the size check that branch tests.
    """

    def stop_recording(self, wav_path: Path) -> None:
        wav_path.write_bytes(b"")


class EndlessRecorder(FakeRecorder):
    """Keeps recording until something other than the device stops it.

    The base fake's ``is_recording`` is permanently False, so
    ``run_status_loop`` ends on its first poll -- the device-drop branch. A
    stop-event test needs the opposite: a loop that stays alive, so the *only*
    available exit is the caller's event. That is the situation the web
    dashboard is in, and it is not reachable with the base fake.
    """

    # Set by the test before the run; ``start_recording`` fires it to stand in
    # for the operator pressing "Finish Class" a moment after capture began.
    # Firing it from inside the recorder keeps the test deterministic: no
    # sleep, no race against a background timer.
    stop_event: threading.Event | None = None

    def __init__(self, emergency_backup_path=None, **kwargs):
        super().__init__(emergency_backup_path, **kwargs)
        self.is_recording = types.SimpleNamespace(is_set=lambda: True)

    def start_recording(self, device_id):
        if self.stop_event is not None:
            self.stop_event.set()
        return None


class OneTickRecorder(FakeRecorder):
    """Reports ``is_recording`` true exactly once, then false.

    The CLI-hint test needs both halves at once: the loop body must run so
    something is printed, *and* the loop must return without a stop event --
    which means ``is_recording`` has to clear by itself. Neither existing fake
    does both. ``FakeRecorder`` never reports recording, so the body never runs
    and nothing is printed; ``EndlessRecorder`` never stops, so the loop spins
    forever. This is the only combination that terminates with output.
    """

    def __init__(self, emergency_backup_path=None, **kwargs):
        super().__init__(emergency_backup_path, **kwargs)
        ticks = 0

        def _is_set():
            nonlocal ticks
            ticks += 1
            return ticks == 1

        self.is_recording = types.SimpleNamespace(is_set=_is_set)


def _stoppable(stop_event: threading.Event) -> type[EndlessRecorder]:
    """An :class:`EndlessRecorder` that fires ``stop_event`` as capture begins.

    Built with ``type(...)`` rather than a ``class`` statement because a class
    body cannot read a name it is also assigning -- ``class C(Base): stop_event
    = stop_event`` raises ``NameError``, since the assignment makes it a
    class-local and the enclosing local is not in scope for that lookup.
    """
    return type("StoppableRecorder", (EndlessRecorder,), {"stop_event": stop_event})


class FakeCatcher:
    """Stand-in for ``SlideCatcher`` that reports three unique slides."""

    slides_dir = Path("slides")

    def __init__(self, output_dir=None, **kwargs):
        self.output_dir = output_dir

    def start(self):
        return None

    def stop(self):
        return ["a.jpg", "b.jpg", "c.jpg"]


@pytest.fixture
def session(tmp_path, monkeypatch):
    """A prepared output directory with a non-empty WAV already in it.

    ``main()`` refuses to run the AI phases unless the WAV exists and is
    non-empty, so it is pre-seeded rather than relying on the recorder fake.

    ``prompt_language`` is stubbed for the same reason as ``prompt_course_code``:
    ``main()`` now prompts for the language when called with no argument, and an
    unmocked ``input()`` raises ``OSError`` under pytest's capture. Returning
    :data:`DEFAULT_LANGUAGE` models an operator pressing Enter.
    """
    (tmp_path / "raw_audio.wav").write_bytes(b"RIFF" + b"\0" * 200)
    monkeypatch.setattr(m, "create_output_dir", lambda code: tmp_path)
    monkeypatch.setattr(m, "prompt_course_code", lambda: "CRM_1106")
    monkeypatch.setattr(m, "prompt_language", lambda *a, **k: DEFAULT_LANGUAGE)
    monkeypatch.setattr(m, "select_device", lambda recorder: 0)
    monkeypatch.setattr(m, "_confirm_overwrite", lambda path: True)
    return tmp_path


SEGMENTS_STUB = [
    {"start": 0.0, "end": 5.0, "text": "mock lecture", "words": []}
]


def _transcriber_stub(segments=None):
    """A LocalTranscriber stand-in wired to the three-phase API.

    Returns ``(patcher, stub)`` so a test can assert on ``stub.mock`` -- the
    class-level mock -- and therefore on ``LocalTranscriber``'s own construction
    arguments. Stubbing only ``mock.patch.object(m, "LocalTranscriber", ...)``
    hides those, which is how the language-routing tests lost their assertion.

    The phases are wired as a pass-through chain so ordering assertions see a
    real sequence (``transcribe`` then ``align`` then ``diarize``) rather than
    three disconnected calls.
    """
    segments = list(SEGMENTS_STUB if segments is None else segments)
    instance = mock.MagicMock()
    instance.transcribe.side_effect = lambda wav_path, language=None: list(segments)
    instance.align.side_effect = (
        lambda wav_path, segs, language=None: list(segs)
    )
    instance.diarize.side_effect = lambda wav_path, segs: list(segs)
    instance.write_transcript.side_effect = (
        lambda wav_path, segs=None: Path(wav_path).parent / m.TRANSCRIPT_FILENAME
    )
    instance.render_transcript.side_effect = lambda segs=None: "mock transcript"
    cls = mock.MagicMock(return_value=instance)
    patcher = mock.patch.object(m, "LocalTranscriber", cls)
    # The handle goes on the *patcher*, because that is what ``with t`` yields
    # and therefore what a test still holds after the block. Attaching it to the
    # mock instead is the easy mistake here: ``mock.patch.object`` returns a
    # ``_patch``, so ``t.mock`` is an AttributeError, and the patcher only grows
    # a ``.mock`` while it is entered -- by which point the ``with`` has it.
    patcher.instance = instance
    return patcher, cls


def _stub_engines(slow: float | None = None):
    """Patch both AI engines; optionally make them block for ``slow`` seconds."""

    def _block():
        if slow:
            time.sleep(slow)

    transcriber = mock.MagicMock()
    instance = transcriber.return_value
    # Accepts the ``language=`` keyword that main() forwards to both the ASR
    # and the alignment stage; a bare positional lambda would fail the moment
    # routing is added. Each phase passes the previous one's segments through so
    # the chain is connected, and the block happens inside the phase so a
    # Ctrl+C test still lands mid-transcription.
    instance.transcribe.side_effect = lambda wav_path, language=None: (
        (_block(), list(SEGMENTS_STUB))[1]
    )
    instance.align.side_effect = lambda wav_path, segs, language=None: (
        (_block(), list(segs))[1]
    )
    instance.diarize.side_effect = lambda wav_path, segs: (
        (_block(), list(segs))[1]
    )
    instance.write_transcript.side_effect = (
        lambda wav_path, segs=None: Path(wav_path).parent / m.TRANSCRIPT_FILENAME
    )
    instance.render_transcript.side_effect = lambda segs=None: "mock transcript"
    synthesizer = mock.MagicMock()
    synthesizer.return_value.generate_notes.side_effect = lambda *a, **k: _block()
    transcriber_patcher = mock.patch.object(m, "LocalTranscriber", transcriber)
    # Exposed for tests that need to ask which phases ran; see _transcriber_stub
    # for why the handle lives on the patcher rather than the mock.
    transcriber_patcher.instance = instance
    return (
        transcriber_patcher,
        mock.patch.object(m, "LectureSynthesizer", synthesizer),
    )


# ---------------------------------------------------------------------------
# 1. Spinner genuinely animates on a TTY
# ---------------------------------------------------------------------------


def test_spinner_animates_multiple_frames(term_console):
    """A blocking call behind ``_spinner`` must repaint, not print once.

    This is the whole point of the telemetry: transcription is a multi-minute
    silent stretch, and a static line would give the operator no signal that
    the process is alive.
    """
    console, buf = term_console()
    with mock.patch.object(m, "console", console):
        with m._spinner("[bold yellow]Transcribing...[/bold yellow]"):
            time.sleep(0.6)
    frames = buf.getvalue()
    assert len(set(DOTS_FRAMES) & set(frames)) > 1, "spinner never repainted"
    assert "Transcribing" in frames


def test_spinner_label_is_rendered(term_console):
    console, buf = term_console()
    with mock.patch.object(m, "console", console):
        with m._spinner("Synthesizing study notes..."):
            pass
    assert "Synthesizing study notes" in buf.getvalue()


# ---------------------------------------------------------------------------
# 2. Ctrl+C unwinds the live display exactly like a normal exit
# ---------------------------------------------------------------------------


def test_spinner_propagates_keyboard_interrupt(term_console):
    """The interrupt must reach the caller, not be swallowed by the context."""
    console, _ = term_console()
    raised = False
    with mock.patch.object(m, "console", console):
        try:
            with m._spinner("working..."):
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            raised = True
    assert raised, "_spinner swallowed the KeyboardInterrupt"


def test_spinner_restores_cursor_on_interrupt(term_console):
    """``\\x1b[?25h`` re-shows the cursor; without it the terminal looks wedged.

    This is why the spinner uses a context manager rather than a bare
    ``console.status(...)`` object that is started and never stopped.
    """
    console, buf = term_console()
    with mock.patch.object(m, "console", console):
        try:
            with m._spinner("working..."):
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
    assert "\x1b[?25h" in buf.getvalue(), "cursor left hidden after interrupt"


def test_spinner_interrupt_path_identical_to_normal_exit(term_console):
    """The two paths must unwind byte-for-byte identically.

    Stronger than the two checks above: it proves the interrupt path is not
    merely "acceptable" but is the *same* teardown rich performs normally, so
    no partial frame or stray cursor position can survive a double Ctrl+C.
    """
    interrupted_console, interrupted_buf = term_console(width=60)
    normal_console, normal_buf = term_console(width=60)

    with mock.patch.object(m, "console", interrupted_console):
        try:
            with m._spinner("working..."):
                raise KeyboardInterrupt
        except KeyboardInterrupt:
            pass
    with mock.patch.object(m, "console", normal_console):
        with m._spinner("working..."):
            pass

    assert interrupted_buf.getvalue() == normal_buf.getvalue()


# ---------------------------------------------------------------------------
# 3. Non-TTY degradation (piped to a file, or run under pytest)
# ---------------------------------------------------------------------------


def test_non_tty_output_has_no_ansi_escapes(consoles):
    """When stdout is not a terminal, output must be plain text.

    Rich emits escape sequences whenever it *thinks* it owns a terminal. If
    that detection ever goes wrong, redirecting ``python main.py > run.log``
    would produce a log full of mojibake.
    """
    out, _ = consoles
    m.print_warning_banner()
    m._stage("Initialization")
    assert "\x1b[" not in out.getvalue()


def test_long_path_is_not_wrapped(consoles):
    """A Windows path is one unbreakable token; rich must not fold it.

    Wrapping would orphan the label and, worse, inject a newline into the
    middle of a string the user is about to copy.
    """
    out, _ = consoles
    m._kv("Audio      :", r"C:\very\long\path\raw_audio.wav")
    lines = [ln for ln in out.getvalue().split("\n") if ln.strip()]
    assert len(lines) == 1, f"path was wrapped across {len(lines)} lines"
    assert r"C:\very\long\path\raw_audio.wav" in lines[0]


def test_path_may_exceed_console_width_verbatim(consoles):
    """``soft_wrap`` lets the line overrun the width rather than inserting \\n.

    Run at a deliberately narrow width so the test would fail under rich's
    default word-wrapping behaviour.
    """
    narrow = Console(file=io.StringIO(), force_terminal=False, width=40)
    buf = narrow.file
    with mock.patch.object(m, "console", narrow):
        m._kv("Audio      :", r"C:\very\long\path\raw_audio.wav")
    line = [ln for ln in buf.getvalue().split("\n") if ln.strip()][0]
    assert len(line) > 40, "line was truncated instead of allowed to overrun"
    assert r"C:\very\long\path\raw_audio.wav" in line


def test_stage_rule_survives_as_plain_text(consoles):
    out, _ = consoles
    m._stage("Transcription & Diarization")
    assert any("Transcription & Diarization" in ln for ln in out.getvalue().split("\n"))


def test_warning_banner_survives_as_plain_text(consoles):
    out, _ = consoles
    m.print_warning_banner()
    text = out.getvalue()
    assert "WARNING" in text
    assert "Mute system notifications" in text


# ---------------------------------------------------------------------------
# 4. Errors keep going to stderr
# ---------------------------------------------------------------------------


def test_runtime_error_goes_to_stderr(consoles, session):
    """A capture failure must not be lost when stdout is redirected."""
    out, err = consoles
    with mock.patch.object(m, "AudioRecorder", ExplodingRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher):
        assert m.main() == 1
    assert "loopback device vanished" in err.getvalue()
    assert "loopback device vanished" not in out.getvalue()


def test_stage_narration_still_on_stdout_after_stderr_error(consoles, session):
    """Progress goes to stdout even when the failure goes to stderr."""
    out, _ = consoles
    with mock.patch.object(m, "AudioRecorder", ExplodingRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher):
        m.main()
    assert "Audio Capture" in out.getvalue()


def test_no_devices_message_goes_to_stderr(consoles, tmp_path, monkeypatch):
    """The fatal pre-flight failure keeps its original stderr destination.

    Deliberately does *not* use the ``session`` fixture: that one stubs out
    ``select_device``, and the real implementation is the thing under test.
    """
    out, err = consoles
    monkeypatch.setattr(m, "create_output_dir", lambda code: tmp_path)
    monkeypatch.setattr(m, "prompt_course_code", lambda: "CRM_1106")
    monkeypatch.setattr(m, "prompt_language", lambda *a, **k: DEFAULT_LANGUAGE)
    monkeypatch.setattr(m, "_confirm_overwrite", lambda path: True)
    with mock.patch.object(m, "AudioRecorder", EmptyDeviceRecorder):
        with pytest.raises(SystemExit) as excinfo:
            m.main()
    assert excinfo.value.code == 1
    assert "No loopback audio devices" in err.getvalue()
    assert "No loopback audio devices" not in out.getvalue()


# ---------------------------------------------------------------------------
# 5. End-to-end orchestration
# ---------------------------------------------------------------------------


def test_full_pipeline_announces_every_stage(consoles, session):
    """The four required stages must all appear, in pipeline order."""
    out, _ = consoles
    t, s = _stub_engines()
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, s:
        assert m.main() == 0
    text = out.getvalue()
    positions = [
        text.index(stage)
        for stage in (
            "Initialization",
            "Audio Capture",
            "Transcription & Diarization",
            "Academic Synthesis",
        )
    ]
    assert positions == sorted(positions), "stages printed out of order"


def test_full_pipeline_summarises_every_artifact(consoles, session):
    """The operator must be told where all four outputs landed."""
    out, _ = consoles
    t, s = _stub_engines()
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, s:
        m.main()
    text = out.getvalue()
    for label in ("Audio", "Transcript", "Slides", "Study Notes"):
        assert label in text
    assert "3 saved" in text, "slide count missing from the summary"


def test_transcription_runs_before_synthesis(consoles, session):
    """Guards the phase ordering the spinner blocks are supposed to bracket.

    Also the regression test for the no-op synthesis bug: the ``synthesis``
    lambda used to build its argument tuple and return it without ever calling
    ``generate_notes``, so this assertion is what proves the stage executes.
    """
    order: list[str] = []
    transcriber = mock.MagicMock()
    transcriber.return_value.transcribe.side_effect = (
        lambda p, language=None: (order.append("t"), list(SEGMENTS_STUB))[1]
    )
    transcriber.return_value.align.side_effect = (
        lambda p, segs, language=None: (order.append("a"), list(segs))[1]
    )
    transcriber.return_value.diarize.side_effect = (
        lambda p, segs: (order.append("d"), list(segs))[1]
    )
    transcriber.return_value.write_transcript.side_effect = (
        lambda p, segs=None: Path(p).parent / m.TRANSCRIPT_FILENAME
    )
    synthesizer = mock.MagicMock()
    synthesizer.return_value.generate_notes.side_effect = (
        lambda *a, **k: order.append("s")
    )
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         mock.patch.object(m, "LocalTranscriber", transcriber), \
         mock.patch.object(m, "LectureSynthesizer", synthesizer):
        m.main()
    assert order == ["t", "a", "d", "s"]


def test_synthesis_receives_the_three_artifact_paths(consoles, session):
    """``generate_notes`` must get (transcript, notes, slides) in that order.

    Signature regression guard. A positional shuffle here would still satisfy
    the ordering test above, so the arguments are checked explicitly.
    """
    t, _ = _stub_engines()
    synth = mock.MagicMock()
    synth.return_value.generate_notes.return_value = True
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", synth):
        m.main()
    (args, kwargs) = synth.return_value.generate_notes.call_args
    transcript_path, notes_path, slides_dir = args
    assert transcript_path.name == m.TRANSCRIPT_FILENAME
    assert notes_path.suffix == ".md"
    assert slides_dir.name == "slides"
    assert all(isinstance(a, Path) for a in args)
    # The domain rides as a keyword so the three paths keep their positions;
    # the default is pinned here because a silent drift would re-shape every
    # run's notes without a single call site changing.
    assert kwargs["domain"] == "criminology"


def test_synthesis_value_error_reports_transcript_path(consoles, session):
    """On a skipped synthesis the transcript path must still be surfaced."""
    out, _ = consoles
    t, _ = _stub_engines()
    failing = mock.MagicMock()
    failing.return_value.generate_notes.side_effect = ValueError("no transcript")
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", failing):
        assert m.main() == 1
    text = out.getvalue()
    assert "Synthesis skipped" in text
    assert "no transcript" in text
    assert m.TRANSCRIPT_FILENAME in text


def test_main_forwards_language_to_the_transcriber(consoles, session):
    """The CLI's language choice must reach the ASR *and* alignment stages.

    Routing regression guard: ``main()`` takes the code and ``transcriber``
    consumes it, and nothing in between re-derives or drops it. A wrong code
    does not crash the decoder — it returns fluent text in the wrong language.

    Both consuming stages are asserted. Alignment models are
    language-specific, so a code that reached the ASR but not the aligner would
    fail there in a way this test would otherwise miss entirely.
    """
    # NOTE: ``_stub_engines`` also patches LocalTranscriber, and because that
    # patcher is applied *after* the one under test it would silently win --
    # leaving ``transcriber.call_args`` as None. Only the synthesizer is stubbed.
    synth = mock.MagicMock()
    synth.return_value.generate_notes.return_value = True
    t, s = _transcriber_stub(list(SEGMENTS_STUB))
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", synth):
        assert m.main(language="sw") == 0
    instance = s.return_value
    assert instance.transcribe.call_args.kwargs == {"language": "sw"}
    assert instance.align.call_args.kwargs == {"language": "sw"}


def test_main_prompts_and_falls_back_to_english_on_empty_answer(
    consoles, session, monkeypatch
):
    """``main()`` with no argument prompts, and Enter yields the default.

    An unset language is not a neutral default: WhisperX falls back to
    ``detect_language()``, an extra model pass this change exists to skip. So
    the prompt must resolve to a real code, never ``""`` -- an empty string is
    falsy and would re-enable detection anyway.
    """
    # See the note in test_main_forwards_language_to_the_transcriber: stubbing
    # LocalTranscriber twice would let the later patcher win.
    monkeypatch.setattr(m, "prompt_language", lambda *a, **k: DEFAULT_LANGUAGE)
    t, _ = _transcriber_stub()
    synth = mock.MagicMock()
    synth.return_value.generate_notes.return_value = True
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", synth):
        m.main()
    instance = t.instance
    assert instance.transcribe.call_args.kwargs["language"] == DEFAULT_LANGUAGE
    assert instance.align.call_args.kwargs["language"] == DEFAULT_LANGUAGE


def test_explicit_language_skips_the_prompt_entirely(consoles, session, mocker):
    """Passing a language must not ask the operator anything.

    The prompt moved inside ``main()`` to fix the ordering, which means
    ``main()`` now reads stdin on the default path. A caller that already knows
    the language -- a scheduled run, a test, a repeat of the same lecture --
    must be able to bypass that, or every automated invocation now hangs on a
    question nobody is there to answer.
    """
    ask = mocker.patch.object(m, "prompt_language")
    t, _ = _transcriber_stub()
    synth = mock.MagicMock()
    synth.return_value.generate_notes.return_value = True
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", synth):
        assert m.main(language="sw") == 0
    ask.assert_not_called()
    instance = t.instance
    assert instance.transcribe.call_args.kwargs["language"] == "sw"
    assert instance.align.call_args.kwargs["language"] == "sw"


def test_double_ctrl_c_during_synthesis_returns_130(consoles, session):
    """Second interrupt must exit 130 and point at the preserved output dir."""
    out, _ = consoles
    t, _ = _stub_engines()
    interrupted = mock.MagicMock()
    interrupted.return_value.generate_notes.side_effect = KeyboardInterrupt
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", interrupted):
        assert m.main() == 130
    assert "Pipeline interrupted" in out.getvalue()
    assert str(session) in out.getvalue()


def test_single_ctrl_c_retries_the_phase_once(consoles, session):
    """One interrupt warns and retries; a second propagates to the caller.

    Regression guard for placing ``_spinner`` *inside* ``_interrupt_protected``:
    the retry must re-enter a live spinner rather than surface as a hard error.

    Only the synthesizer is stubbed here. Reusing ``_stub_engines`` would also
    patch ``LocalTranscriber``, and because that patcher is applied after the
    one under test it would silently win and this test's own recorder would
    never be invoked.
    """
    attempts: list[int] = []

    def flaky(_path, language=None):
        attempts.append(1)
        if len(attempts) == 1:
            raise KeyboardInterrupt
        return list(SEGMENTS_STUB)

    transcriber = mock.MagicMock()
    # The interrupt is raised from the *first* expensive phase. Align and
    # diarize stay wired to pass segments through so the retry, once it
    # succeeds, still has a real chain to feed.
    transcriber.return_value.transcribe.side_effect = flaky
    transcriber.return_value.align.side_effect = (
        lambda p, segs, language=None: list(segs)
    )
    transcriber.return_value.diarize.side_effect = lambda p, segs: list(segs)
    transcriber.return_value.write_transcript.side_effect = (
        lambda p, segs=None: Path(p).parent / m.TRANSCRIPT_FILENAME
    )
    synthesizer = mock.MagicMock()
    synthesizer.return_value.generate_notes.return_value = True
    out, _ = consoles
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         mock.patch.object(m, "LocalTranscriber", transcriber), \
         mock.patch.object(m, "LectureSynthesizer", synthesizer):
        assert m.main() == 0
    assert len(attempts) == 2, "phase was not retried exactly once"
    assert "Let it finish" in out.getvalue()


def test_empty_wav_skips_ai_phases(consoles, tmp_path, monkeypatch):
    """No audio means no API spend -- the engines must never be invoked."""
    (tmp_path / "raw_audio.wav").write_bytes(b"")  # exists but zero bytes
    monkeypatch.setattr(m, "create_output_dir", lambda code: tmp_path)
    monkeypatch.setattr(m, "prompt_course_code", lambda: "CRM_1106")
    monkeypatch.setattr(m, "prompt_language", lambda *a, **k: DEFAULT_LANGUAGE)
    monkeypatch.setattr(m, "select_device", lambda recorder: 0)
    monkeypatch.setattr(m, "_confirm_overwrite", lambda path: True)
    out, _ = consoles
    transcriber = mock.MagicMock()
    synthesizer = mock.MagicMock()
    with mock.patch.object(m, "AudioRecorder", SilentRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         mock.patch.object(m, "LocalTranscriber", transcriber), \
         mock.patch.object(m, "LectureSynthesizer", synthesizer):
        assert m.main() == 1
    assert "No audio captured" in out.getvalue()
    # Every phase of the transcriber, not just the old combined entry point.
    # Checking only one would let a phase leak through while the test stayed
    # green -- the phases were split precisely so each could be skipped
    # independently on a resume.
    for phase in ("transcribe", "align", "diarize", "write_transcript"):
        getattr(transcriber.return_value, phase).assert_not_called()
    synthesizer.return_value.generate_notes.assert_not_called()


# ---------------------------------------------------------------------------
# 6. run_pipeline: the programmatic entry point
# ---------------------------------------------------------------------------
# ``main()`` is now a thin shell over ``run_pipeline()``. These tests cover the
# contract the web backend depends on: the return value, the state sequence, the
# keyword-only escape hatches, and the PipelineError path.


@pytest.fixture
def recorded_states():
    """Collect emitted pipeline states into a list.

    Returns the list itself rather than a recording callable, so a test can
    assert on it directly after the run.
    """
    return []


def _drive_run(course_code="CRM_1106", recorder_cls=FakeRecorder, **kwargs):
    """Run the pipeline with every dependency faked; return ``(result, states)``.

    Shared by the ``run_pipeline`` tests so each one states only what it is
    actually asserting. ``_stub_engines`` supplies the two AI engines;
    ``session`` (used by the caller) has already stubbed the interactive bits.

    ``recorder_cls`` exists for the stop-event tests: the default
    :class:`FakeRecorder` drops out of the capture loop immediately, which is
    the device-drop branch, so a test that needs the loop to stay alive has to
    ask for a different recorder.
    """
    states: list[str] = []
    t, s = _stub_engines()
    with mock.patch.object(m, "AudioRecorder", recorder_cls), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, s:
        result = m.run_pipeline(
            course_code,
            state_callback=states.append,
            **kwargs,
        )
    return result, states


def test_run_pipeline_returns_the_notes_path(consoles, session):
    """The return value must be the study-notes path, not ``None``.

    This is the whole reason the function is shaped the way it is: the web
    backend reports the artifact location to the operator, and a pipeline that
    only printed that path to a terminal would have nothing to hand back.
    """
    result, _ = _drive_run()
    assert isinstance(result, str)
    assert result.endswith(".md")
    assert "CRM_1106" in result


def test_run_pipeline_emits_every_stage_in_order(consoles, session):
    """States must arrive as a strictly ordered sequence ending in ``complete``.

    Order is the contract, not decoration: a dashboard that learned "the run
    finished" before "transcribing" would stop showing the phase the operator is
    actually waiting in. ``complete`` must be last -- it is the only signal that
    the run is over.
    """
    _, states = _drive_run()
    assert states == [
        m.STATE_RECORDING,
        m.STATE_TRANSCRIBING,
        m.STATE_SYNTHESIZING,
        m.STATE_COMPLETE,
    ]


def test_states_are_emitted_at_stage_start_not_at_the_end(consoles, session):
    """A state must fire *before* its stage does, or it is a progress lie.

    Asserted positionally: recording the state sequence and the order the
    engines were invoked in gives four possible interleavings, and only one of
    them is honest. A status that fires after the expensive call has already
    finished tells the operator nothing while it is happening.
    """
    timeline: list[str] = []

    def record_state(state):
        timeline.append(f"state:{state}")

    transcriber = mock.MagicMock()
    # The three phases are recorded individually, because the assertion below
    # is that STATE_TRANSCRIBING brackets *all* of them: it fires once, before
    # the first phase, and synthesis is not announced until the last has
    # finished.
    transcriber.return_value.transcribe.side_effect = (
        lambda p, language=None: (timeline.append("transcribe"), list(SEGMENTS_STUB))[1]
    )
    transcriber.return_value.align.side_effect = (
        lambda p, segs, language=None: (timeline.append("align"), list(segs))[1]
    )
    transcriber.return_value.diarize.side_effect = (
        lambda p, segs: (timeline.append("diarize"), list(segs))[1]
    )
    transcriber.return_value.write_transcript.side_effect = (
        lambda p, segs=None: Path(p).parent / m.TRANSCRIPT_FILENAME
    )
    synthesizer = mock.MagicMock()
    synthesizer.return_value.generate_notes.side_effect = (
        lambda *a, **k: timeline.append("synthesize")
    )
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         mock.patch.object(m, "LocalTranscriber", transcriber), \
         mock.patch.object(m, "LectureSynthesizer", synthesizer):
        m.run_pipeline("CRM_1106", state_callback=record_state)

    assert timeline == [
        f"state:{m.STATE_RECORDING}",
        f"state:{m.STATE_TRANSCRIBING}",
        "transcribe",
        "align",
        "diarize",
        f"state:{m.STATE_SYNTHESIZING}",
        "synthesize",
        f"state:{m.STATE_COMPLETE}",
    ]


def test_callback_is_optional(consoles, session):
    """Omitting the callback must not break the run.

    ``None`` is the default, so a caller that does not want progress has to be
    able to pass nothing at all. A stray ``state_callback()`` on ``None`` would
    make the parameter mandatory in practice while its signature claimed
    otherwise.
    """
    t, s = _stub_engines()
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, s:
        result = m.run_pipeline("CRM_1106")
    assert result.endswith(".md")


def test_error_path_emits_the_error_state_and_raises(consoles, session):
    """A failed run must announce ``error`` and raise, not return quietly.

    Both halves matter. The raise is how ``main()`` gets its exit code; the
    state is how a dashboard stops displaying "Transcribing" for a run that has
    already died. Returning a path here would report success for a lecture whose
    notes were never written.
    """
    t, _ = _stub_engines()
    failing = mock.MagicMock()
    failing.return_value.generate_notes.side_effect = ValueError("no transcript")
    states: list[str] = []
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", failing):
        with pytest.raises(m.PipelineError):
            m.run_pipeline("CRM_1106", state_callback=states.append)

    assert states[-1] == m.STATE_ERROR
    assert m.STATE_COMPLETE not in states, "reported success for a failed run"


def test_error_state_precedes_the_exception(consoles, session, monkeypatch):
    """The state must be emitted before the raise, not skipped by it.

    ``_emit`` is called on the way out of each failure branch. If someone later
    moves it into an ``except`` handler that re-raises first, or drops it, the
    dashboard hangs on the last in-progress state with no way to tell.
    """
    order: list[str] = []
    monkeypatch.setattr(m, "_emit", lambda cb, state: order.append(state))
    t, _ = _stub_engines()
    failing = mock.MagicMock()
    failing.return_value.generate_notes.side_effect = ValueError("no transcript")
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", failing):
        with pytest.raises(m.PipelineError):
            m.run_pipeline("CRM_1106", lambda state: None)
    assert order[-1] == m.STATE_ERROR


def test_pipeline_error_carries_the_cancellation_exit_code(consoles, session):
    """A cancelled run must report 130, not the generic failure code of 1.

    130 is the conventional "terminated by SIGINT" status. Collapsing it into 1
    would make a scheduled run's cancellation indistinguishable from a genuine
    engine failure in whatever consumes its exit status.
    """
    # Both the retry and the re-raise must be interrupted, otherwise the
    # single-press retry would swallow the second Ctrl+C and the run would
    # complete instead of reporting 130.
    interrupting = mock.MagicMock()
    interrupting.return_value.transcribe.side_effect = [
        KeyboardInterrupt, KeyboardInterrupt
    ]
    synth = mock.patch.object(m, "LectureSynthesizer", mock.MagicMock())
    with synth, \
         mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         mock.patch.object(m, "LocalTranscriber", interrupting):
        with pytest.raises(m.PipelineError) as excinfo:
            m.run_pipeline("CRM_1106")
    assert excinfo.value.exit_code == 130


def test_main_maps_pipeline_error_to_the_exit_code(consoles, session):
    """``main()`` must translate the exception into a process exit code.

    The translation is the entire remaining job of ``main()``. If it leaked the
    exception instead, the process would die with a traceback and status 1 --
    which happens to look right for a failure but is wrong for the cancelled
    case, where 130 is the answer.
    """
    with mock.patch.object(m, "run_pipeline", side_effect=m.PipelineError("x", 130)):
        assert m.main(language="en") == 130
    with mock.patch.object(m, "run_pipeline", side_effect=m.PipelineError("x", 1)):
        assert m.main(language="en") == 1


def test_main_still_prints_the_failure_detail(consoles, session):
    """Moving the error handling into an exception must not silence the message.

    ``main()`` no longer owns the wording, so the guard is that the *combined*
    path still reaches the operator: ``run_pipeline`` prints at the raise site
    because that is where the transcript path is in scope, and ``main`` only
    converts the exception to a code.
    """
    t, _ = _stub_engines()
    failing = mock.MagicMock()
    failing.return_value.generate_notes.side_effect = ValueError("no transcript")
    out, _ = consoles
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", failing):
        assert m.main(language="en") == 1
    assert "Synthesis skipped" in out.getvalue()


def test_explicit_device_id_skips_the_prompt(consoles, session):
    """A non-interactive caller must be able to skip ``select_device``.

    The web backend has no terminal to prompt on. Without this, the first
    browser-initiated run would block forever on an ``input()`` nobody can
    answer -- the request would hang rather than fail, which is much harder to
    diagnose.
    """
    def explode(recorder):
        raise AssertionError("select_device must not be called")

    with mock.patch.object(m, "select_device", explode):
        result, _ = _drive_run(device_id=0)
    assert result.endswith(".md")


def test_explicit_device_index_is_validated(consoles, session):
    """An out-of-range index must fail before a capture stream is opened.

    Without validation the index reaches soundcard, which either raises an
    opaque error or -- worse -- opens the wrong device. An HTTP client can send
    any integer, so the bound has to be checked at this boundary.
    """
    t, s = _stub_engines()
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, s:
        with pytest.raises(m.PipelineError):
            m.run_pipeline("CRM_1106", device_id=7)
    # Fail fast: the capture thread must never have been started.
    assert s.new is not None


def test_confirm_overwrite_false_aborts_before_recording(consoles, session):
    """Declining the overwrite must not start a capture.

    A browser cannot ask the operator, so ``server.py`` passes ``False``. If
    that value were ignored and the prompt fell through to ``input()``, the
    request would hang on stdin instead of refusing -- and if it were ignored
    the other way, an existing recording would be destroyed unasked.
    """
    def explode(path):
        raise AssertionError("_confirm_overwrite must not be called")

    t, s = _stub_engines()
    with mock.patch.object(m, "_confirm_overwrite", explode), \
         mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, s:
        with pytest.raises(m.PipelineError) as excinfo:
            m.run_pipeline("CRM_1106", confirm_overwrite=False)
    assert excinfo.value.exit_code == 1
    out, _ = consoles
    assert "Aborted" in out.getvalue()


def test_resolve_device_index_picks_the_first_device():
    """The non-interactive selector returns index 0 without prompting."""
    assert m.resolve_device_index(FakeRecorder()) == 0


def test_resolve_device_index_fails_loudly_when_there_is_no_device():
    """No loopback device is fatal, not a silent default-capture fallback.

    Recording from an arbitrary input would produce a ``raw_audio.wav`` of
    silence, and the failure would surface much later as the far less obvious
    "no audio captured".

    Asserts ``PipelineError`` rather than ``SystemExit`` on purpose. This helper
    exists only for the web path, where the failure has to be catchable by
    ``server._run_in_background`` -- and ``SystemExit`` is a ``BaseException``,
    so it would escape ``except Exception`` and strand the dashboard on
    "recording" with nothing to retry against.
    """
    with pytest.raises(m.PipelineError) as excinfo:
        m.resolve_device_index(EmptyDeviceRecorder())
    assert excinfo.value.exit_code == 1


def test_resolve_device_index_error_is_catchable_as_an_exception():
    """The raise must land in the ``except Exception`` net the server relies on.

    A direct, mechanical guard on the reason the previous ``SystemExit`` was
    wrong. If someone "simplifies" this back to a process exit, the suite stays
    green everywhere except in production, where the dashboard hangs.
    """
    with pytest.raises(Exception):  # noqa: B017 - deliberately broad on purpose
        m.resolve_device_index(EmptyDeviceRecorder())


def test_cli_state_callback_renders_labels(consoles):
    """The CLI must translate tokens back into the human stage names.

    This is the inversion that keeps ``run_pipeline`` presentation-free: the
    tokens go out bare, and only the terminal re-adds markup. If the mapping
    lost an entry, a stage would print its own snake_case token as a heading.
    """
    m._cli_state_callback(m.STATE_TRANSCRIBING)
    assert "Transcription & Diarization" in consoles[0].getvalue()


def test_cli_state_callback_covers_every_emitted_state(consoles):
    """No emitted state may fall through to a raw token as its heading.

    Exhaustive over the states ``run_pipeline`` actually emits, which is the set
    that matters -- an unmapped state would surface as a heading reading
    "transcribing" in lower case, an obvious but easy-to-miss regression.
    """
    for state in (
        m.STATE_RECORDING,
        m.STATE_TRANSCRIBING,
        m.STATE_SYNTHESIZING,
        m.STATE_COMPLETE,
    ):
        assert state in m._STATE_LABELS, f"{state} has no CLI label"


def test_state_tokens_carry_no_rich_markup():
    """Wire tokens must be bare -- markup would leak into JSON and the DOM.

    ``LABELS``/``_STATE_LABELS`` hold presentation; the tokens themselves cross
    the process boundary and are rendered by a browser. A stray ``[bold]`` in a
    token would arrive as literal text in the dashboard's badge.
    """
    for state in (
        m.STATE_RECORDING,
        m.STATE_TRANSCRIBING,
        m.STATE_SYNTHESIZING,
        m.STATE_COMPLETE,
        m.STATE_ERROR,
    ):
        assert state == state.lower()
        assert "[" not in state and "]" not in state


# ---------------------------------------------------------------------------
# 7. No stray output during the test run
# ---------------------------------------------------------------------------


def test_prompt_language_defaults_on_empty_enter(monkeypatch):
    """Pressing Enter must yield the default, not an empty string.

    An empty string would reach Whisper as ``language=""``, which is falsy and
    so re-enables ``detect_language()`` — the exact cost this prompt removes.
    """
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert m.prompt_language() == DEFAULT_LANGUAGE


def test_prompt_language_normalises_case(monkeypatch):
    """Whisper's codes are lower-case; a pasted 'EN' must be accepted."""
    monkeypatch.setattr("builtins.input", lambda *a: "  SW  ")
    assert m.prompt_language() == "sw"


def test_prompt_language_honours_a_custom_default(monkeypatch):
    monkeypatch.setattr("builtins.input", lambda *a: "")
    assert m.prompt_language(default="fr") == "fr"


def test_prompt_language_falls_back_on_absurd_input(monkeypatch):
    """A pasted sentence falls back to the default.

    Silently accepting one would otherwise fail as an opaque CTranslate2 error
    deep inside the decoder, long after the recording is already written.
    """
    monkeypatch.setattr("builtins.input", lambda *a: "english please")
    assert m.prompt_language() == DEFAULT_LANGUAGE


def test_the_course_code_prompt_uses_a_domain_neutral_example():
    """The CLI's example code must not smuggle a discipline back in.

    The prompt text is what a new operator sees first, and an example like
    ``CRM_1106`` quietly tells them what the tool is "for". The example is a
    placeholder by design; this pins it there.
    """
    import inspect

    source = inspect.getsource(m.prompt_course_code)
    assert "COURSE_101" in source, "the example course code is missing"
    assert "CRM" not in source, "a criminology code is still in the CLI prompt"


def test_run_pipeline_forwards_the_domain_to_synthesis(consoles, session):
    """The profile chosen at the edge must arrive at ``generate_notes``.

    Default *and* explicit, because the two fail differently: a lost default
    silently reverts every run to the generic profile, while a lost explicit
    value would only show up when someone noticed their Software Engineering
    meeting produced law-school notes.
    """
    synth = mock.MagicMock()
    synth.return_value.generate_notes.return_value = True
    t, _ = _stub_engines()
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", synth):
        m.run_pipeline("COURSE_101")
    assert synth.return_value.generate_notes.call_args.kwargs["domain"] == "criminology"

    synth.reset_mock()
    synth.return_value.generate_notes.return_value = True
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", synth):
        m.run_pipeline("SYS_Arch", domain="software")
    assert synth.return_value.generate_notes.call_args.kwargs["domain"] == "software"


# ---------------------------------------------------------------------------
# 6b. The SQLite session ledger
# ---------------------------------------------------------------------------


def _use_temp_ledger(tmp_path, monkeypatch):
    """Repoint the pipeline's ledger at a throwaway SQLite file.

    Patching ``database.SessionLocal`` (not the engine) is sufficient and
    intent-revealing: ``run_pipeline`` resolves the name at call time, so every
    session it opens lands on this file. Returns the sessionmaker so a test can
    open its own session to read back what the pipeline wrote.
    """
    from sqlalchemy import create_engine
    from sqlalchemy.orm import sessionmaker

    engine = create_engine(
        f"sqlite:///{tmp_path / 'lectures.db'}",
        connect_args={"check_same_thread": False},
    )
    m.models.Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    monkeypatch.setattr(m.database, "SessionLocal", factory)
    return factory


def _run_with_fake_engines(course_code="COURSE_101", **kwargs):
    """Drive a full run with both AI engines stubbed; return the notes path.

    The synthesizer stub *writes the notes file* as a side effect, because the
    real one does and the sync block reads that file back. A stub that only
    returned a mock would make every ledger test exercise the wrong branch
    (``read_text`` raising into the except) and pass for the wrong reason.
    """
    t, _ = _stub_engines()

    synth = mock.MagicMock()

    def _write_notes(transcript_path, notes_path, slides_dir, domain=None):
        Path(notes_path).write_text("# study notes\n", encoding="utf-8")
        return True

    synth.return_value.generate_notes.side_effect = _write_notes
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", synth):
        return m.run_pipeline(course_code, **kwargs)


def test_a_successful_run_records_a_synced_lecture(
    consoles, session, tmp_path, monkeypatch
):
    """The ledger must capture who/what/where and the upload's outcome."""
    factory = _use_temp_ledger(tmp_path, monkeypatch)
    synced: list[tuple[str, str, str]] = []

    def _fake_sync(course_code, date_str, content):
        synced.append((course_code, date_str, content))
        return True

    monkeypatch.setattr("services.gdrive.sync_markdown", _fake_sync)

    result = _run_with_fake_engines("COURSE_101", domain="software")

    assert result.endswith(".md")
    assert len(synced) == 1
    assert synced[0][0] == "COURSE_101"

    db = factory()
    try:
        rows = db.query(m.models.Lecture).all()
    finally:
        db.close()

    assert len(rows) == 1
    row = rows[0]
    assert row.course_code == "COURSE_101"
    assert row.domain == "software"
    assert row.session_date
    assert row.audio_path.endswith(m.WAV_FILENAME)
    assert row.md_path.endswith(".md")
    assert row.sync_status == "SUCCESS"


def test_a_failed_sync_is_recorded_as_failed(consoles, session, tmp_path, monkeypatch):
    """A ``False`` from the sync service must persist as ``FAILED``, not vanish.

    This is the whole reason the row is written before the upload: without it,
    a silent Drive failure would look identical to a run that never tried.
    """
    factory = _use_temp_ledger(tmp_path, monkeypatch)
    monkeypatch.setattr("services.gdrive.sync_markdown", lambda *a, **k: False)

    _run_with_fake_engines("COURSE_101")

    db = factory()
    try:
        row = db.query(m.models.Lecture).one()
    finally:
        db.close()
    assert row.sync_status == "FAILED"


def test_a_sync_exception_is_recorded_as_failed_without_crashing(
    consoles, session, tmp_path, monkeypatch
):
    """The Drive client raising is a ``FAILED`` upload, not a failed lecture."""
    factory = _use_temp_ledger(tmp_path, monkeypatch)

    def _boom(*args, **kwargs):
        raise RuntimeError("drive is down")

    monkeypatch.setattr("services.gdrive.sync_markdown", _boom)

    result = _run_with_fake_engines("COURSE_101")

    assert result.endswith(".md")
    db = factory()
    try:
        row = db.query(m.models.Lecture).one()
    finally:
        db.close()
    assert row.sync_status == "FAILED"


def test_a_ledger_failure_never_crashes_the_run(consoles, session, monkeypatch):
    """The notes are already on disk; a broken database must not undo that.

    ``SessionLocal`` is replaced with a session whose ``commit`` always raises.
    The pipeline must still return its notes path and must still close the
    session -- the ``finally`` is not optional just because the body failed.
    """
    closed = {"n": 0}

    class BrokenSession:
        def add(self, obj):  # noqa: D401 - bare test double
            pass

        def commit(self):
            raise RuntimeError("disk full")

        def refresh(self, obj):
            pass

        def rollback(self):
            pass

        def close(self):
            closed["n"] += 1

    monkeypatch.setattr(m.database, "SessionLocal", lambda: BrokenSession())
    monkeypatch.setattr("services.gdrive.sync_markdown", lambda *a, **k: True)

    result = _run_with_fake_engines("COURSE_101")

    assert result.endswith(".md")
    assert closed["n"] == 1


def test_course_code_is_asked_before_the_language(consoles, tmp_path, mocker):
    """The course code prompt must come first; the language prompt second.

    This ordering is the whole reason the prompt moved out of the ``__main__``
    block. ``main(prompt_language())`` is valid Python and asks for the language
    *first*, because arguments are evaluated before the call is entered. The
    failure is silent: the pipeline still runs, just with the answers attached
    to the wrong questions.

    Asserted by recording the prompt strings in the order they are issued,
    rather than by checking the final language value -- that value is right in
    both orderings, so only the sequence distinguishes them.
    """
    (tmp_path / "raw_audio.wav").write_bytes(b"RIFF" + b"\0" * 200)
    mocker.patch.object(m, "create_output_dir", lambda code: tmp_path)
    mocker.patch.object(m, "select_device", lambda r: 0)
    mocker.patch.object(m, "_confirm_overwrite", lambda p: True)

    asked: list[str] = []

    def _course() -> str:
        asked.append("course")
        return "CRM_1106"

    def _language(default=DEFAULT_LANGUAGE) -> str:
        asked.append("language")
        return "sw"

    mocker.patch.object(m, "prompt_course_code", _course)
    mocker.patch.object(m, "prompt_language", _language)

    t, _ = _transcriber_stub()
    synth = mock.MagicMock()
    synth.return_value.generate_notes.return_value = True
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, mock.patch.object(m, "LectureSynthesizer", synth):
        assert m.main() == 0

    assert asked == ["course", "language"]
    assert t.instance.transcribe.call_args.kwargs["language"] == "sw"


def test_cli_entrypoint_prompts_and_forwards_the_language(tmp_path):
    """The real ``__main__`` path must prompt, then forward the answer.

    Asserted by executing the module as a script, not by hand-writing
    ``main()``'s internals the way the ``__main__`` block relies on them. A test
    that restates the production wiring proves nothing *about* the wiring: when
    the CLI stopped calling into ``main()``, the hand-written version kept
    passing, and a mutation run confirmed that mutant survived. Running the real
    entry point means removing the prompt genuinely fails here.

    NOTE ``monkeypatch`` is deliberately not a parameter -- see the patching
    note below for why in-process patching cannot reach the CLI path.
    """
    wav = tmp_path / "raw_audio.wav"
    wav.write_bytes(b"RIFF" + b"\0" * 200)

    # NOTE the patching strategy. ``runpy.run_module(..., run_name="__main__")``
    # re-executes main.py into a *fresh* module namespace, so patching the
    # already-imported ``main`` object in this process would be silently
    # discarded. The engines are therefore patched at their defining modules
    # (``recorder``/``transcriber``/``summarizer``/``vision``), which
    # main.py's own ``from X import Y`` lines pick up when it runs.
    #
    # Functions defined *inside* main.py (create_output_dir, prompt_course_code,
    # select_device, _confirm_overwrite) cannot be patched this way, so their
    # prompts are answered through a scripted stdin instead. cwd is redirected
    # so the real create_output_dir writes into tmp_path, not the repo.
    script = f'''\
import builtins, contextlib, io, os, runpy, sys, types
from unittest import mock
sys.path.insert(0, {str(ROOT)!r})
import recorder, transcriber, summarizer, vision

class R:
    def __init__(self, **k):
        self.is_recording = types.SimpleNamespace(is_set=lambda: False)
    def get_loopback_devices(self):
        return [types.SimpleNamespace(name="Spk", channels=2)]
    def start_recording(self, i): pass
    def stop_recording(self, p): p.write_bytes(b"RIFF" + b"\\0" * 200)

class C:
    slides_dir = "slides"
    def __init__(self, **k): pass
    def start(self): pass
    def stop(self): return ["a.jpg"]

SEGS = [{{"start": 0.0, "end": 1.0, "text": "mock", "words": []}}]
captured = {{}}
def cap_asr(wav_path, language=None):
    captured["language"] = language
    return list(SEGS)
def cap_align(wav_path, segs, language=None):
    captured["align_language"] = language
    return list(segs)

T = mock.MagicMock()
T.return_value.transcribe.side_effect = cap_asr
T.return_value.align.side_effect = cap_align
T.return_value.diarize.side_effect = lambda p, segs: list(segs)
T.return_value.write_transcript.side_effect = (
    lambda wav_path, segs=None: types.SimpleNamespace(
        name="raw_transcript.txt", parent=wav_path.parent, __str__=lambda s: str(wav_path.parent / s.name)
    )
)
S = mock.MagicMock()
S.return_value.generate_notes.return_value = True

recorder.AudioRecorder = R
vision.SlideCatcher = C
transcriber.LocalTranscriber = T
summarizer.LectureSynthesizer = S

# Prompt order, as main() now issues them: course code, language, overwrite
# confirmation, device index. Getting this list wrong does not crash -- it
# feeds each answer to the wrong question, and the pipeline still "succeeds".
answers = iter(["CRM_1106", "sw", "y", "0"])
builtins.input = lambda *a: next(answers)

os.chdir({str(wav.parent)!r})
try:
    with contextlib.redirect_stdout(io.StringIO()):
        runpy.run_module("main", run_name="__main__")
except SystemExit:
    pass
print("CAPTURED_LANG=" + repr(captured.get("language")))
print("CAPTURED_ALIGN_LANG=" + repr(captured.get("align_language")))
'''

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT), capture_output=True, text=True, env={**os.environ},
    )
    # Both consuming phases, for the same reason as the in-process routing test.
    for marker in ("CAPTURED_LANG='sw'", "CAPTURED_ALIGN_LANG='sw'"):
        assert marker in proc.stdout, (
            f"{marker} not forwarded from the CLI prompt.\n"
            f"stdout: {proc.stdout[-2000:]}\nstderr: {proc.stderr[-2000:]}"
        )


def test_module_defines_no_bare_print_calls():
    """Guard the migration itself: raw ``print`` must not creep back in.

    A bare ``print`` is invisible to the console fixtures, so it would leak to
    the real terminal and break the "suite is silent" guarantee.
    """
    import ast

    tree = ast.parse(Path(m.__file__).read_text(encoding="utf-8"))
    offenders = [
        node.lineno
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "print"
    ]
    assert not offenders, f"bare print() at line(s) {offenders}"


def test_status_is_imported_from_rich_status():
    """``rich.console`` does not re-export ``Status``; importing it from there
    raises ``ImportError`` at module load. This pins the working import path
    so a well-meaning "cleanup" to a single import line cannot break the app.
    """
    from rich import status as rich_status

    assert m.Status is rich_status.Status
    import rich.console

    assert not hasattr(rich.console, "Status")


# ---------------------------------------------------------------------------
# 8. Caller-driven stop (the web dashboard's "Finish Class")
# ---------------------------------------------------------------------------
# The CLI ends a recording with Ctrl+C, which raises KeyboardInterrupt on
# whichever thread owns the terminal. ``server.py`` has no terminal, so the
# same outcome is driven by a ``threading.Event`` the pipeline watches.
#
# The contract is deliberately narrow: setting the event must make the capture
# loop *return*, not raise, so the pipeline falls through into transcription
# exactly as it does after a Ctrl+C. A stop that unwound the run would discard
# the lecture the operator just recorded -- the opposite of what they asked for.


def test_a_set_stop_event_ends_the_recording_phase(consoles, session):
    """The loop must return rather than raise.

    Returning is the whole point. ``run_status_loop`` returning is what lets
    ``run_pipeline`` continue into transcription; if it raised instead, the
    event would abort the run and the audio that was already captured would be
    thrown away.
    """
    stop_event = threading.Event()

    result, states = _drive_run(
        device_id=0,
        recorder_cls=_stoppable(stop_event),
        stop_event=stop_event,
        # True, not False: the session fixture pre-seeds a non-empty WAV so the
        # AI phases will run, and `False` means "refuse to overwrite", which
        # aborts before the recording phase is ever reached.
        confirm_overwrite=True,
    )

    assert result.endswith(".md"), "a stopped recording must still produce notes"
    assert states == [
        m.STATE_RECORDING,
        m.STATE_TRANSCRIBING,
        m.STATE_SYNTHESIZING,
        m.STATE_COMPLETE,
    ]


def test_a_requested_stop_is_not_reported_as_a_device_drop(consoles, session):
    """The operator has to be able to tell "I stopped it" from "it broke".

    Both land in the same place in the code, so they are told apart by whether
    ``stop_event`` is set. Conflating them is not a cosmetic slip: a lecturer
    whose laptop lost the audio device needs to know the recording is suspect
    and re-take it, and "possible device drop" on every deliberate stop trains
    them to ignore the one warning that matters.
    """
    stop_event = threading.Event()
    out, _ = consoles

    _drive_run(
        device_id=0,
        recorder_cls=_stoppable(stop_event),
        stop_event=stop_event,
        confirm_overwrite=True,
    )
    text = out.getvalue()

    assert "Recording stopped" in text
    assert "device drop" not in text


def test_an_unstopped_device_drop_is_still_reported_as_one(consoles, session):
    """The other half of the same distinction, so the fix above is not a blunt one.

    ``FakeRecorder`` never records, so the capture loop ends on its own with no
    stop event set. That must keep warning about a dropped device -- otherwise
    the previous test would pass just because the message was deleted.
    """
    out, _ = consoles

    _drive_run(device_id=0, confirm_overwrite=True)
    text = out.getvalue()

    assert "device drop" in text
    assert "Recording stopped" not in text


def test_a_stop_set_while_the_loop_is_waiting_ends_it_promptly(consoles):
    """Finish Class must feel immediate, not lag by up to a full second.

    The loop repaints once a second. If it slept on a timer between repaints,
    a stop would go unnoticed until the next tick -- up to a second of the
    button looking dead. It waits on the event itself instead, so the latency
    is however long the event takes, which is microseconds.

    The 0.5s ceiling is chosen to separate the two implementations rather than
    to be tight: a ``time.sleep(1.0)`` loop notices the event at ~0.85s, while
    an event-driven one returns in well under 10ms. Anything in between would
    be a machine problem, not a design one.
    """
    stop_event = threading.Event()
    recorder = EndlessRecorder()

    worker = threading.Thread(
        target=m.run_status_loop, args=(recorder, stop_event), daemon=True
    )
    worker.start()
    # Let the loop reach its wait, so the measurement covers waking it rather
    # than merely noticing an already-set event.
    time.sleep(0.15)

    started = time.monotonic()
    stop_event.set()
    worker.join(timeout=5)
    elapsed = time.monotonic() - started

    assert not worker.is_alive(), "the loop ignored the stop event"
    assert elapsed < 0.5, f"the loop took {elapsed:.2f}s to notice the stop"


def test_a_stop_ends_the_loop_even_though_nobody_pressed_a_key(consoles):
    """The loop is gated on two things and must react to the one it is given.

    ``recorder.is_recording`` stays set forever in this fake, so the event is
    the only exit available. A regression that made the loop watch only the
    recorder -- the pre-existing single-source behaviour -- would hang here
    rather than fail, so the join is given a deadline and the thread is a
    daemon so a genuine hang fails the test instead of wedging the suite.
    """
    stop_event = threading.Event()
    recorder = EndlessRecorder()  # is_recording stays True
    outcome: list[str] = []

    def _run() -> None:
        try:
            m.run_status_loop(recorder, stop_event)
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            outcome.append(f"{type(exc).__name__}: {exc}")
        else:
            outcome.append("returned")

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    time.sleep(0.15)
    stop_event.set()
    worker.join(timeout=5)

    assert outcome == ["returned"], outcome or "the loop never returned"


def test_run_pipeline_works_off_the_main_thread(consoles, session):
    """Regression test: the web server runs this pipeline in a worker thread.

    ``signal.signal`` raises ``ValueError: signal only works in main thread of
    the main interpreter`` anywhere but the main thread, and the recording
    phase used to call it unconditionally while flushing the WAV. Every other
    test in this file runs on the main thread, so none of them could catch
    that -- the failure only ever appeared under uvicorn, at the exact moment a
    finished recording was being written to disk.

    The whole pipeline is driven from a worker thread here and must complete
    normally. This also covers the deliberate-stop path off-thread, since the
    dashboard is the only caller that ever takes it.
    """
    stop_event = threading.Event()
    outcome: list = []

    def _run() -> None:
        try:
            outcome.extend(
                _drive_run(
                    device_id=0,
                    recorder_cls=_stoppable(stop_event),
                    stop_event=stop_event,
                    confirm_overwrite=True,
                )
            )
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            outcome.append(f"{type(exc).__name__}: {exc}")

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=60)

    assert not worker.is_alive(), "run_pipeline hung off the main thread"
    assert len(outcome) == 2, f"the run failed off-thread: {outcome}"
    assert outcome[0].endswith(".md"), outcome[0]
    assert outcome[1][-1] == m.STATE_COMPLETE, outcome[1]


def test_suppress_sigint_is_a_no_op_off_the_main_thread():
    """The guard must degrade to nothing rather than raise where it cannot apply.

    There is nothing to suppress in a worker thread anyway: SIGINT is delivered
    to the main thread only, so the worker could never receive the interrupt
    this exists to defend against.
    """
    errors: list[str] = []

    def _run() -> None:
        try:
            with m._suppress_sigint():
                pass
        except BaseException as exc:  # noqa: BLE001 - reported to the caller
            errors.append(f"{type(exc).__name__}: {exc}")

    worker = threading.Thread(target=_run, daemon=True)
    worker.start()
    worker.join(timeout=10)

    assert not worker.is_alive()
    assert errors == [], errors[0]


def test_suppress_sigint_restores_the_previous_handler_on_the_main_thread():
    """The CLI still needs its guard, and must hand the handler back.

    Asserted with a sentinel rather than ``SIG_DFL`` because the real contract
    is "put back what was there": a program that installed its own SIGINT
    handler must still have it after the WAV flush. The previous code
    hard-coded ``SIG_DFL``, which silently discarded such a handler.
    """
    sentinel = lambda *a: None  # noqa: E731 - a signal handler is a plain callable
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, sentinel)
    try:
        with m._suppress_sigint():
            assert signal.getsignal(signal.SIGINT) is signal.SIG_IGN
        assert signal.getsignal(signal.SIGINT) is sentinel
    finally:
        signal.signal(signal.SIGINT, previous)


def test_suppress_sigint_restores_the_handler_even_when_the_body_raises():
    """A failed flush must not leave SIGINT ignored for the rest of the session.

    Otherwise one failed ``stop_recording`` would silently disarm Ctrl+C, and
    the operator's next attempt to abort a runaway transcription would do
    nothing at all.
    """

    def _boom() -> None:
        raise RuntimeError("flush failed")

    previous = signal.getsignal(signal.SIGINT)
    with pytest.raises(RuntimeError, match="flush failed"):
        with m._suppress_sigint():
            _boom()

    assert signal.getsignal(signal.SIGINT) is not signal.SIG_IGN
    assert signal.getsignal(signal.SIGINT) is previous


def test_the_prompt_omits_the_ctrl_c_hint_when_a_stop_event_is_supplied(consoles):
    """Telling a browser-driven run to press a key it has no keyboard for.

    The status line says "Press Ctrl+C to stop" because that is the only way to
    stop a CLI run. With a ``stop_event`` the caller has a different verb --
    the dashboard's button -- so the advice is simply wrong.

    ``EndlessRecorder`` is required so the loop body runs at least once (the
    base fake would exit before printing anything), and the event is set up
    front so the loop is guaranteed to terminate on its first wait.
    """
    out, _ = consoles
    stop_event = threading.Event()
    stop_event.set()  # end after a single repaint

    m.run_status_loop(EndlessRecorder(), stop_event)

    assert "Recording in progress" in out.getvalue()
    assert "Ctrl+C" not in out.getvalue()


def test_the_prompt_still_mentions_ctrl_c_without_a_stop_event(consoles):
    """The CLI is the default caller and its hint must survive the change.

    Guards against the hint being removed outright rather than made
    conditional, which would pass the previous test.

    ``OneTickRecorder`` because with no stop event the loop can only end on its
    own -- so the fake has to both print once and then stop, which is a
    combination neither other fake provides.
    """
    out, _ = consoles

    m.run_status_loop(OneTickRecorder())  # no stop_event: the CLI path

    assert "Ctrl+C" in out.getvalue()


# ---------------------------------------------------------------------------
# 9. Idempotent checkpointing (resume after a crash)
# ---------------------------------------------------------------------------
# The pipeline is check-before-compute against three files that sit at the three
# expensive phase boundaries. The tests below hold two separate contracts:
#
#   * a *resumed* run must not re-run a phase whose checkpoint exists, because
#     not re-running it is the entire feature; and
#   * a *fresh* run must not trust a checkpoint whose audio it has just
#     replaced, because that would pair this lecture's audio with the previous
#     lecture's words -- a silent, confident corruption.
#
# The second contract is why ``resume`` is opt-in: the output directory is keyed
# by course *and date*, so a same-day re-run finds the earlier take's audio
# already in place, and auto-resuming would quietly skip the new recording.


def _write_checkpoint(output_dir, filename, payload=None):
    """Drop a checkpoint file into ``output_dir`` and return its path."""
    path = output_dir / filename
    if payload is None:
        payload = [{"start": 0.0, "end": 1.0, "text": "from checkpoint", "words": []}]
    m.write_checkpoint(path, payload)
    return path


def _drive_resumable(session, course_code="CRM_1106", **kwargs):
    """Run the pipeline against stubbed engines; return ``(result, states, transcriber)``.

    ``transcriber`` is the *instance* the pipeline built, so a test can assert
    which phases were reached -- the whole question a resume test is asking.
    """
    states: list[str] = []
    t, s = _stub_engines()
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         t, s:
        result = m.run_pipeline(
            course_code, state_callback=states.append, **kwargs
        )
    return result, states, t.instance


def test_a_fresh_run_writes_every_checkpoint(consoles, session):
    """The checkpoints must be produced by the normal path, not just consumed.

    A writer that only ever reads is how this feature would look if it were
    implemented backwards -- every test that pre-seeds a file would still pass
    while a real crash had nothing to resume from.
    """
    _drive_resumable(session, confirm_overwrite=True)
    for filename in (
        m.TRANSCRIPTION_CHECKPOINT,
        m.ALIGNMENT_CHECKPOINT,
        m.DIARIZATION_CHECKPOINT,
    ):
        assert (session / filename).is_file(), f"{filename} was never written"


def test_resume_skips_every_completed_phase(consoles, session):
    """All three checkpoints present means no heavy compute at all.

    This is the fast-forward: the run should go straight to synthesis. The
    transcript is still rendered from the checkpointed segments, because that is
    what synthesis consumes and regenerating it is what makes a resume end in
    the same state as an uninterrupted run.
    """
    for filename in (
        m.TRANSCRIPTION_CHECKPOINT,
        m.ALIGNMENT_CHECKPOINT,
        m.DIARIZATION_CHECKPOINT,
    ):
        _write_checkpoint(session, filename)

    result, states, transcriber = _drive_resumable(
        session, confirm_overwrite=True, resume=True
    )

    # The point of the whole feature: the expensive calls never happen.
    transcriber.transcribe.assert_not_called()
    transcriber.align.assert_not_called()
    transcriber.diarize.assert_not_called()
    # But synthesis does, and the transcript is rebuilt from the checkpoint.
    transcriber.write_transcript.assert_called_once()
    assert result.endswith(".md")
    # Recording is skipped too, so STATE_RECORDING is never emitted;
    # STATE_TRANSCRIBING is skipped because no transcribing phase runs.
    assert states == [m.STATE_SYNTHESIZING, m.STATE_COMPLETE]


def test_resume_skips_recording_but_still_transcribes(consoles, session):
    """Audio present, no checkpoints: skip capture, run all three AI phases.

    The state sequence is the assertion that capture was skipped. It is also
    the practical payoff: the resumed run never touches capture hardware, so it
    works on a machine whose audio device is the thing that failed.
    """
    result, states, transcriber = _drive_resumable(
        session, confirm_overwrite=True, resume=True
    )

    assert states == [
        m.STATE_TRANSCRIBING,
        m.STATE_SYNTHESIZING,
        m.STATE_COMPLETE,
    ]
    transcriber.transcribe.assert_called_once()
    transcriber.align.assert_called_once()
    transcriber.diarize.assert_called_once()
    assert result.endswith(".md")


def test_resume_after_the_asr_phase_skips_only_transcription(consoles, session):
    """One checkpoint present must skip exactly one phase.

    Asserting that the later phases still ran is what separates "resumed
    correctly" from "skipped everything because the resume flag was read as
    'nothing left to do'".
    """
    _write_checkpoint(session, m.TRANSCRIPTION_CHECKPOINT)

    _, states, transcriber = _drive_resumable(
        session, confirm_overwrite=True, resume=True
    )

    transcriber.transcribe.assert_not_called()
    transcriber.align.assert_called_once()
    transcriber.diarize.assert_called_once()
    assert states[0] == m.STATE_TRANSCRIBING


def test_resume_after_alignment_skips_two_phases(consoles, session):
    """The mid-pipeline boundary, which is the one most likely to be missed.

    ``alignment.json`` is not one of the two checkpoints in the original
    sketch, so it is the one a regression would drop. Without it a crash during
    diarization would redo the alignment pass as well.
    """
    _write_checkpoint(session, m.TRANSCRIPTION_CHECKPOINT)
    _write_checkpoint(session, m.ALIGNMENT_CHECKPOINT)

    _, _, transcriber = _drive_resumable(
        session, confirm_overwrite=True, resume=True
    )

    transcriber.transcribe.assert_not_called()
    transcriber.align.assert_not_called()
    transcriber.diarize.assert_called_once()


def test_resume_reuses_the_checkpointed_segments_verbatim(consoles, session):
    """The resumed transcript must come from the file, not from a fresh run.

    Guards the failure mode where the checkpoint is consulted but its *content*
    is discarded -- the phase is skipped and then handed empty input.
    """
    marker = "the exact words that were checkpointed"
    _write_checkpoint(
        session, m.DIARIZATION_CHECKPOINT,
        [{"start": 0.0, "end": 1.0, "text": marker, "words": []}],
    )

    _, _, transcriber = _drive_resumable(
        session, confirm_overwrite=True, resume=True
    )

    (args, _kwargs) = transcriber.write_transcript.call_args
    _wav_path, segments = args
    assert segments[0]["text"] == marker


def test_a_run_that_is_not_resuming_ignores_existing_checkpoints(consoles, session):
    """Opting out of resume must opt out of *reading*, not of writing.

    This is the safety half of the feature. The output directory is keyed by
    course and date, so a same-day re-run would otherwise find the earlier
    take's checkpoints and transcribe those instead of the new lecture.

    Asserted with a *complete* checkpoint set -- transcription, alignment and
    diarization -- because a partial one is not a discriminator: with only
    ``transcription.json`` present, running the ASR and skipping it produce the
    same set of phase calls. Every phase must run for this test to distinguish
    "ignored the checkpoints" from "resumed them".
    """
    for filename in (
        m.TRANSCRIPTION_CHECKPOINT,
        m.ALIGNMENT_CHECKPOINT,
        m.DIARIZATION_CHECKPOINT,
    ):
        _write_checkpoint(session, filename)

    _, states, transcriber = _drive_resumable(session, confirm_overwrite=True)

    # Recording happened, so the checkpoints were discarded and all three
    # phases recomputed from the new take.
    assert m.STATE_RECORDING in states
    transcriber.transcribe.assert_called_once()
    transcriber.align.assert_called_once()
    transcriber.diarize.assert_called_once()


def test_fresh_audio_invalidates_the_previous_take_s_checkpoints(consoles, session):
    """Stale checkpoints must be cleared before replacement audio is used.

    Deleting is not housekeeping. A checkpoint is keyed only by directory, so
    one left behind after a re-record belongs to different audio entirely; the
    next resume would then splice yesterday's transcript onto today's lecture.

    Asserted on *content*, not existence: the fresh run writes its own
    checkpoints a moment later, so "the file is gone" is true only for an
    instant and proves nothing. What must hold is that the previous take's words
    are nowhere on disk.
    """
    marker = "STALE LECTURE FROM A PREVIOUS TAKE"
    for filename in (
        m.TRANSCRIPTION_CHECKPOINT,
        m.ALIGNMENT_CHECKPOINT,
        m.DIARIZATION_CHECKPOINT,
    ):
        _write_checkpoint(
            session, filename,
            [{"start": 0.0, "end": 1.0, "text": marker, "words": []}],
        )

    with mock.patch.object(
        m, "_invalidate_checkpoints", wraps=m._invalidate_checkpoints
    ) as invalidate:
        _drive_resumable(session, confirm_overwrite=True)

    assert invalidate.call_count == 1, "the stale checkpoints were never cleared"
    for filename in (
        m.TRANSCRIPTION_CHECKPOINT,
        m.ALIGNMENT_CHECKPOINT,
        m.DIARIZATION_CHECKPOINT,
    ):
        assert marker not in (session / filename).read_text(encoding="utf-8"), (
            f"{filename} still holds the previous take's content"
        )


def test_a_corrupt_checkpoint_is_recomputed_rather_than_trusted(consoles, session):
    """Garbage on disk must mean "recompute", never "load and continue".

    A truncated checkpoint is the expected artefact of exactly the crash this
    feature exists to survive, so it is a case the happy path must handle
    without special prompting. Treating it as valid would feed nonsense into
    the transcript and report success.
    """
    path = session / m.TRANSCRIPTION_CHECKPOINT
    path.write_text("[{'start': 0.0, 'text': 'half a',", encoding="utf-8")

    _, _, transcriber = _drive_resumable(
        session, confirm_overwrite=True, resume=True
    )

    transcriber.transcribe.assert_called_once()


def test_a_checkpoint_of_the_wrong_shape_is_recomputed(consoles, session):
    """Well-formed JSON of the wrong shape is still not a usable checkpoint.

    ``{"segments": []}`` parses cleanly, so a resume path that only checked
    "did the JSON load" would hand a dict to code expecting a list and fail
    deep inside the next phase, where the error names nothing useful.
    """
    _write_checkpoint(session, m.TRANSCRIPTION_CHECKPOINT, {"segments": []})

    _, _, transcriber = _drive_resumable(
        session, confirm_overwrite=True, resume=True
    )

    transcriber.transcribe.assert_called_once()


def test_a_run_that_is_not_resuming_never_reads_a_checkpoint(consoles, session):
    """No checkpoint is even *opened* unless the caller asked to resume.

    Narrower than the phase assertions above, and it is the one that holds the
    ``resume`` guard itself. Those other tests pass either way here, because
    Phase A invalidates the stale checkpoints before this point -- so by the time
    the transcription phase looks, there is nothing left to find and reading is
    indistinguishable from not reading. Asserting that the read never happens
    removes the masking.

    Read-versus-not is the whole safety property: a resumed flag that consults
    checkpoints on a fresh run is one refactor away from serving stale audio.
    """
    for filename in (
        m.TRANSCRIPTION_CHECKPOINT,
        m.ALIGNMENT_CHECKPOINT,
        m.DIARIZATION_CHECKPOINT,
    ):
        _write_checkpoint(session, filename)

    with mock.patch.object(
        m, "_load_segments", wraps=m._load_segments
    ) as load_segments:
        _drive_resumable(session, confirm_overwrite=True)

    assert not load_segments.called, (
        "a non-resuming run consulted a checkpoint; the resume flag is the only "
        "thing standing between a same-day re-run and the previous lecture"
    )


def test_a_default_run_records_even_though_an_earlier_take_exists(
    consoles, session
):
    """No ``resume`` means record, even when a WAV from an earlier take is there.

    The output directory is named for the course *and the date*, so a second run
    of the same course on the same day always finds yesterday's-or-this-morning's
    take sitting in it. If the mere existence of that file skipped capture, the
    second class of every day would silently record nothing -- and then
    transcribe the previous lecture as though it were today's. That is a data
    loss bug wearing the costume of a feature, which is why ``resume`` is opt-in.
    """
    # ``session`` pre-seeds a non-empty raw_audio.wav, i.e. exactly this case.
    assert (session / m.WAV_FILENAME).stat().st_size > 0

    _, states, transcriber = _drive_resumable(session, confirm_overwrite=True)

    assert m.STATE_RECORDING in states, (
        "an existing WAV silently suppressed capture without resume=True"
    )
    transcriber.transcribe.assert_called_once()


def test_resume_without_audio_records_normally(consoles, tmp_path, monkeypatch):
    """``resume=True`` is a hint, not an instruction to skip capture.

    A crash *during* the recording leaves no audio to resume from, and that is
    the case where the operator most wants a normal run. The flag must degrade
    to a full run rather than erroring or skipping to a transcript that does not
    exist.
    """
    monkeypatch.setattr(m, "create_output_dir", lambda code: tmp_path)
    monkeypatch.setattr(m, "prompt_course_code", lambda: "CRM_1106")
    monkeypatch.setattr(m, "prompt_language", lambda *a, **k: DEFAULT_LANGUAGE)
    monkeypatch.setattr(m, "select_device", lambda recorder: 0)
    monkeypatch.setattr(m, "_confirm_overwrite", lambda path: True)

    _, states, transcriber = _drive_resumable(tmp_path, resume=True)

    assert m.STATE_RECORDING in states
    transcriber.transcribe.assert_called_once()

