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
import sys
import time
import types
from pathlib import Path
from unittest import mock

import pytest
from rich.console import Console

import main as m

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
    """
    (tmp_path / "raw_audio.wav").write_bytes(b"RIFF" + b"\0" * 200)
    monkeypatch.setattr(m, "create_output_dir", lambda code: tmp_path)
    monkeypatch.setattr(m, "prompt_course_code", lambda: "CRM_1106")
    monkeypatch.setattr(m, "select_device", lambda recorder: 0)
    monkeypatch.setattr(m, "_confirm_overwrite", lambda path: True)
    return tmp_path


def _stub_engines(slow: float | None = None):
    """Patch both AI engines; optionally make them block for ``slow`` seconds."""

    def _block():
        if slow:
            time.sleep(slow)

    transcriber = mock.MagicMock()
    transcriber.return_value.transcribe_audio.side_effect = lambda p: (_block(), p)[1]
    synthesizer = mock.MagicMock()
    synthesizer.return_value.generate_notes.side_effect = lambda *a: _block()
    return (
        mock.patch.object(m, "LocalTranscriber", transcriber),
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
    transcriber.return_value.transcribe_audio.side_effect = lambda p: order.append("t")
    synthesizer = mock.MagicMock()
    synthesizer.return_value.generate_notes.side_effect = lambda *a: order.append("s")
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         mock.patch.object(m, "LocalTranscriber", transcriber), \
         mock.patch.object(m, "LectureSynthesizer", synthesizer):
        m.main()
    assert order == ["t", "s"]


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
    (args, _kwargs) = synth.return_value.generate_notes.call_args
    transcript_path, notes_path, slides_dir = args
    assert transcript_path.name == m.TRANSCRIPT_FILENAME
    assert notes_path.suffix == ".md"
    assert slides_dir.name == "slides"
    assert all(isinstance(a, Path) for a in args)


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

    def flaky(_path):
        attempts.append(1)
        if len(attempts) == 1:
            raise KeyboardInterrupt
        return _path

    transcriber = mock.MagicMock()
    transcriber.return_value.transcribe_audio.side_effect = flaky
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
    transcriber.return_value.transcribe_audio.assert_not_called()
    synthesizer.return_value.generate_notes.assert_not_called()


# ---------------------------------------------------------------------------
# 6. No stray output during the test run
# ---------------------------------------------------------------------------


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
