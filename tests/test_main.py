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
import subprocess
import sys
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


def _stub_engines(slow: float | None = None):
    """Patch both AI engines; optionally make them block for ``slow`` seconds."""

    def _block():
        if slow:
            time.sleep(slow)

    transcriber = mock.MagicMock()
    # Accepts the ``language=`` keyword that main() now forwards; a bare
    # positional lambda would fail here the moment routing is added.
    transcriber.return_value.transcribe_audio.side_effect = (
        lambda p, language=None: (_block(), p)[1]
    )
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
    transcriber.return_value.transcribe_audio.side_effect = (
        lambda p, language=None: order.append("t")
    )
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


def test_main_forwards_language_to_the_transcriber(consoles, session):
    """The CLI's language choice must reach ``transcribe_audio``.

    Routing regression guard: ``main()`` takes the code and ``transcriber``
    consumes it, and nothing in between re-derives or drops it. A wrong code
    does not crash the decoder — it returns fluent text in the wrong language.
    """
    # NOTE: ``_stub_engines`` also patches LocalTranscriber, and because that
    # patcher is applied *after* the one under test it would silently win --
    # leaving ``transcriber.call_args`` as None. Only the synthesizer is stubbed.
    synth = mock.MagicMock()
    synth.return_value.generate_notes.return_value = True
    transcriber = mock.MagicMock()
    transcriber.return_value.transcribe_audio.side_effect = (
        lambda p, language=None: p
    )
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         mock.patch.object(m, "LocalTranscriber", transcriber), \
         mock.patch.object(m, "LectureSynthesizer", synth):
        assert m.main(language="sw") == 0
    assert transcriber.return_value.transcribe_audio.call_args.kwargs == {
        "language": "sw"
    }


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
    transcriber = mock.MagicMock()
    transcriber.return_value.transcribe_audio.side_effect = (
        lambda p, language=None: p
    )
    synth = mock.MagicMock()
    synth.return_value.generate_notes.return_value = True
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         mock.patch.object(m, "LocalTranscriber", transcriber), \
         mock.patch.object(m, "LectureSynthesizer", synth):
        m.main()
    assert transcriber.return_value.transcribe_audio.call_args.kwargs[
        "language"
    ] == DEFAULT_LANGUAGE


def test_explicit_language_skips_the_prompt_entirely(consoles, session, mocker):
    """Passing a language must not ask the operator anything.

    The prompt moved inside ``main()`` to fix the ordering, which means
    ``main()`` now reads stdin on the default path. A caller that already knows
    the language -- a scheduled run, a test, a repeat of the same lecture --
    must be able to bypass that, or every automated invocation now hangs on a
    question nobody is there to answer.
    """
    ask = mocker.patch.object(m, "prompt_language")
    transcriber = mock.MagicMock()
    transcriber.return_value.transcribe_audio.side_effect = (
        lambda p, language=None: p
    )
    synth = mock.MagicMock()
    synth.return_value.generate_notes.return_value = True
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         mock.patch.object(m, "LocalTranscriber", transcriber), \
         mock.patch.object(m, "LectureSynthesizer", synth):
        assert m.main(language="sw") == 0
    ask.assert_not_called()
    assert transcriber.return_value.transcribe_audio.call_args.kwargs[
        "language"
    ] == "sw"


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
    transcriber.return_value.transcribe_audio.assert_not_called()
    synthesizer.return_value.generate_notes.assert_not_called()


# ---------------------------------------------------------------------------
# 6. No stray output during the test run
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

    transcriber = mock.MagicMock()
    transcriber.return_value.transcribe_audio.side_effect = (
        lambda p, language=None: p
    )
    synth = mock.MagicMock()
    synth.return_value.generate_notes.return_value = True
    with mock.patch.object(m, "AudioRecorder", FakeRecorder), \
         mock.patch.object(m, "SlideCatcher", FakeCatcher), \
         mock.patch.object(m, "LocalTranscriber", transcriber), \
         mock.patch.object(m, "LectureSynthesizer", synth):
        assert m.main() == 0

    assert asked == ["course", "language"]
    assert transcriber.return_value.transcribe_audio.call_args.kwargs[
        "language"
    ] == "sw"


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

captured = {{}}
def cap(p, language=None):
    captured["language"] = language
    return p

T = mock.MagicMock()
T.return_value.transcribe_audio.side_effect = cap
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
'''

    proc = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(ROOT), capture_output=True, text=True, env={**os.environ},
    )
    assert "CAPTURED_LANG='sw'" in proc.stdout, (
        f"language not forwarded from the CLI prompt.\n"
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
