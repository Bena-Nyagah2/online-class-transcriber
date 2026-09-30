"""CLI entry point for the full lecture pipeline (Phase 4).

Chains the Phase 1 recorder, Phase 2 transcriber, Phase 3 synthesizer, and
Phase 5 visual slide catcher into a single automated workflow::

    warn → course code → output dir → pick device → RECORD → [Ctrl+C]
        → save raw_audio.wav → transcribe → raw_transcript.txt
        → synthesize → {course_code}_{date}.md

Ctrl+C handling:
* 1st Ctrl+C during recording — clean stop, then hand off to AI.
* Ctrl+C while the API is working — warned once ("let it finish"), the
  interrupted phase is retried; a 2nd Ctrl+C exits safely, keeping every
  file that was already written to disk intact.
"""

from __future__ import annotations

import re
import signal
import sys
import time
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from typing import Any, Callable, Iterator

from recorder import AudioRecorder
from transcriber import (
    DEFAULT_LANGUAGE,
    LocalTranscriber,
    TRANSCRIPT_FILENAME,
)
from summarizer import LectureSynthesizer
from vision import SlideCatcher

# ``Console`` renders the markup; ``Status`` is the live spinner type returned
# by ``Console.status``. NOTE the split: ``Status`` is *not* re-exported from
# ``rich.console`` -- ``from rich.console import Status`` raises ImportError --
# so it must come from ``rich.status``.
from rich.console import Console
from rich.status import Status

# Single shared console. Rich auto-detects and enables VT processing on legacy
# Windows terminals, captures stdout at import time, and degrades to plain
# unstyled text when the output is not a TTY (piped to a file, run under
# pytest), so every one of these call sites is safe unconditionally.
console = Console()

# A second console bound to stderr, so genuine errors keep going to stderr
# even when stdout is piped to a file (``python main.py > run.log``). The
# original code distinguished these via ``print(..., file=sys.stderr)``; folding
# them into ``console`` would have silently downgraded them.
console_err = Console(stderr=True)

# Pipeline stage labels, kept in one place so the outline the operator reads at
# startup is the same one each phase announces.
STAGE_INIT = "[bold blue]Initialization[/bold blue]"
STAGE_CAPTURE = "[bold blue]Audio Capture[/bold blue]"
STAGE_TRANSCRIBE = "[bold blue]Transcription & Diarization[/bold blue]"
STAGE_SYNTHESIZE = "[bold blue]Academic Synthesis[/bold blue]"

# ---------------------------------------------------------------------------
# Pipeline states (the programmatic vocabulary)
# ---------------------------------------------------------------------------
# The rich labels above are *presentation*: they carry markup and are tuned for
# a terminal. These are *state*: bare lowercase tokens with no markup, safe to
# put on a wire, in a DOM id, or in a log aggregator. ``run_pipeline`` emits
# only these, and the CLI decides how to render them -- that inversion is what
# lets the same pipeline drive a terminal and a browser without either knowing
# about the other.
STATE_RECORDING = "recording"
STATE_TRANSCRIBING = "transcribing"
STATE_SYNTHESIZING = "synthesizing"
STATE_COMPLETE = "complete"
STATE_ERROR = "error"

# How the terminal renders each state. A state with no entry here would print
# its own token as a bare heading, so the mapping is exhaustive by construction.
_STATE_LABELS: dict[str, str] = {
    STATE_RECORDING: STAGE_CAPTURE,
    STATE_TRANSCRIBING: STAGE_TRANSCRIBE,
    STATE_SYNTHESIZING: STAGE_SYNTHESIZE,
    STATE_COMPLETE: "Pipeline Complete",
}


class PipelineError(RuntimeError):
    """A pipeline run failed, carrying the process exit code it maps to.

    ``run_pipeline`` cannot return an exit code -- it returns the path to the
    notes it produced -- so failure has to travel by exception for ``main()`` to
    keep reporting 0/1/130. The code is carried on the exception rather than
    left to the handler so the mapping stays in one place: a run that stops
    with ``PipelineError.exit_code`` ends the process with that same number.

    The message is for logs and tracebacks. Operator-facing wording is printed
    by ``run_pipeline`` at the raise site, where ``output_dir`` and the failing
    path are still in scope.
    """

    def __init__(self, message: str, exit_code: int = 1) -> None:
        super().__init__(message)
        self.exit_code = exit_code


def _stage(title: str) -> None:
    """Announce a pipeline stage with a rule so the phases are scannable.

    A lecture run is dominated by two long silent phases, so the operator needs
    a persistent visual marker of *where the pipeline currently is* that
    survives scrolling: a labelled rule, not just another line of text.
    """
    console.rule(f"[bold]{title}[/bold]", style="blue")


def _emit(
    state_callback: Callable[[str], None] | None, state: str
) -> None:
    """Report a pipeline state, if anyone is listening.

    The ``is None`` check rather than a default no-op callable: a caller that
    passes nothing should pay nothing, and a shared module-level no-op would be
    one more indirection to keep alive for a branch that is almost always taken.
    A callback that raises is *not* swallowed here. A broken status sink should
    be loud -- silently degrading it would hide exactly the kind of failure that
    leaves an operator staring at a stalled dashboard.
    """
    if state_callback is not None:
        state_callback(state)


def _kv(label: str, value: object, width: int = 13) -> None:
    """Print an aligned ``label : value`` row without ever breaking the value.

    ``soft_wrap=True`` is load-bearing here, not cosmetic. A Windows path is a
    single unbreakable token that routinely exceeds the terminal width; rich's
    default word-wrap would push the path onto its own line, orphaning the
    label and, worse, inserting a newline into the middle of a string the user
    is likely to select and paste. ``soft_wrap`` lets the terminal fold the
    line visually while leaving the underlying text intact.
    """
    console.print(
        f"   [cyan]{label:<{width}}[/cyan]{value}",
        soft_wrap=True,
        highlight=False,
    )


@contextmanager
def _spinner(label: str, spinner: str = "dots") -> Iterator[Status]:
    """Run a blocking call behind a live spinner.

    Yields the live :class:`~rich.status.Status` so a caller can update
    ``status.update(...)`` mid-stage, but nothing in this pipeline needs to.

    The ``with`` block is what makes Ctrl+C safe: ``Status.__exit__`` stops the
    Live display on the exception path too, so a double Ctrl+C during
    transcription cannot leave the cursor parked inside a spinner frame with
    no newline.
    """
    with console.status(label, spinner=spinner) as status:
        yield status


def _configure_streams() -> None:
    """Never let emoji crash output that is redirected to a file."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(errors="replace")


def _enable_windows_ansi() -> None:
    """Turn on ANSI colours on legacy Windows consoles (best effort)."""
    if sys.platform != "win32":
        return
    try:
        import ctypes

        STD_OUTPUT_HANDLE = -11
        ENABLE_VIRTUAL_TERMINAL_PROCESSING = 0x0004
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetStdHandle(STD_OUTPUT_HANDLE)
        mode = ctypes.c_uint32()
        if kernel32.GetConsoleMode(handle, ctypes.byref(mode)):
            kernel32.SetConsoleMode(
                handle, mode.value | ENABLE_VIRTUAL_TERMINAL_PROCESSING
            )
    except Exception:
        pass  # plain text still prints fine without colour


def print_warning_banner() -> None:
    """Print the startup warning: loopback records *everything* you hear."""
    bar = "!" * 78
    console.print(f"\n[bold red]{bar}[/bold red]")
    console.print(
        "[bold red]⚠️  WARNING:[/bold red] Mute system notifications (Slack, "
        "Email). Loopback captures ALL OS audio."
    )
    console.print(f"[bold red]{bar}[/bold red]\n")


def _sanitize_course_code(raw: str) -> str:
    """Collapse path-unsafe characters in a course code to underscores.

    Args:
        raw: User-supplied course unit code.

    Returns:
        A filesystem-safe course code the output directory can be built from.
    """
    return re.sub(r'[<>:"/\\|?*\s]+', "_", raw)


def prompt_course_code() -> str:
    """Prompt until a filesystem-safe course unit code is entered.

    Returns:
        The course code (e.g. ``CRM_1106``), sanitised for use in a path.
    """
    while True:
        raw = input("Enter Course Unit Code (e.g. CRM_1106): ").strip()
        if not raw:
            console.print("  [yellow]Course code cannot be empty — try again.[/yellow]")
            continue
        safe = _sanitize_course_code(raw)
        if safe != raw:
            console.print(
                f"  [yellow]Note: using [bold]{safe!r}[/bold] as the folder name.[/yellow]"
            )
        return safe


def prompt_language(default: str = DEFAULT_LANGUAGE) -> str:
    """Prompt for the spoken language, falling back to ``default`` on Enter.

    WhisperX can detect the language itself, but that costs an extra model pass
    over the opening 30 s, so the code is asked for up front instead. An empty
    answer is the documented "just use the default" path rather than an error —
    the overwhelming majority of lectures are English, and making the operator
    type ``en`` every run trains them to mash Enter without reading.

    The value is lower-cased because Whisper's language codes are ISO-639-1
    and a capitalised ``EN`` is not recognised. Length is bounded to two
    characters to catch a pasted sentence before it reaches the decoder, where
    it would fail as an opaque error deep in CTranslate2.

    Args:
        default: Code used when the operator just presses Enter.

    Returns:
        A normalised, lower-case language code.
    """
    raw = input(
        f"Language Code [Press Enter for {default!r}]: "
    ).strip().lower()
    if not raw:
        return default
    if len(raw) > 2:
        console.print(
            f"  [yellow]Using [bold]{default!r}[/bold] — expected a 2-letter "
            f"code like 'en' or 'sw', not a whole phrase.[/yellow]"
        )
        return default
    return raw


def create_output_dir(course_code: str) -> Path:
    """Create (if needed) and return ``./output/{course_code}_{YYYY-MM-DD}/``.

    Args:
        course_code: Sanitised course unit code.

    Returns:
        The output directory for today's take.
    """
    today = date.today().strftime("%Y-%m-%d")
    output_dir = Path("output") / f"{course_code}_{today}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def _confirm_overwrite(wav_path: Path) -> bool:
    """Ask before clobbering a recording from an earlier take today.

    Args:
        wav_path: The ``raw_audio.wav`` that would be overwritten.

    Returns:
        ``True`` if it does not exist or the user confirmed.
    """
    if not wav_path.exists():
        return True
    console.print(
        f"[bold yellow]⚠️  {wav_path} already exists and will be "
        f"OVERWRITTEN.[/bold yellow]",
        soft_wrap=True,
    )
    return input("Continue? [y/N]: ").strip().lower() in {"y", "yes"}


def select_device(recorder: AudioRecorder) -> int:
    """List loopback devices and prompt for one by integer index.

    Args:
        recorder: Recorder whose cached device list is shown.

    Returns:
        The selected device index.

    Raises:
        SystemExit: No loopback devices were found (code 1).
    """
    devices = recorder.get_loopback_devices()
    if not devices:
        console_err.print(
            "[bold red]No loopback audio devices found.[/bold red] Enable "
            "your speakers (or Stereo Mix) and retry.",
        )
        raise SystemExit(1)

    console.print("\n[bold]Available loopback devices:[/bold]")
    for index, device in enumerate(devices):
        console.print(
            f"  [bold cyan][{index}][/bold cyan] {device.name}  "
            f"({device.channels} ch)"
        )

    while True:
        raw = input("Select a device by number: ").strip()
        try:
            index = int(raw)
        except ValueError:
            console.print("  [yellow]Enter an integer, e.g. 0.[/yellow]")
            continue
        if 0 <= index < len(devices):
            return index
        console.print(
            f"  [yellow]Choose a number between 0 and {len(devices) - 1}.[/yellow]"
        )


def resolve_device_index(recorder: AudioRecorder | None = None) -> int:
    """Return the index of the first loopback device without prompting.

    The non-interactive counterpart to :func:`select_device`, for callers that
    have no terminal to prompt on -- the web backend, where the device has to be
    chosen for the operator or the run cannot start at all.

    First device is a policy, not a proof: a machine with several loopback
    endpoints (Speakers, Stereo Mix, a headset) will capture whichever the
    soundcard enumerates first, which may not be the one the operator wanted.
    That is why this is opt-in and why ``select_device`` remains the default
    everywhere a human is present.

    Args:
        recorder: Recorder whose cached device list to read. Constructed on
            demand when omitted.

    Returns:
        The index of the first available loopback device.

    Raises:
        PipelineError: No loopback devices were found. Failing here is
            deliberate: recording from an arbitrary default capture device would
            produce a silent ``raw_audio.wav``, and the run would only fail later
            with a confusing "no audio captured".

            ``PipelineError`` rather than ``SystemExit`` like
            :func:`select_device`, and the difference is load-bearing.
            ``SystemExit`` derives from ``BaseException``, not ``Exception``, so
            it would sail straight past the ``except Exception`` in
            ``server._run_in_background`` -- releasing the run lock on the way
            out but never updating the state, stranding the dashboard on
            "recording" for a run that never began. A machine with no loopback
            device is precisely the machine most likely to hit this.
    """
    if recorder is None:
        recorder = AudioRecorder()
    if not recorder.get_loopback_devices():
        console_err.print(
            "[bold red]No loopback audio devices found.[/bold red] Enable "
            "your speakers (or Stereo Mix) and retry.",
        )
        raise PipelineError("no loopback audio devices found")
    return 0


def run_status_loop(recorder: AudioRecorder) -> None:
    """Refresh the status line once per second until recording ends.

    Returns when the user presses Ctrl+C (KeyboardInterrupt propagates to
    the caller) or when the capture thread stops on its own after a device
    failure.

    Args:
        recorder: The active recorder; its ``is_recording`` event gates
            the loop.
    """
    started = time.monotonic()
    try:
        while recorder.is_recording.is_set():
            elapsed = int(time.monotonic() - started)
            minutes, seconds = divmod(elapsed, 60)
            # Written in place with a carriage return. Deliberately a plain
            # console.print rather than a ``Status`` spinner: the Live display
            # would repaint the line on its own refresh timer and fight this
            # one-second tick, and a Live display started here would have to be
            # torn down before every later print in the pipeline.
            console.print(
                "\r[bold yellow]●[/bold yellow] Recording in progress... "
                "Press [bold]Ctrl+C[/bold] to stop  "
                f"[bold cyan]{minutes:02d}:{seconds:02d}[/bold cyan]",
                end="",
                highlight=False,
            )
            time.sleep(1.0)
    finally:
        console.print()  # terminate the \r line before further output


# ---------------------------------------------------------------------------
# AI handoff (double Ctrl+C trap)
# ---------------------------------------------------------------------------


def _interrupt_protected(label: str, fn: Callable[[], Any]) -> Any:
    """Run one pipeline phase, absorbing the first Ctrl+C into a warning.

    A single Ctrl+C while the API is processing is treated as impatience,
    not an abort request: we warn and retry the phase so the pipeline can
    finish ("Let it finish"). A second Ctrl+C propagates so ``main()`` can
    exit safely — everything already flushed to disk is left untouched.

    Args:
        label: Human-readable phase name (used in the retry message).
        fn: Zero-argument callable performing the phase.

    Returns:
        Whatever ``fn`` returned.
    """
    try:
        return fn()
    except KeyboardInterrupt:
        console.print(
            f"\n[bold yellow]⚠️  Pipeline processing in progress ({label}).[/bold yellow] "
            "Aborting now may result in data loss. [italic]Let it finish.[/italic]"
        )
        return fn()  # single press only: give the phase one clean retry


# ---------------------------------------------------------------------------
# Programmatic entry point
# ---------------------------------------------------------------------------


def run_pipeline(
    course_code: str,
    language: str = DEFAULT_LANGUAGE,
    state_callback: Callable[[str], None] | None = None,
    *,
    device_id: int | None = None,
    confirm_overwrite: bool | None = None,
) -> str:
    """Run record → transcribe → synthesize and return the study-notes path.

    This is the reusable core. Everything that makes the pipeline *runnable*
    lives here; everything that makes it *presentable to a human* stays in
    :func:`main`. The two concerns are split so the same run can be driven by
    the CLI, by the HTTP backend in ``server.py``, or by a test, with no
    branching on "am I interactive?" inside the pipeline itself.

    Args:
        course_code: Sanitised course unit code, used to name the output
            directory.
        language: Spoken-language code forwarded to the transcriber.
        state_callback: Optional sink for pipeline states. Called with
            :data:`STATE_RECORDING`, :data:`STATE_TRANSCRIBING`,
            :data:`STATE_SYNTHESIZING` as each stage *begins*, and then exactly
            once more with either :data:`STATE_COMPLETE` or
            :data:`STATE_ERROR`. The terminal states are what make this usable
            for progress display: a consumer that only watched the three
            in-progress states could never tell a finished run from a wedged
            one.
        device_id: Loopback device index. ``None`` (the default) prompts via
            :func:`select_device`, which is what the CLI wants. A caller with no
            terminal must pass a real index -- see
            :func:`resolve_device_index`.
        confirm_overwrite: ``None`` (the default) prompts before clobbering an
            existing ``raw_audio.wav``. Pass a bool to decide in advance; a
            non-interactive caller has no way to ask.

    Returns:
        The path to the generated study notes, as a string.

    Raises:
        PipelineError: The run failed. ``exc.exit_code`` is the process exit
            code the CLI reports: 1 for a handled failure, 130 for a cancelled
            run.
    """
    # -- setup ---------------------------------------------------------------
    output_dir = create_output_dir(course_code)
    wav_path = output_dir / "raw_audio.wav"
    backup_path = output_dir / "emergency_backup.wav"
    console.print(
        f"[bold blue]ℹ[/bold blue] Output directory : {output_dir}",
        soft_wrap=True,
        highlight=False,
    )

    if confirm_overwrite is None:
        may_overwrite = _confirm_overwrite(wav_path)
    else:
        may_overwrite = confirm_overwrite
    if not may_overwrite:
        console.print("[bold yellow]Aborted[/bold yellow] — nothing was recorded.")
        _emit(state_callback, STATE_ERROR)
        raise PipelineError(f"overwrite declined for {wav_path}", 1)

    # -- instantiate the three components ------------------------------------
    recorder = AudioRecorder(emergency_backup_path=backup_path)
    transcriber = LocalTranscriber()
    synthesizer = LectureSynthesizer()

    if device_id is None:
        device_id = select_device(recorder)
    else:
        # Validated rather than trusted. An out-of-range index from a web client
        # would otherwise fail deep inside soundcard, after the output directory
        # had already been created, with an error that names no slide and no
        # device list.
        devices = recorder.get_loopback_devices()
        if not 0 <= device_id < len(devices):
            _emit(state_callback, STATE_ERROR)
            raise PipelineError(
                f"device_id {device_id} out of range "
                f"(0..{len(devices) - 1 if devices else 0})"
            )
    # Same cached list as the selection, so the index still points at this
    # device:
    device_name = recorder.get_loopback_devices()[device_id].name

    _emit(state_callback, STATE_RECORDING)
    console.print(f"[bold blue]ℹ[/bold blue] Recording from  : {device_name}")
    console.print(f"[bold blue]ℹ[/bold blue] Emergency backup: {backup_path}\n",
                  soft_wrap=True, highlight=False)

    # -- Phase A: recording (1st Ctrl+C = clean stop) -------------------------
    # Phase 5: visual capture runs alongside the audio recording loop.
    slide_catcher = SlideCatcher(output_dir=output_dir)
    slide_catcher.start()

    try:
        recorder.start_recording(device_id)
        run_status_loop(recorder)
    except KeyboardInterrupt:
        # First Ctrl+C — stop cleanly, then hand off to the AI pipeline.
        # Suppress further interrupts during the WAV flush so a panicked
        # double Ctrl+C cannot corrupt the file (Phase 1 behaviour).
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            recorder.stop_recording(wav_path)
        finally:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        console.print(
            "[bold green]🛑  Recording stopped.[/bold green] "
            "Initiating AI Transcription..."
        )
        captured_slides = slide_catcher.stop()
        console.print(
            f"[bold green]🛑[/bold green] Slide capture stopped. "
            f"Captured [bold]{len(captured_slides)}[/bold] unique slide(s)."
        )
    except (RuntimeError, OSError) as exc:
        slide_catcher.stop()
        console_err.print(f"\n[bold red]✗ Error:[/bold red] {exc}")
        _emit(state_callback, STATE_ERROR)
        raise PipelineError(f"capture failed: {exc}") from exc
    else:
        # Loop ended without Ctrl+C: the device dropped and the capture
        # thread stopped itself — flush whatever audio was buffered.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            recorder.stop_recording(wav_path)
        finally:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        console.print(
            "[bold yellow]⚠️  Recording ended on its own "
            "(possible device drop).[/bold yellow]"
        )
        captured_slides = slide_catcher.stop()
        console.print(
            f"[bold green]🛑[/bold green] Slide capture stopped. "
            f"Captured [bold]{len(captured_slides)}[/bold] unique slide(s)."
        )

    # -- Phase B: AI handoff (double-Ctrl+C protected) -----------------------
    if not (wav_path.exists() and wav_path.stat().st_size > 0):
        console.print(
            "[bold yellow]⚠️  No audio captured — skipping AI processing.[/bold yellow]"
        )
        _emit(state_callback, STATE_ERROR)
        raise PipelineError("no audio captured")

    transcript_path = output_dir / TRANSCRIPT_FILENAME
    notes_path = output_dir / f"{course_code}_{date.today():%Y-%m-%d}.md"
    slides_dir = output_dir / "slides"

    try:
        # -- Transcription & Diarization -----------------------------------
        # The longest silent stretch in the pipeline (tens of minutes on CPU),
        # so it is the one that most needs a visible "still working" signal.
        # The spinner sits *inside* _interrupt_protected, not around it, so the
        # deliberate single-Ctrl+C retry re-enters a live spinner rather than
        # dropping to a frozen prompt.
        _emit(state_callback, STATE_TRANSCRIBING)
        with _spinner("[bold yellow]Transcribing and diarizing...[/bold yellow]"):
            _interrupt_protected(
                "transcription",
                lambda: transcriber.transcribe_audio(wav_path, language=language),
            )
        console.print(
            "[bold green]✓[/bold green] Transcription complete → "
            f"[bold cyan]{transcript_path.name}[/bold cyan]"
        )

        # -- Academic Synthesis --------------------------------------------
        # NOTE: this lambda must *call* generate_notes. It previously only
        # built the argument tuple and returned it, so the whole synthesis
        # phase was a silent no-op -- no LLM call, no notes file -- while still
        # reporting "All stages completed" and printing a path that did not
        # exist. Covered by test_transcription_runs_before_synthesis.
        _emit(state_callback, STATE_SYNTHESIZING)
        with _spinner("[bold yellow]Synthesizing study notes...[/bold yellow]"):
            _interrupt_protected(
                "synthesis",
                lambda: synthesizer.generate_notes(
                    transcript_path, notes_path, slides_dir
                ),
            )
    except ValueError as exc:
        console.print(f"\n[bold yellow]⚠️  Synthesis skipped:[/bold yellow] {exc}")
        console.print("   [yellow]The raw transcript is still available at:[/yellow]")
        console.print(f"   {transcript_path}", soft_wrap=True, highlight=False)
        _emit(state_callback, STATE_ERROR)
        raise PipelineError(f"synthesis skipped: {exc}") from exc
    except KeyboardInterrupt:
        console.print(
            "\n[bold yellow]⚠️  Pipeline interrupted — saved files kept in:[/bold yellow]"
        )
        console.print(f"   {output_dir}", soft_wrap=True, highlight=False)
        _emit(state_callback, STATE_ERROR)
        raise PipelineError("pipeline interrupted", 130)

    _emit(state_callback, STATE_COMPLETE)
    console.print(f"[bold green]✅  All stages completed.[/bold green]\n")
    _kv("Audio      :", wav_path)
    _kv("Transcript :", transcript_path)
    _kv("Slides     :", f"{len(captured_slides)} saved in {slide_catcher.slides_dir}")
    _kv("Study Notes:", f"[bold]{notes_path}[/bold]")
    return str(notes_path)


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def _cli_state_callback(state: str) -> None:
    """Render a pipeline state as a labelled rule on the terminal.

    This is the CLI's whole contribution to the run: it maps the markup-free
    state tokens onto the presentation labels, which is why
    :func:`run_pipeline` never needs to know that rich exists.
    """
    _stage(_STATE_LABELS.get(state, state))


def main(language: str | None = None) -> int:
    """Run the pipeline from a terminal: prompt, then hand off to the core.

    Deliberately thin. The run itself belongs to :func:`run_pipeline`; what
    stays here is the part that only makes sense with a person attached -- the
    loopback warning, the interactive prompts, and the state-to-presentation
    translation.

    Args:
        language: Spoken-language code forwarded to the transcriber, or
            ``None`` to prompt for it interactively. Skipping language
            detection is a deliberate speed trade; see
            :data:`transcriber.DEFAULT_LANGUAGE`.

            ``None`` rather than a literal default is what makes the prompt
            order work. The alternative -- prompting in the ``__main__`` block
            and passing the result in, e.g. ``main(prompt_language())`` -- is
            correct Python but asks the *language* question first, because
            arguments are evaluated before the call. That put the language
            prompt above the course code, and the two questions are related
            enough that whoever is recording a Swahili lecture should name the
            course before being told the language. The prompt therefore lives
            here, where it can be ordered against :func:`prompt_course_code`,
            and a caller who already knows the language can skip the question
            entirely by passing it.

    Returns:
        Process exit code: 0 success, 1 error, 130 cancelled/interrupted.
    """
    _configure_streams()
    _enable_windows_ansi()
    print_warning_banner()

    # ---- setup: course code, then language --------------------------------
    _stage(STAGE_INIT)
    course_code = prompt_course_code()
    if language is None:
        language = prompt_language()

    try:
        run_pipeline(course_code, language, _cli_state_callback)
    except PipelineError as exc:
        # The operator-facing wording was already printed by run_pipeline at the
        # raise site, where the output directory and the failing path were in
        # scope. All that is left is to translate the failure into a code.
        return exc.exit_code
    return 0


if __name__ == "__main__":
    try:
        # The language prompt is *not* made here. Doing so -- e.g.
        # ``main(prompt_language())`` -- is correct Python but evaluates the
        # argument before main() is entered, which asks for the language before
        # the course code. ``main()`` prompts for it, in order, instead; pass
        # ``main(language="sw")`` to skip the question entirely.
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        console.print("\n[bold yellow]Cancelled — exiting.[/bold yellow]")
        raise SystemExit(130)
