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
from datetime import date
from pathlib import Path
from typing import Any, Callable

from recorder import AudioRecorder
from transcriber import LocalTranscriber, TRANSCRIPT_FILENAME
from summarizer import LectureSynthesizer
from vision import SlideCatcher


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
    print(f"\n\033[1;31m{bar}")
    print(
        "⚠️ WARNING: Mute system notifications (Slack, Email). "
        "Loopback captures ALL OS audio."
    )
    print(f"{bar}\033[0m\n")


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
            print("  Course code cannot be empty — try again.")
            continue
        safe = _sanitize_course_code(raw)
        if safe != raw:
            print(f"  Note: using {safe!r} as the folder name.")
        return safe


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
    print(f"⚠️ {wav_path} already exists and will be OVERWRITTEN.")
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
        print(
            "No loopback audio devices found. Enable your speakers "
            "(or Stereo Mix) and retry.",
            file=sys.stderr,
        )
        raise SystemExit(1)

    print("\nAvailable loopback devices:")
    for index, device in enumerate(devices):
        print(f"  [{index}] {device.name}  ({device.channels} ch)")

    while True:
        raw = input("Select a device by number: ").strip()
        try:
            index = int(raw)
        except ValueError:
            print("  Enter an integer, e.g. 0.")
            continue
        if 0 <= index < len(devices):
            return index
        print(f"  Choose a number between 0 and {len(devices) - 1}.")


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
            print(
                "\r● Recording in progress... Press Ctrl+C to stop  "
                f"[{minutes:02d}:{seconds:02d}]",
                end="",
                flush=True,
            )
            time.sleep(1.0)
    finally:
        print()  # terminate the \r status line before further output


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
        print(
            f"\n⚠️ Pipeline processing in progress ({label}). "
            "Aborting now may result in data loss. Let it finish.",
            flush=True,
        )
        return fn()  # single press only: give the phase one clean retry


# ---------------------------------------------------------------------------
# main entry point
# ---------------------------------------------------------------------------


def main() -> int:
    """Run the Phase 4 pipeline: record → transcribe → synthesize.

    Returns:
        Process exit code: 0 success, 1 error, 130 cancelled/interrupted.
    """
    _configure_streams()
    _enable_windows_ansi()
    print_warning_banner()

    # ---- setup: course code + output directory (Phase 1 logic) ------------
    course_code = prompt_course_code()
    output_dir = create_output_dir(course_code)
    wav_path = output_dir / "raw_audio.wav"
    backup_path = output_dir / "emergency_backup.wav"
    print(f"Output directory : {output_dir}")

    if not _confirm_overwrite(wav_path):
        print("Aborted — nothing was recorded.")
        return 1

    # ---- instantiate the three components ----------------------------------
    recorder = AudioRecorder(emergency_backup_path=backup_path)
    transcriber = LocalTranscriber()
    synthesizer = LectureSynthesizer()

    device_id = select_device(recorder)
    # Same cached list as above, so the index still points at this device:
    device_name = recorder.get_loopback_devices()[device_id].name
    print(f"\nRecording from  : {device_name}")
    print(f"Emergency backup: {backup_path}\n")

    # ---- Phase A: recording (1st Ctrl+C = clean stop) ----------------------
    # Phase 5: visual capture runs alongside the audio recording loop.
    slide_catcher = SlideCatcher(output_dir=output_dir, interval_seconds=10.0)
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
        print("🛑 Recording stopped. Initiating AI Transcription...")
        captured_slides = slide_catcher.stop()
        print(
            f"🛑 Slide capture stopped. "
            f"Captured {len(captured_slides)} unique slide(s)."
        )
    except (RuntimeError, OSError) as exc:
        slide_catcher.stop()
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    else:
        # Loop ended without Ctrl+C: the device dropped and the capture
        # thread stopped itself — flush whatever audio was buffered.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        try:
            recorder.stop_recording(wav_path)
        finally:
            signal.signal(signal.SIGINT, signal.SIG_DFL)
        print("⚠️ Recording ended on its own (possible device drop).")
        captured_slides = slide_catcher.stop()
        print(
            f"🛑 Slide capture stopped. "
            f"Captured {len(captured_slides)} unique slide(s)."
        )

    # ---- Phase B: AI handoff (double-Ctrl+C protected) ---------------------
    if not (wav_path.exists() and wav_path.stat().st_size > 0):
        print("⚠️ No audio captured — skipping AI processing.")
        return 1

    transcript_path = output_dir / TRANSCRIPT_FILENAME
    notes_path = output_dir / f"{course_code}_{date.today():%Y-%m-%d}.md"

    try:
        _interrupt_protected(
            "transcription",
            lambda: transcriber.transcribe_audio(wav_path),
        )
        print("🧠 Transcription complete. Synthesizing Study Notes...")

        _interrupt_protected(
            "synthesis",
            lambda: synthesizer.generate_notes(transcript_path, notes_path),
        )
    except ValueError as exc:
        print(f"\n⚠️ Synthesis skipped: {exc}")
        print(f"   The raw transcript is still available at: {transcript_path}")
        return 1
    except KeyboardInterrupt:
        print(f"\n⚠️ Pipeline interrupted — saved files kept in: {output_dir}")
        return 130

    print(
        f"\n{'=' * 60}\n"
        "✅ PIPELINE COMPLETE\n"
        f"   Audio      : {wav_path}\n"
        f"   Transcript : {transcript_path}\n"
        f"   Slides     : {len(captured_slides)} saved in {slide_catcher.slides_dir}\n"
        f"   Study Notes: {notes_path}\n"
        f"{'=' * 60}"
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled — exiting.")
        raise SystemExit(130)