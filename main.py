"""CLI entry point for the lecture recording pipeline (Phase 1).

Flow: print the loopback warning, collect the course unit code, create
``./output/{course_code}_{YYYY-MM-DD}/``, list loopback devices for
selection, record until Ctrl+C, then save ``raw_audio.wav``.
"""

from __future__ import annotations

import re
import signal
import sys
import time
from datetime import date
from pathlib import Path

from recorder import AudioRecorder


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


def main() -> int:
    """Run the Phase 1 CLI: warn → course code → devices → record → save.

    Returns:
        Process exit code: 0 success, 1 error, 130 cancelled.
    """
    _configure_streams()
    _enable_windows_ansi()
    print_warning_banner()

    course_code = prompt_course_code()
    output_dir = create_output_dir(course_code)
    wav_path = output_dir / "raw_audio.wav"
    backup_path = output_dir / "emergency_backup.wav"
    print(f"Output directory : {output_dir}")

    if not _confirm_overwrite(wav_path):
        print("Aborted — nothing was recorded.")
        return 1

    recorder = AudioRecorder(emergency_backup_path=backup_path)

    device_id = select_device(recorder)
    # Same cached list as above, so the index still points at this device:
    device_name = recorder.get_loopback_devices()[device_id].name
    print(f"\nRecording from  : {device_name}")
    print(f"Emergency backup: {backup_path}\n")

    try:
        recorder.start_recording(device_id)
        run_status_loop(recorder)
    except KeyboardInterrupt:
        print("\n■ Ctrl+C received — stopping recording...")
    except (RuntimeError, OSError) as exc:
        print(f"\nError: {exc}", file=sys.stderr)
        return 1
    finally:
        # stop_recording() is safe in every state (idempotent, thread-safe),
        # so it always runs — even if start failed or Ctrl+C raced the start.
        # A second Ctrl+C during the flush would abort the WAV write:
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        saved = recorder.stop_recording(wav_path)
        if saved.exists() and saved.stat().st_size > 0:
            size_mb = saved.stat().st_size / 1_048_576
            print(f"✅ Saved: {saved} ({size_mb:.1f} MB)")
        else:
            print(f"⚠️ No audio captured — nothing written to {saved}")

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (KeyboardInterrupt, EOFError):
        print("\nCancelled — exiting.")
        raise SystemExit(130)
