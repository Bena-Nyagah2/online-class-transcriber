"""Audio capture for the lecture transcription pipeline (Phase 1).

Captures system-audio loopback (WASAPI on Windows) on a daemon thread and
saves 16 kHz, mono, 16-bit PCM WAV files.

Architectural notes
-------------------
* ``soundcard`` does not export its device class publicly, so this module
  type-checks against the small :class:`LoopbackDevice` protocol instead.
* The soundcard WASAPI backend is documented to record garbage when asked
  for a single channel, so capture happens at the device's native channel
  count and is downmixed to mono in NumPy.
* The capture thread is a daemon and is always joined by
  :meth:`AudioRecorder.stop_recording`, so no zombie threads survive the CLI.
"""

from __future__ import annotations

import sys
import threading
import warnings
from contextlib import AbstractContextManager
from pathlib import Path
from typing import Any, Optional, Protocol

import numpy as np
import soundcard as sc
import soundfile as sf

SAMPLE_RATE: int = 16_000
"""Output sample rate in Hz — the standard rate for ASR pipelines."""

CHANNELS: int = 1
"""Number of channels in the written WAV (mono)."""

CHUNK_FRAMES: int = 512
"""Frames requested per ``record()`` call (~32 ms at 16 kHz)."""

OPEN_TIMEOUT_SECONDS: float = 5.0
"""How long ``start_recording`` waits for the stream to open."""

STOP_JOIN_TIMEOUT_SECONDS: float = 5.0
"""How long ``stop_recording`` waits for the capture thread to exit."""

EMERGENCY_BACKUP_FILENAME: str = "emergency_backup.wav"
"""Default filename for the mid-lecture device-failure dump."""


class LoopbackDevice(Protocol):
    """Structural type for a ``soundcard`` device.

    ``soundcard``'s concrete classes (``_Microphone``) are private and
    differ per platform backend, so this protocol describes only the part
    of the API this module actually uses.
    """

    @property
    def name(self) -> str:
        """Human-readable device name, e.g. ``Speakers (Realtek(R) Audio)``."""
        ...

    @property
    def channels(self) -> int:
        """Native channel count of the device (loopback devices are stereo)."""
        ...

    def recorder(
        self,
        samplerate: int,
        channels: Optional[int] = ...,
        blocksize: Optional[int] = ...,
        exclusive_mode: bool = ...,
    ) -> AbstractContextManager[Any]:
        """Open (and context-manage) a capture stream at ``samplerate``."""
        ...


def _to_mono(chunk: np.ndarray) -> np.ndarray:
    """Downmix one ``frames x channels`` chunk to a 1-D mono float32 array.

    WASAPI single-channel capture is documented to produce garbage in
    ``soundcard``, so we always capture at the device's native channel
    count and average the channels here.

    Args:
        chunk: Audio data as returned by ``Recorder.record`` — normally a
            ``frames x channels`` float32 array scaled to 0 dBFS.

    Returns:
        A contiguous 1-D float32 array of ``frames`` mono samples.
    """
    if chunk.ndim > 1 and chunk.shape[1] > 1:
        chunk = chunk.mean(axis=1)
    return np.ascontiguousarray(chunk, dtype=np.float32)


def _stack_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate buffered chunks into one 1-D mono array.

    Args:
        chunks: Mono chunks previously produced by :func:`_to_mono`.

    Returns:
        The concatenated audio, or an empty float32 array.
    """
    if not chunks:
        return np.empty(0, dtype=np.float32)
    return np.concatenate(chunks, axis=0)


class AudioRecorder:
    """Records system-audio loopback to WAV on a background daemon thread.

    Usage::

        recorder = AudioRecorder()
        devices = recorder.get_loopback_devices()
        recorder.start_recording(0)          # non-blocking
        ...
        recorder.stop_recording("talk.wav")  # flushes the buffer and saves
    """

    def __init__(self, emergency_backup_path: Optional[Path] = None) -> None:
        """Create a recorder in the stopped state.

        Args:
            emergency_backup_path: Where to dump the buffer if the audio
                device fails mid-recording. Defaults to
                ``./emergency_backup.wav`` in the current directory.
        """
        self.is_recording: threading.Event = threading.Event()
        # Set while the capture loop should run; ``stop_recording`` clears it.

        self._buffer: list[np.ndarray] = []
        self._buffer_lock = threading.Lock()
        self._thread: Optional[threading.Thread] = None
        self._devices: Optional[list[LoopbackDevice]] = None

        # Handshake between start_recording() and the capture thread:
        self._open_complete = threading.Event()
        self._open_error: Optional[Exception] = None

        self._emergency_backup_path = Path(
            emergency_backup_path or EMERGENCY_BACKUP_FILENAME
        )
        self._finalized: bool = False
        self._last_path: Optional[Path] = None

        # soundcard forces its runtime warnings to print on every occurrence;
        # during a multi-hour take, "data discontinuity" warnings would flood
        # over the status line. Show each distinct warning only once.
        soundcard_warning = getattr(sc, "SouncardRuntimeWarning", RuntimeWarning)
        warnings.simplefilter("once", soundcard_warning)

    def get_loopback_devices(self) -> list[LoopbackDevice]:
        """Return the loopback (system-audio) capture devices.

        On Windows, WASAPI exposes one virtual loopback device per speaker
        via ``include_loopback=True``; regular microphones are filtered out
        so every entry captures *system* audio. Other platforms expose no
        loopback API in ``soundcard``, so all microphones are returned as a
        best-effort fallback.

        The result is cached: repeated calls return the same list so that
        index-based selection stays stable between listing and recording.

        Returns:
            A list of devices; index it to pick a ``device_id`` for
            :meth:`start_recording`.
        """
        if self._devices is None:
            if sys.platform == "win32":
                found: list[Any] = list(sc.all_microphones(include_loopback=True))
                self._devices = [
                    device for device in found if getattr(device, "isloopback", False)
                ]
            else:
                self._devices = list(sc.all_microphones())
        return list(self._devices)

    def start_recording(self, device_id: int) -> None:
        """Start capturing on a daemon thread (non-blocking).

        The WASAPI stream is opened *inside* the capture thread, but this
        method waits for a handshake, so device-open failures surface here,
        on the caller's thread, instead of dying silently in the background.

        Args:
            device_id: Index into :meth:`get_loopback_devices`.

        Raises:
            IndexError: ``device_id`` is out of range.
            RuntimeError: A recording is already running, the device failed
                to open, or opening it timed out.
        """
        if self._thread is not None and self._thread.is_alive():
            raise RuntimeError("a recording is already in progress")

        devices = self.get_loopback_devices()
        if not 0 <= device_id < len(devices):
            raise IndexError(
                f"device_id {device_id} out of range (0..{len(devices) - 1})"
            )

        with self._buffer_lock:
            self._buffer = []
        self._finalized = False
        self._last_path = None
        self._open_error = None
        self._open_complete.clear()
        self.is_recording.set()

        self._thread = threading.Thread(
            target=self._capture_loop,
            args=(devices[device_id],),
            name="AudioRecorder.capture",
            daemon=True,  # dies with the interpreter — never a zombie
        )
        self._thread.start()

        try:
            opened = self._open_complete.wait(timeout=OPEN_TIMEOUT_SECONDS)
        except KeyboardInterrupt:
            self.is_recording.clear()  # don't leave the thread capturing
            raise

        if not opened:
            self.is_recording.clear()
            raise RuntimeError(
                f"timed out opening audio device after {OPEN_TIMEOUT_SECONDS:.0f}s"
            )
        if self._open_error is not None:
            raise RuntimeError(
                f"failed to open audio device {devices[device_id].name!r}: "
                f"{self._open_error!r}"
            ) from self._open_error

    def stop_recording(self, output_filepath: Path | str) -> Path:
        """Stop recording, flush the buffer, and write a 16 kHz mono WAV.

        Safe to call when capture already stopped on its own (device drop):
        the thread join returns immediately and everything captured before
        the failure is still written. Calling it twice returns the path of
        the first write instead of overwriting it with an empty file.

        Args:
            output_filepath: Destination for the ``.wav`` file; parent
                directories are created if missing.

        Returns:
            The path the recording was written to.

        Note:
            A second Ctrl+C during the flush would abort the WAV write, so
            the CLI disables SIGINT immediately before calling this.
        """
        self.is_recording.clear()

        thread = self._thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=STOP_JOIN_TIMEOUT_SECONDS)
            if thread.is_alive():  # pathological backend — daemon will die anyway
                print(
                    "warning: capture thread did not stop within "
                    f"{STOP_JOIN_TIMEOUT_SECONDS:.0f}s; the audio backend "
                    "may be wedged.",
                    file=sys.stderr,
                    flush=True,
                )

        if self._finalized and self._last_path is not None:
            return self._last_path

        with self._buffer_lock:
            chunks = list(self._buffer)
            self._buffer = []  # release memory now that it is flushed
        audio = _stack_chunks(chunks)

        path = Path(output_filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if audio.shape[0] > 0:
            sf.write(str(path), audio, SAMPLE_RATE, subtype="PCM_16")

        self._finalized = True
        self._last_path = path
        return path

    # ------------------------------------------------------------------
    # capture thread
    # ------------------------------------------------------------------

    def _capture_loop(self, device: LoopbackDevice) -> None:
        """Daemon-thread body: capture until stopped or the device fails.

        Args:
            device: The loopback device to record from.
        """
        try:
            # ``blocksize`` is left unset so WASAPI chooses its default
            # engine period, which is always valid in shared mode.
            with device.recorder(samplerate=SAMPLE_RATE) as microphone:
                self._open_complete.set()  # release start_recording()
                while self.is_recording.is_set():
                    chunk = microphone.record(numframes=CHUNK_FRAMES)
                    with self._buffer_lock:
                        self._buffer.append(_to_mono(chunk))
        except Exception as exc:
            if not self._open_complete.is_set():
                # Failed before the stream ever opened: report it on the
                # caller's thread via start_recording()'s handshake.
                self._open_error = exc
                return
            # Defense (device drop): the device disconnected mid-lecture —
            # salvage the buffer to disk immediately and alert the user.
            self.is_recording.clear()
            backup_path = self._try_emergency_backup()
            if backup_path is not None:
                saved_line = f"          Buffered audio saved to: {backup_path}"
            else:
                saved_line = "          Emergency BACKUP FAILED — buffered audio may be lost!"
            print(
                "\n‼️ ALERT: audio device failed mid-recording "
                f"({exc!r}).\n"
                f"{saved_line}\n"
                "          Recording has been stopped.",
                flush=True,
            )
        finally:
            self._open_complete.set()  # never leave start_recording() hanging

    def _try_emergency_backup(self) -> Optional[Path]:
        """Best-effort dump of the buffer to the emergency backup WAV.

        Returns:
            The backup path on success, or ``None`` if the dump failed.
        """
        try:
            with self._buffer_lock:
                chunks = list(self._buffer)  # keep buffer: stop_recording
            audio = _stack_chunks(chunks)    # still flushes it later
            self._emergency_backup_path.parent.mkdir(parents=True, exist_ok=True)
            sf.write(
                str(self._emergency_backup_path),
                audio,
                SAMPLE_RATE,
                subtype="PCM_16",
            )
            return self._emergency_backup_path
        except Exception:
            return None
