"""Audio capture for the lecture transcription pipeline (Phase 1).

Records **two audio sources simultaneously** — the system-audio loopback
(WASAPI on Windows) and the room microphone — mixes them down to a single
16 kHz, mono, 16-bit PCM WAV. Both streams are required for WhisperX speaker
diarization: the loopback carries the lecturer's Zoom/Teams/YouTube audio
while the microphone carries the students in the physical room, and
pyannote can only tell those voices apart if both are present in the file.

Architectural notes
-------------------
* ``soundcard`` does not export its device class publicly, so this module
  type-checks against the small :class:`LoopbackDevice` protocol instead.
* The soundcard WASAPI backend is documented to record garbage when asked
  for a single channel, so capture happens at the device's native channel
  count and is downmixed to mono in NumPy.
* **One capture thread per device.** Polling both recorders from a single
  loop would block on each ``record()`` call in turn, so the loop would run
  at half wall-clock speed and the finished WAV would play back at double
  speed with the two sources drifting apart. Independent threads let each
  backend block at its own real-time pace; the streams are only merged once
  at flush time, after the take is over.
* The capture threads are daemons and are always joined by
  :meth:`AudioRecorder.stop_recording`, so no zombie threads survive the CLI.
"""

from __future__ import annotations

import os
import sys
import threading
import time
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

REPLACE_RETRY_ATTEMPTS: int = 5
"""How many times to retry an ``os.replace`` that Windows refused.

Five attempts with a growing delay spans a few hundred milliseconds, which is
long enough for an indexer or scanner to release a handle and short enough that
an operator would notice if a flush genuinely stalled.
"""

REPLACE_RETRY_DELAY_SECONDS: float = 0.05
"""Base delay for the ``os.replace`` retry backoff (grows linearly per attempt)."""

EMERGENCY_BACKUP_FILENAME: str = "emergency_backup.wav"
"""Default filename for the mid-lecture device-failure dump."""

CHECKPOINT_INTERVAL_SECONDS: float = 30.0
"""How often the mixed audio so far is flushed to the emergency backup.

Why this exists: before it, *nothing* reached disk until
:meth:`AudioRecorder.stop_recording` wrote the final file in one shot, so a
power loss, a kernel panic, or a pulled battery during a 90-minute take
destroyed 100% of the audio — the in-memory chunks were the only copy. The
device-drop backup did not help, because it only fires on an *exception* in the
capture thread, which a power cut by definition never raises.

The cost of a power cut is now bounded by this interval rather than being total.
Thirty seconds is chosen against the write cost: each flush mixes and rewrites
the whole take, so this is O(duration) work per interval, and the resulting
quadratic I/O is why the value is not smaller. A lecture is a one-shot,
write-once workload, so trading I/O for recoverability is the right side of
that trade.
"""

AUTO_MICROPHONE: Any = object()
"""Sentinel: "use :meth:`AudioRecorder.get_default_microphone`".

A distinct object is required because ``None`` is a meaningful value here —
it explicitly disables microphone capture — so the default argument cannot
double as the "not supplied" marker.
"""


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


def _replace_with_retry(temp: Path, path: Path) -> None:
    """``os.replace`` the temp file into place, retrying on Windows contention.

    POSIX has no equivalent problem, but on Windows ``os.replace`` fails with
    ``PermissionError`` whenever *anything* holds the destination open --
    another reader, an indexer, a virus scanner passing through. Those handles
    are transient by nature, so a short bounded backoff turns a spurious
    failure into a slightly slower write.

    The retry is bounded and the final error is re-raised, so a genuinely
    stuck destination still surfaces to the caller (which is the checkpoint
    loop, where it is warned about and retried on the next interval) instead of
    hanging the recording.
    """
    for attempt in range(REPLACE_RETRY_ATTEMPTS):
        try:
            os.replace(temp, path)
            return
        except PermissionError:
            if attempt == REPLACE_RETRY_ATTEMPTS - 1:
                raise
            time.sleep(REPLACE_RETRY_DELAY_SECONDS * (attempt + 1))


def _write_wav_atomic(path: Path, audio: np.ndarray) -> None:
    """Write ``audio`` to ``path`` so the file is never observed half-written.

    ``soundfile.write`` truncates the destination and streams the header and
    data into it, so a crash partway through leaves either a 0-byte file or one
    whose RIFF length fields claim more data than the file contains -- the two
    exact outcomes this durability work exists to prevent. Writing to a sibling
    temp file and ``os.replace``-ing it in makes the swap atomic on both POSIX
    and Windows, so ``path`` is only ever the previous good file or the new
    good file.

    The temp name is derived from the destination rather than randomised so a
    leftover from a killed run is reused (and overwritten) instead of
    accumulating one per crash.

    ``format="WAV"`` is passed explicitly and is load-bearing, not decoration:
    soundfile otherwise infers the container from the *filename extension*, and
    the temp name ends in ``.partial`` -- which it rejects with
    ``TypeError: No format specified and unable to get format from file
    extension``. Naming the format directly keeps the atomic-write trick from
    depending on the one thing it necessarily changes.
    """
    temp = path.with_name(f"{path.name}.partial")
    sf.write(str(temp), audio, SAMPLE_RATE, subtype="PCM_16", format="WAV")
    _replace_with_retry(temp, path)


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


def _combine_streams(*streams: np.ndarray) -> np.ndarray:
    """Average any number of mono capture buffers into one mono array.

    The two capture threads start together and both run at
    :data:`SAMPLE_RATE`, so their buffers should stay within a chunk of each
    other in length. Any residual difference is absorbed by zero-padding the
    shorter stream, which preserves the total duration of the longer one
    instead of truncating real audio.

    Averaging (rather than summing) keeps the result inside [-1, 1] so a loud
    passage on the loopback cannot clip the lecturer's microphone. Each of the
    two sources therefore sits ~6 dB lower whenever both are active, which is
    the right trade for a lecture hall where the microphone is the channel
    that must never distort.

    Args:
        *streams: Mono arrays produced by :func:`_to_mono` (or empty arrays
            for a source that never captured anything).

    Returns:
        A 1-D float32 array holding the average of the inputs, clipped to
        [-1, 1]. Empty inputs are ignored; if every input is empty the result
        is an empty float32 array.
    """
    usable = [s for s in streams if s is not None and s.size > 0]
    if not usable:
        return np.empty(0, dtype=np.float32)

    # Absorb length drift by padding the shorter streams with trailing zeros.
    target = max(s.size for s in usable)
    padded = [
        s if s.size == target
        else np.concatenate([s, np.zeros(target - s.size, dtype=np.float32)])
        for s in usable
    ]

    mixed = np.mean(np.stack(padded, axis=0), axis=0, dtype=np.float32)
    # np.clip guards against any float32 rounding that pushes a sample past
    # full scale once the two streams are averaged.
    return np.ascontiguousarray(np.clip(mixed, -1.0, 1.0), dtype=np.float32)


class AudioRecorder:
    """Records system-audio loopback **and** the microphone to one WAV.

    Each source gets its own capture thread so neither can starve the other;
    they are averaged into a single mono track when the take is flushed::

        recorder = AudioRecorder()
        devices = recorder.get_loopback_devices()
        recorder.start_recording(0)          # non-blocking, mic auto-detected
        ...
        recorder.stop_recording("talk.wav")  # mixes both streams and saves
    """

    def __init__(self, emergency_backup_path: Optional[Path] = None) -> None:
        """Create a recorder in the stopped state.

        Args:
            emergency_backup_path: Where to dump the buffer if an audio
                device fails mid-recording. Defaults to
                ``./emergency_backup.wav`` in the current directory.
        """
        self.is_recording: threading.Event = threading.Event()
        # Set while the capture loops should run; ``stop_recording`` clears it.

        self._buffer_lock = threading.Lock()
        self._devices: Optional[list[LoopbackDevice]] = None

        # Separate per-source chunk buffers, plus the threads draining them.
        self._system_buffer: list[np.ndarray] = []
        self._mic_buffer: list[np.ndarray] = []
        self._system_thread: Optional[threading.Thread] = None
        self._mic_thread: Optional[threading.Thread] = None
        self._mic_device: Optional[LoopbackDevice] = None
        self._mic_active: bool = False
        self._mic_failed: bool = False

        # Handshake between start_recording() and the capture threads. Both
        # threads must report in before the caller's wait() is released.
        self._open_lock = threading.Lock()
        self._streams_expected: int = 0
        self._streams_opened: int = 0
        self._open_complete = threading.Event()
        self._open_error: Optional[Exception] = None

        self._emergency_backup_path = Path(
            emergency_backup_path or EMERGENCY_BACKUP_FILENAME
        )
        self._finalized: bool = False
        self._last_path: Optional[Path] = None

        # Periodic durability flush. ``_checkpoint_stop`` is cleared by
        # stop_recording() to retire the thread; the thread itself only ever
        # exits through that flag, so it can never outlive the recorder.
        self._checkpoint_stop = threading.Event()
        self._checkpoint_thread: Optional[threading.Thread] = None

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

    def get_default_microphone(self) -> LoopbackDevice:
        """Return the OS default input device for the room microphone.

        Kept as a method (rather than calling ``sc.default_microphone()``
        inline) so tests can inject a fake device without touching hardware.

        Returns:
            The default ``soundcard`` microphone device.
        """
        return sc.default_microphone()

    def start_recording(
        self,
        device_id: int,
        microphone: Optional[LoopbackDevice] = AUTO_MICROPHONE,
    ) -> None:
        """Start capturing loopback **and** microphone on daemon threads.

        The loopback stream is authoritative: if it cannot be opened the call
        fails loudly, preserving the original single-source contract. The
        microphone is best-effort — a machine with no usable input (or a
        denied permission prompt) degrades to system-audio-only capture with a
        warning instead of losing the whole lecture.

        The streams are opened *inside* their capture threads, but this method
        waits for a handshake, so device-open failures surface here, on the
        caller's thread, instead of dying silently in the background.

        Args:
            device_id: Index into :meth:`get_loopback_devices`.
            microphone: Microphone device to record alongside the loopback.
                Defaults to :data:`AUTO_MICROPHONE`, which resolves the OS
                default input. Pass ``None`` to record system audio only.

        Raises:
            IndexError: ``device_id`` is out of range.
            RuntimeError: A recording is already running, the loopback
                device failed to open, or opening it timed out.
        """
        if self._is_capturing():
            raise RuntimeError("a recording is already in progress")

        devices = self.get_loopback_devices()
        if not 0 <= device_id < len(devices):
            raise IndexError(
                f"device_id {device_id} out of range (0..{len(devices) - 1})"
            )

        with self._buffer_lock:
            self._system_buffer = []
            self._mic_buffer = []
        self._finalized = False
        self._last_path = None
        self._open_error = None
        self._mic_active = False
        self._mic_failed = False
        # Drop references to any previous take's threads so a reused recorder
        # cannot report a dead thread as the live microphone stream.
        self._system_thread = None
        self._mic_thread = None
        self._mic_device = None

        # The loopback is mandatory; the microphone only joins if a device
        # was resolved. ``None`` disables it outright; the sentinel triggers a
        # default-device lookup, done here so a raising lookup is reported on
        # the caller's thread rather than inside a daemon.
        if microphone is AUTO_MICROPHONE:
            try:
                self._mic_device = self.get_default_microphone()
            except Exception as exc:  # no input device / backend unavailable
                self._mic_device = None
                print(
                    f"\n⚠️ No default microphone found ({exc!r}) — "
                    "recording system audio only. Diarization will not be "
                    "able to separate room voices.",
                    file=sys.stderr,
                    flush=True,
                )
        else:
            self._mic_device = microphone

        self._streams_expected = 2 if self._mic_device is not None else 1
        with self._open_lock:
            self._streams_opened = 0
        self._open_complete.clear()
        self.is_recording.set()

        # System audio first: its thread is the one whose failure is fatal.
        self._system_thread = threading.Thread(
            target=self._capture_stream,
            args=(devices[device_id], self._system_buffer, "system audio"),
            name="AudioRecorder.system",
            daemon=True,  # dies with the interpreter — never a zombie
        )
        self._system_thread.start()

        if self._mic_device is not None:
            self._mic_thread = threading.Thread(
                target=self._capture_stream,
                args=(self._mic_device, self._mic_buffer, "microphone"),
                name="AudioRecorder.microphone",
                daemon=True,
            )
            self._mic_thread.start()

        try:
            opened = self._open_complete.wait(timeout=OPEN_TIMEOUT_SECONDS)
        except KeyboardInterrupt:
            self.is_recording.clear()  # don't leave the threads capturing
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

        self._mic_active = self._mic_thread is not None and not self._mic_failed
        print(
            f"🎙️  Capturing: {devices[device_id].name} + "
            f"{getattr(self._mic_device, 'name', 'microphone') if self._mic_active else 'no microphone'}",
            flush=True,
        )

        # Started only once the streams are confirmed open, so the periodic
        # flush never races a failed handshake and writes an empty backup over
        # a previous take's.
        self._start_checkpointing()

    def _is_capturing(self) -> bool:
        """Return True if either capture thread is still running."""
        return any(
            thread is not None and thread.is_alive()
            for thread in (self._system_thread, self._mic_thread)
        )

    # ------------------------------------------------------------------
    # periodic durability flush
    # ------------------------------------------------------------------

    def _start_checkpointing(self) -> None:
        """Launch the background thread that keeps the backup current."""
        self._checkpoint_stop.clear()
        self._checkpoint_thread = threading.Thread(
            target=self._checkpoint_loop,
            name="AudioRecorder.checkpoint",
            daemon=True,
        )
        self._checkpoint_thread.start()

    def _stop_checkpointing(self) -> None:
        """Retire the checkpoint thread and wait for it to leave the file alone.

        Joining is not optional. Without it, stop_recording could begin its
        final write while a flush is midway, and the two would race for the
        backup path -- leaving a half-written backup next to a good final WAV,
        which is precisely the confusing state this mechanism exists to avoid.
        """
        self._checkpoint_stop.set()
        thread = self._checkpoint_thread
        if thread is not None and thread.is_alive():
            thread.join(timeout=STOP_JOIN_TIMEOUT_SECONDS)
        self._checkpoint_thread = None

    def _checkpoint_loop(self) -> None:
        """Flush the mixed audio to the backup path every interval, until stopped.

        A failed flush is reported once and then *retried on the next interval*
        rather than ending the loop. That is the whole point of the mechanism:
        it is insurance against a power cut, so a transient failure -- a full
        disk, an antivirus scanner holding the file on Windows -- must not
        quietly leave the operator unprotected for the remaining hour. Going
        quiet after the first warning keeps a persistent problem from printing
        once per interval for the rest of the lecture, while the retry means
        recovery is automatic and invisible.
        """
        warned = False
        while not self._checkpoint_stop.wait(timeout=CHECKPOINT_INTERVAL_SECONDS):
            # Stopped cleanly while we were sleeping: the final write is
            # happening or has happened, so do not race it.
            if not self.is_recording.is_set():
                return
            try:
                if not self._flush_backup():
                    return
                warned = False
            except Exception as exc:  # noqa: BLE001 - daemon thread boundary
                if not warned:
                    print(
                        f"\n⚠️  Periodic audio checkpoint failed ({exc!r}) — "
                        "continuing to record, but a power loss would now lose "
                        f"more than the last "
                        f"{CHECKPOINT_INTERVAL_SECONDS:.0f}s. Will keep retrying.",
                        file=sys.stderr,
                        flush=True,
                    )
                    warned = True

    def _flush_backup(self) -> bool:
        """Write everything captured so far to the emergency backup path.

        Returns:
            ``False`` when there is nothing worth writing yet, so the caller can
            stop looping rather than churn a 0-byte file every interval.
        """
        audio = self._mix_buffers()
        if audio.shape[0] == 0:
            return False
        path = self._emergency_backup_path
        path.parent.mkdir(parents=True, exist_ok=True)
        _write_wav_atomic(path, audio)
        return True

    def stop_recording(self, output_filepath: Path | str) -> Path:
        """Stop recording, mix both streams, and write a 16 kHz mono WAV.

        Safe to call when capture already stopped on its own (device drop):
        the thread joins return immediately and everything captured before
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

        # First, retire the periodic flusher. Joining before any write is what
        # guarantees the background thread cannot overwrite the final WAV's
        # backup mid-write.
        self._stop_checkpointing()

        for thread in (self._system_thread, self._mic_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=STOP_JOIN_TIMEOUT_SECONDS)
                if thread.is_alive():  # pathological backend — daemon will die
                    print(
                        "warning: capture thread did not stop within "
                        f"{STOP_JOIN_TIMEOUT_SECONDS:.0f}s; the audio backend "
                        "may be wedged.",
                        file=sys.stderr,
                        flush=True,
                    )

        if self._finalized and self._last_path is not None:
            return self._last_path

        audio = self._mix_buffers()
        self._release_buffers()

        path = Path(output_filepath)
        path.parent.mkdir(parents=True, exist_ok=True)
        if audio.shape[0] > 0:
            _write_wav_atomic(path, audio)

        # The periodic backup is now strictly a subset of the final WAV (same
        # audio, fewer seconds), so leaving it behind would only invite the
        # operator to open the stale one. Removing it also preserves the
        # long-standing behaviour that a clean stop leaves no backup file.
        # Best-effort: a failure here must not fail an otherwise good capture.
        try:
            self._emergency_backup_path.unlink(missing_ok=True)
        except OSError:
            pass

        self._finalized = True
        self._last_path = path
        return path

    def _mix_buffers(self) -> np.ndarray:
        """Average the per-source chunk buffers into one mono array.

        Only sources that actually captured audio are mixed in, so a
        microphone that never opened does not drag the take to silence.

        Returns:
            The mixed 1-D float32 recording.
        """
        with self._buffer_lock:
            system = _stack_chunks(list(self._system_buffer))
            mic = _stack_chunks(list(self._mic_buffer))
        return _combine_streams(system, mic)

    def _release_buffers(self) -> None:
        """Drop the per-source chunk lists once they have been mixed."""
        with self._buffer_lock:
            self._system_buffer = []
            self._mic_buffer = []

    # ------------------------------------------------------------------
    # capture thread
    # ------------------------------------------------------------------

    def _capture_stream(
        self,
        device: LoopbackDevice,
        buffer: list[np.ndarray],
        label: str,
    ) -> None:
        """Daemon-thread body: capture one device until stopped or it fails.

        Args:
            device: The capture device (loopback or microphone).
            buffer: The per-source list this thread appends mono chunks to.
            label: Human-readable source name used in log/alert messages.
        """
        is_mic = label == "microphone"
        try:
            # ``blocksize`` is left unset so WASAPI chooses its default
            # engine period, which is always valid in shared mode.
            with device.recorder(samplerate=SAMPLE_RATE) as recorder:
                # Report in; release start_recording() once BOTH streams are
                # open so the caller never starts timing against a half-open
                # pair of devices.
                with self._open_lock:
                    self._streams_opened += 1
                    if self._streams_opened >= self._streams_expected:
                        self._open_complete.set()
                while self.is_recording.is_set():
                    chunk = recorder.record(numframes=CHUNK_FRAMES)
                    with self._buffer_lock:
                        buffer.append(_to_mono(chunk))
        except Exception as exc:
            if is_mic:
                # Best-effort source: degrade to system-audio-only rather
                # than throwing away the lecture.
                self._mic_failed = True
                self._open_complete.set()  # never hang start_recording()
                if self.is_recording.is_set():
                    print(
                        f"\n⚠️ Microphone capture failed ({exc!r}) — "
                        "continuing with system audio only. Diarization will "
                        "not be able to separate room voices.",
                        file=sys.stderr,
                        flush=True,
                    )
                return

            if not self._open_complete.is_set():
                # Failed before the stream ever opened: report it on the
                # caller's thread via start_recording()'s handshake.
                self._open_error = exc
                self._open_complete.set()
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
        """Best-effort dump of the mixed buffer to the emergency backup WAV.

        Returns:
            The backup path on success, or ``None`` if the dump failed.
        """
        try:
            audio = self._mix_buffers()  # buffers are kept: stop_recording
                                         # still flushes them afterwards
            self._emergency_backup_path.parent.mkdir(parents=True, exist_ok=True)
            _write_wav_atomic(self._emergency_backup_path, audio)
            return self._emergency_backup_path
        except Exception:
            return None
