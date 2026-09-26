"""Headless unit tests for ``recorder.py`` and the filesystem/CLI utilities
in ``main.py`` (Phase 1).

No test here requires physical audio hardware, active sound output, or user
input:

* every audio backend call (``soundcard``) is mocked via ``mocker``;
* every output file is written under pytest's isolated ``tmp_path``;
* course-code prompting is exercised by mocking ``builtins.input``.
"""

from __future__ import annotations

import threading
from datetime import date

import numpy as np
import pytest
import soundfile as sf

from main import _sanitize_course_code, create_output_dir, prompt_course_code
from recorder import (
    AudioRecorder,
    CHUNK_FRAMES,
    SAMPLE_RATE,
    _combine_streams,
    _stack_chunks,
    _to_mono,
)


# ---------------------------------------------------------------------------
# recorder.py — downmix / buffer logic
# ---------------------------------------------------------------------------


def test_downmix_to_mono() -> None:
    """A stereo chunk collapses to a 1-D mono float32 array by averaging."""
    stereo = np.array(
        [[0.6, 0.2], [1.0, 0.0], [0.0, -1.0], [-0.5, 0.5]],
        dtype=np.float32,
    )

    mono = _to_mono(stereo)

    assert mono.ndim == 1
    assert mono.dtype == np.float32
    assert mono.shape == (4,)
    np.testing.assert_allclose(mono, stereo.mean(axis=1), rtol=1e-6)
    assert mono[0] == pytest.approx(0.4)  # mean(0.6, 0.2)

    # A mono (1-D) input passes through unchanged.
    already_mono = np.ones(64, dtype=np.float32)
    out = _to_mono(already_mono)
    assert out.ndim == 1 and out.dtype == np.float32
    np.testing.assert_array_equal(out, already_mono)


def test_buffer_stacks_to_single_mono_record() -> None:
    """Chunks produced by _to_mono concatenate into one mono audio record."""
    stereo = np.array([[0.6, 0.2], [1.0, 0.0]], dtype=np.float32)
    buffer = [_to_mono(stereo), _to_mono(np.ones((3, 2), dtype=np.float32))]

    joined = _stack_chunks(buffer)

    assert joined.ndim == 1
    assert joined.shape == (5,)
    assert joined.dtype == np.float32


# ---------------------------------------------------------------------------
# recorder.py — two-way stream mixing
# ---------------------------------------------------------------------------


def test_combine_streams_averages_without_clipping() -> None:
    """Loopback and mic are averaged, and loud input stays inside [-1, 1]."""
    loopback = np.array([1.0, -1.0, 0.8], dtype=np.float32)
    mic = np.array([0.5, 0.5, 0.2], dtype=np.float32)

    mixed = _combine_streams(loopback, mic)

    assert mixed.ndim == 1
    assert mixed.dtype == np.float32
    # Averaged, not summed: summing would clip the first sample to 1.5.
    np.testing.assert_allclose(mixed, [0.75, -0.25, 0.5], rtol=1e-6)
    assert mixed.max() <= 1.0 and mixed.min() >= -1.0


def test_combine_streams_pads_shorter_and_ignores_empty() -> None:
    """Length drift is zero-padded; an empty source cannot mute the mix."""
    loopback = np.array([0.4, 0.4, 0.4, 0.4], dtype=np.float32)
    mic = np.array([0.2, 0.2], dtype=np.float32)  # two frames behind

    mixed = _combine_streams(loopback, mic)

    # Duration follows the LONGER stream; the short one is zero-padded.
    assert mixed.shape == (4,)
    np.testing.assert_allclose(mixed, [0.3, 0.3, 0.2, 0.2], rtol=1e-6)

    # A microphone that never opened contributes nothing (system-only take).
    solo = _combine_streams(loopback, np.empty(0, dtype=np.float32))
    np.testing.assert_allclose(solo, loopback, rtol=1e-6)

    # Both sources empty -> empty result, not a crash.
    assert _combine_streams(
        np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    ).shape == (0,)


def test_two_way_capture_opens_both_devices_at_same_rate(mocker, tmp_path) -> None:
    """Both sources are polled in parallel at 16 kHz and yield one mono WAV.

    ``record()`` blocks on a gate that the test opens only after
    ``start_recording`` has returned, so each thread is guaranteed to have
    captured at least one chunk and the leading chunk is a true two-source
    overlap. Beyond that chunk the per-thread chunk counts are
    scheduler-dependent, so only bounds are asserted — the exact mixing maths
    is covered deterministically by the ``_combine_streams`` tests above.
    """
    recorder = AudioRecorder(emergency_backup_path=tmp_path / "backup.wav")
    gate = threading.Event()

    def _fake_device(name: str) -> mocker.MagicMock:
        """A device whose recorder yields constant mono chunks after ``gate``."""
        chunk = np.full((CHUNK_FRAMES, 2), 0.5, dtype=np.float32)

        def _record(numframes=None):  # noqa: ARG001 - soundcard's signature
            gate.wait(timeout=5.0)
            return chunk

        stream = mocker.MagicMock()
        stream.record.side_effect = _record
        context = mocker.MagicMock()
        context.__enter__.return_value = stream
        device = mocker.MagicMock()
        device.name = name
        device.recorder.return_value = context
        return device

    system_device = _fake_device("Speakers (Loopback)")
    mic_device = _fake_device("Built-in Microphone")

    mocker.patch.object(
        recorder, "get_loopback_devices", return_value=[system_device]
    )

    recorder.start_recording(0, microphone=mic_device)

    # Both devices opened, at the identical sample rate, before we let them run.
    system_device.recorder.assert_called_once_with(samplerate=SAMPLE_RATE)
    mic_device.recorder.assert_called_once_with(samplerate=SAMPLE_RATE)
    assert recorder._system_thread is not None
    assert recorder._mic_thread is not None

    gate.set()  # release both capture loops

    wav_path = tmp_path / "two_way.wav"
    recorder.stop_recording(wav_path)

    info = sf.info(str(wav_path))
    assert info.samplerate == SAMPLE_RATE
    assert info.channels == 1            # downmixed to a single mono track
    assert info.frames >= CHUNK_FRAMES   # both threads contributed audio

    data, _ = sf.read(str(wav_path))
    # First chunk: both sources present, so mean(0.5, 0.5) == 0.5.
    np.testing.assert_allclose(data[:CHUNK_FRAMES], 0.5, atol=1e-3)
    # The tail is a partial overlap (or zero-padding), so it is bounded by the
    # louder source and never clips.
    assert data.min() >= 0.0 and data.max() <= 0.5 + 1e-3


def test_start_recording_none_microphone_skips_mic_thread(mocker, tmp_path) -> None:
    """``microphone=None`` records system audio only — no input device opened."""
    recorder = AudioRecorder(emergency_backup_path=tmp_path / "backup.wav")

    stream = mocker.MagicMock()
    stream.record.return_value = np.zeros((CHUNK_FRAMES, 2), dtype=np.float32)
    context = mocker.MagicMock()
    context.__enter__.return_value = stream
    system_device = mocker.MagicMock()
    system_device.name = "Speakers"
    system_device.recorder.return_value = context

    mocker.patch.object(
        recorder, "get_loopback_devices", return_value=[system_device]
    )
    # get_default_microphone must never be consulted when None is explicit.
    default_mic = mocker.patch.object(recorder, "get_default_microphone")

    recorder.start_recording(0, microphone=None)

    assert default_mic.call_count == 0
    assert recorder._mic_thread is None
    assert recorder._mic_active is False

    recorder.stop_recording(tmp_path / "solo.wav")


# ---------------------------------------------------------------------------
# recorder.py — device enumeration (fully mocked)
# ---------------------------------------------------------------------------


def test_device_enumeration_mocked(mocker) -> None:
    """get_loopback_devices filters to loopback devices and caches them."""
    # Force the Windows code path regardless of the host OS.
    mocker.patch("recorder.sys.platform", "win32")

    loopback_speaker = mocker.Mock(name="Speakers", isloopback=True, channels=2)
    loopback_headphones = mocker.Mock(name="Headphones", isloopback=True, channels=2)
    regular_mic = mocker.Mock(name="Built-in Mic", isloopback=False, channels=1)
    usb_mic = mocker.Mock(name="USB Mic", isloopback=False, channels=2)
    all_devices = [regular_mic, loopback_speaker, usb_mic, loopback_headphones]

    mock_all_mics = mocker.patch(
        "recorder.sc.all_microphones", return_value=list(all_devices)
    )

    recorder = AudioRecorder()
    devices = recorder.get_loopback_devices()

    # Filtering: only isloopback=True devices survive, order preserved.
    assert devices == [loopback_speaker, loopback_headphones]
    mock_all_mics.assert_called_once_with(include_loopback=True)

    # Caching: the second call reuses the cache without re-enumerating.
    again = recorder.get_loopback_devices()
    assert again == devices
    assert again is not devices  # defensive copy — callers can't corrupt cache
    mock_all_mics.assert_called_once()


# ---------------------------------------------------------------------------
# recorder.py — device-drop defence path
# ---------------------------------------------------------------------------


def test_device_drop_recovery(mocker, tmp_path) -> None:
    """A mid-lecture device failure dumps the buffer and stops cleanly."""
    emergency_wav = tmp_path / "emergency_backup.wav"
    recorder = AudioRecorder(emergency_backup_path=emergency_wav)

    # One valid stereo chunk, then the device "disappears" mid-lecture:
    valid_chunk = np.tile(
        np.array([0.4, 0.2], dtype=np.float32), (CHUNK_FRAMES, 1)
    )
    capture_mock = mocker.MagicMock()
    capture_mock.record.side_effect = [valid_chunk, RuntimeError("device unplugged")]

    context_mock = mocker.MagicMock()
    context_mock.__enter__.return_value = capture_mock

    device_mock = mocker.MagicMock()
    device_mock.recorder.return_value = context_mock

    mocker.patch.object(recorder, "get_loopback_devices", return_value=[device_mock])

    # microphone=None records system audio only, so no real input device is
    # opened during the test.
    recorder.start_recording(0, microphone=None)  # handshake: stream "opens"

    # Let the daemon thread run to completion (it fails on the 2nd record()).
    assert recorder._system_thread is not None
    recorder._system_thread.join(timeout=5.0)

    # Thread exited cleanly; the recording event was cleared; no open error.
    assert not recorder._system_thread.is_alive()
    assert not recorder.is_recording.is_set()
    assert recorder._open_complete.is_set()
    assert recorder._open_error is None

    # Emergency backup holds the single captured mono chunk: 16 kHz mono.
    assert emergency_wav.exists()
    data, sample_rate = sf.read(str(emergency_wav))
    assert sample_rate == SAMPLE_RATE
    assert data.shape == (CHUNK_FRAMES,)
    np.testing.assert_allclose(data, 0.3, atol=1e-3)  # mean(0.4, 0.2)


# ---------------------------------------------------------------------------
# recorder.py — stop / flush behaviour
# ---------------------------------------------------------------------------


def test_stop_recording_idempotent(tmp_path) -> None:
    """stop_recording writes the buffer once and is safe to call twice."""
    recorder = AudioRecorder()

    # Simulate 0.5 s of captured mono audio waiting in the buffer:
    stereo = np.tile(
        np.array([0.5, 0.5], dtype=np.float32), (SAMPLE_RATE // 2, 1)
    )
    with recorder._buffer_lock:
        recorder._system_buffer.append(_to_mono(stereo))

    wav_path = tmp_path / "test.wav"

    first_result = recorder.stop_recording(wav_path)

    assert first_result == wav_path
    assert wav_path.exists()
    assert not recorder.is_recording.is_set()

    info = sf.info(str(wav_path))
    assert info.samplerate == SAMPLE_RATE
    assert info.channels == 1
    assert info.frames == SAMPLE_RATE // 2

    original_bytes = wav_path.read_bytes()

    # The second call must not raise, and must not clobber the file with an
    # (empty) buffer — the first take is preserved byte for byte.
    second_result = recorder.stop_recording(wav_path)
    assert second_result == first_result
    assert wav_path.read_bytes() == original_bytes

    # The buffer was flushed on the first call.
    with recorder._buffer_lock:
        assert recorder._system_buffer == []
        assert recorder._mic_buffer == []


# ---------------------------------------------------------------------------
# main.py — filesystem / sanitisation utilities
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("CRM_1106", "CRM_1106"),   # already safe
        ("CRM/1106", "CRM_1106"),   # forward slash stripped
        ("CRM\\1106", "CRM_1106"),  # backslash stripped
        ("CRM 1106", "CRM_1106"),   # whitespace stripped
        ("A:B|C?*", "A_B_C_"),      # other path-unsafe chars, runs collapse
    ],
)
def test_sanitize_course_code(raw: str, expected: str) -> None:
    assert _sanitize_course_code(raw) == expected


def test_prompt_course_code_mocked_input(mocker) -> None:
    """Prompting sanitises input and re-prompts on empty entries."""
    mocker.patch("builtins.input", side_effect=["", "  ", "CRM/1106"])
    assert prompt_course_code() == "CRM_1106"


def test_create_output_dir(tmp_path, monkeypatch) -> None:
    """Directories are created as ./output/{course_code}_{YYYY-MM-DD}/."""
    monkeypatch.chdir(tmp_path)
    today = date.today().strftime("%Y-%m-%d")

    result = create_output_dir("CRM_1106")

    assert result.name == f"CRM_1106_{today}"
    assert result.parent.name == "output"
    assert result.is_dir()


def test_prompt_then_create_output_dir_uses_sanitized_code(
    tmp_path, monkeypatch, mocker
) -> None:
    """CRM/1106 flows through prompting into output/CRM_1106_{date}/."""
    monkeypatch.chdir(tmp_path)
    mocker.patch("builtins.input", return_value="CRM/1106")

    code = prompt_course_code()
    result = create_output_dir(code)

    assert result.name == f"CRM_1106_{date.today():%Y-%m-%d}"
    assert result.parent.name == "output"
    assert result.is_dir()