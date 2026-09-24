"""Headless unit tests for ``transcriber.py`` (Phase 2).

The whole suite runs **offline** — no NVIDIA API calls, no network I/O, no
audio hardware, no ffmpeg:

* the HTTP client (``httpx2``) is always mocked out via ``mocker``,
* the MP3 export (the only step that would shell out to ffmpeg) is mocked,
* synthetic audio is generated in pure Python (:mod:`pydub.generators` +
  the stdlib :mod:`wave` module),
* every file written by the code under test lands in ``tmp_path``.
"""

from __future__ import annotations

import io
import math
import struct
import wave
from pathlib import Path
from typing import Any, Tuple

import pytest
import httpx2
from pydub import AudioSegment
from pydub.generators import Sine

from transcriber import (
    NIMTranscriber,
    NVCF_TRANSCRIPTION_URL,
    CHUNK_LENGTH_MS,
    MODEL_NAME,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_transcriber(mocker: Any) -> Tuple[NIMTranscriber, Any]:
    """A transcriber whose HTTP client is replaced by a mock."""
    transcriber = NIMTranscriber(api_key="test-key")
    client = mocker.MagicMock()
    transcriber.client = client
    return transcriber, client


def _tone_segment(duration_ms: int, sample_rate: int = 8000) -> AudioSegment:
    """Synthesise a sine-wave AudioSegment in pure Python (no ffmpeg)."""
    return Sine(440, sample_rate=sample_rate).to_audio_segment(duration=duration_ms)


def _tone_silence_tone(sample_rate: int = 8000) -> AudioSegment:
    """2 s tone + 4 s silence + 2 s tone (8 s total, one 4 s silent gap)."""
    silence = AudioSegment.silent(duration=4000, frame_rate=sample_rate)
    return _tone_segment(2000, sample_rate) + silence + _tone_segment(2000, sample_rate)


def _write_tone_wav(path: Path, duration_ms: int = 1000, sample_rate: int = 8000) -> Path:
    """Write a tiny real mono WAV using only the stdlib (works headless)."""
    n_frames = int(sample_rate * duration_ms / 1000)
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n_frames):
            sample = int(
                0.5 * 32767 * math.sin(2 * math.pi * 440.0 * i / sample_rate)
            )
            frames += struct.pack("<h", sample)
        wav_file.writeframes(bytes(frames))
    return path


# ---------------------------------------------------------------------------
# silence stripping
# ---------------------------------------------------------------------------


def test_silence_stripping() -> None:
    """A 4 s silent gap is removed from an 8 s tone/silence/tone segment."""
    audio = _tone_silence_tone()
    assert len(audio) == 8000

    transcriber = NIMTranscriber(api_key="test-key")

    # keep_silence_ms=0 -> the gap is removed completely; 2s+2s of tone remain.
    stripped = transcriber._strip_silence(audio, keep_silence_ms=0)
    assert len(stripped) == pytest.approx(4000, abs=20)
    assert stripped.rms > 1000  # audio survives; dead air is gone

    # Default behaviour (keep_silence=500) keeps 500 ms of silence per edge.
    with_keep = transcriber._strip_silence(audio)
    assert len(with_keep) == pytest.approx(5000, abs=20)


# ---------------------------------------------------------------------------
# chunking
# ---------------------------------------------------------------------------


def test_chunking_splits_into_capped_segments() -> None:
    """Audio longer than the chunk cap is sliced into <=cap segments.

    Expectations are derived from ``CHUNK_LENGTH_MS`` so the test tracks
    whatever micro-chunking threshold the pipeline is configured with.
    """
    # 25 minutes of audio at frame_rate=1000 -> only 3 MB of data (cheap).
    total_ms = 25 * 60 * 1000
    audio = AudioSegment.silent(duration=total_ms, frame_rate=1000)
    transcriber = NIMTranscriber(api_key="test-key")

    chunks = transcriber._split_into_chunks(audio)

    expected = [CHUNK_LENGTH_MS] * (total_ms // CHUNK_LENGTH_MS)
    remainder = total_ms % CHUNK_LENGTH_MS
    if remainder:
        expected.append(remainder)

    assert [len(chunk) for chunk in chunks] == expected
    assert all(len(chunk) <= CHUNK_LENGTH_MS for chunk in chunks)


# ---------------------------------------------------------------------------
# retry logic
# ---------------------------------------------------------------------------


def test_api_retry_logic(mocker) -> None:
    """httpx transport failures are retried until a successful response."""
    transcriber, client = _make_transcriber(mocker)

    ok_response = mocker.MagicMock()
    ok_response.json.return_value = {"text": "mock lecture transcript"}

    # Two transient transport failures (connection reset, read timeout),
    # then a success. These are real httpx2 exceptions, so tenacity's
    # retry_if_exception_type actually matches them.
    client.post.side_effect = [
        httpx2.ConnectError("mock connection reset"),
        httpx2.ReadTimeout("mock read timeout"),
        ok_response,
    ]

    # Neutralise the exponential backoff so this test runs in milliseconds.
    mocker.patch.object(
        NIMTranscriber._transcribe_chunk.retry, "sleep", lambda seconds: None
    )

    audio_buffer = io.BytesIO()
    audio_buffer.write(b"fake-mp3-audio-bytes")
    audio_buffer.name = "chunk.mp3"

    # Reproduce the real-world bug: pydub's export leaves the cursor at EOF,
    # so without rewinding the backend would receive a 0-byte payload.
    assert audio_buffer.tell() > 0

    text = transcriber._transcribe_chunk(audio_buffer)

    assert text == "mock lecture transcript"
    assert audio_buffer.tell() == 0  # rewound before building the POST
    assert client.post.call_count == 3
    assert client.post.call_args.args[0] == NVCF_TRANSCRIPTION_URL
    call_kwargs = client.post.call_args.kwargs
    assert call_kwargs["data"] == {"model": MODEL_NAME, "response_format": "json"}
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
    # The file is sent as an explicit (filename, buffer, mime) tuple so the
    # gateway always sees audio/mpeg.
    filename, buffer, mime = call_kwargs["files"]["file"]
    assert filename == "chunk.mp3"
    assert buffer is audio_buffer
    assert mime == "audio/mpeg"


# ---------------------------------------------------------------------------
# end-to-end pipeline (every external call mocked)
# ---------------------------------------------------------------------------


def test_transcribe_audio_integration(mocker, tmp_path) -> None:
    """A tiny WAV transcribes to raw_transcript.txt next to it, fully offline."""
    wav_path = _write_tone_wav(tmp_path / "mini_lecture.wav", duration_ms=1000)
    transcriber, client = _make_transcriber(mocker)

    expected_text = "This lecture covers phase 2 transcription."
    ok_response = mocker.MagicMock()
    ok_response.json.return_value = {"text": expected_text}
    client.post.return_value = ok_response

    # The only step needing ffmpeg is the MP3 export — mock it away so the
    # test needs zero external programs.
    mocker.patch.object(AudioSegment, "export", return_value=None)

    text = transcriber.transcribe_audio(wav_path)

    assert text == expected_text
    transcript_path = tmp_path / "raw_transcript.txt"
    assert transcript_path.exists()
    assert transcript_path.read_text(encoding="utf-8") == expected_text

    assert client.post.call_count == 1
    assert client.post.call_args.args[0] == NVCF_TRANSCRIPTION_URL
    call_kwargs = client.post.call_args.kwargs
    assert call_kwargs["data"] == {"model": MODEL_NAME, "response_format": "json"}
    assert call_kwargs["headers"]["Authorization"] == "Bearer test-key"
    # The upload is an explicit (filename, buffer, audio/mpeg) tuple.
    filename, buffer, mime = call_kwargs["files"]["file"]
    assert filename == "chunk.mp3"
    assert mime == "audio/mpeg"
    # The in-memory buffer is still named so the SDK can treat it as a file.
    assert buffer.name == "chunk.mp3"