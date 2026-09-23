"""Headless unit tests for ``transcriber.py`` (Phase 2).

The whole suite runs **offline** — no NVIDIA API calls, no network I/O, no
audio hardware, no ffmpeg:

* the ``openai`` client is always mocked out via ``mocker``,
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
from openai import RateLimitError
from pydub import AudioSegment
from pydub.generators import Sine

from transcriber import NIMTranscriber, CHUNK_LENGTH_MS, MODEL_NAME

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_transcriber(mocker: Any) -> Tuple[NIMTranscriber, Any]:
    """A transcriber whose OpenAI client is replaced by a mock."""
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


class _FakeResponse:
    """Duck-typed stand-in for ``httpx.Response`` used by openai errors."""

    status_code = 429
    headers: dict = {}
    request = None
    text = "rate limited"
    reason_phrase = "Too Many Requests"
    http_version = "HTTP/1.1"
    url = "https://integrate.api.nvidia.com/v1/audio/transcriptions"


def _fake_rate_limit_error() -> RateLimitError:
    """A real :class:`RateLimitError` constructed without any HTTP I/O."""
    return RateLimitError("NVIDIA rate limit exceeded", response=_FakeResponse(), body=None)


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


def test_chunking_splits_into_ten_minute_segments() -> None:
    """Audio longer than 10 minutes is sliced into <=10-minute chunks."""
    # 25 minutes of audio at frame_rate=1000 -> only 3 MB of data (cheap).
    audio = AudioSegment.silent(duration=25 * 60 * 1000, frame_rate=1000)
    transcriber = NIMTranscriber(api_key="test-key")

    chunks = transcriber._split_into_chunks(audio)

    assert len(chunks) == 3
    assert [len(chunk) for chunk in chunks] == [
        CHUNK_LENGTH_MS,
        CHUNK_LENGTH_MS,
        5 * 60 * 1000,
    ]


# ---------------------------------------------------------------------------
# retry logic
# ---------------------------------------------------------------------------


def test_api_retry_logic(mocker) -> None:
    """RateLimitError 429s are retried until a successful response arrives."""
    transcriber, client = _make_transcriber(mocker)

    ok_transcript = mocker.MagicMock()
    ok_transcript.text = "mock lecture transcript"
    client.audio.transcriptions.create.side_effect = [
        _fake_rate_limit_error(),
        _fake_rate_limit_error(),
        ok_transcript,
    ]

    # Neutralise the exponential backoff so this test runs in milliseconds.
    mocker.patch.object(
        NIMTranscriber._transcribe_chunk.retry, "sleep", lambda seconds: None
    )

    audio_buffer = io.BytesIO(b"fake-mp3-audio-bytes")
    audio_buffer.name = "chunk.mp3"

    text = transcriber._transcribe_chunk(audio_buffer)

    assert text == "mock lecture transcript"
    assert client.audio.transcriptions.create.call_count == 3
    call_kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == MODEL_NAME
    assert call_kwargs["file"] is audio_buffer


# ---------------------------------------------------------------------------
# end-to-end pipeline (every external call mocked)
# ---------------------------------------------------------------------------


def test_transcribe_audio_integration(mocker, tmp_path) -> None:
    """A tiny WAV transcribes to raw_transcript.txt next to it, fully offline."""
    wav_path = _write_tone_wav(tmp_path / "mini_lecture.wav", duration_ms=1000)
    transcriber, client = _make_transcriber(mocker)

    expected_text = "This lecture covers phase 2 transcription."
    client.audio.transcriptions.create.return_value = mocker.MagicMock(
        text=expected_text
    )

    # The only step needing ffmpeg is the MP3 export — mock it away so the
    # test needs zero external programs.
    mocker.patch.object(AudioSegment, "export", return_value=None)

    text = transcriber.transcribe_audio(wav_path)

    assert text == expected_text
    transcript_path = tmp_path / "raw_transcript.txt"
    assert transcript_path.exists()
    assert transcript_path.read_text(encoding="utf-8") == expected_text

    assert client.audio.transcriptions.create.call_count == 1
    call_kwargs = client.audio.transcriptions.create.call_args.kwargs
    assert call_kwargs["model"] == MODEL_NAME
    # The in-memory buffer was named so the SDK can treat it as an upload.
    assert call_kwargs["file"].name == "chunk.mp3"