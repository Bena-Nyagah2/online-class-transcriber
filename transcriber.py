"""Phase 2 — Lecture audio transcription via NVIDIA NIM (Whisper).

Turns a recorded lecture WAV into plain text:

1.  silence is stripped with :mod:`pydub.silence`, so we never pay to send
    dead air to the model;
2.  the remaining audio is sliced into 10-minute chunks so each MP3 payload
    stays comfortably below the 25 MB request limit;
3.  every chunk is encoded to MP3 **in memory** (``io.BytesIO`` — no temp
    files) and sent to ``openai/whisper-large-v3`` through the
    NVIDIA-hosted, OpenAI-compatible endpoint with exponential-backoff
    retries for transient failures;
4.  the concatenated transcript is written next to the WAV as
    ``raw_transcript.txt`` and also returned to the caller.

Production note: :mod:`pydub` shells out to ``ffmpeg`` for the MP3 export,
so the transcriber expects ``ffmpeg`` on ``PATH`` when run for real. The
test-suite mocks the export away, so the tests run without it.
"""

from __future__ import annotations

import gc
import io
import os
from pathlib import Path
from typing import List, Optional, Union

import openai
from openai import APIConnectionError, APITimeoutError, RateLimitError
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

MODEL_NAME = "openai/whisper-large-v3"
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"

SILENCE_MIN_LEN_MS = 3000    # gaps shorter than this are kept as content
SILENCE_THRESH_DB = -40       # dBFS below which a window counts as silence
SILENCE_KEEP_MS = 500         # silence retained around each spoken chunk
CHUNK_LENGTH_MS = 600_000     # 10 minutes of audio per request
TRANSCRIPT_FILENAME = "raw_transcript.txt"

# Transient errors we are willing to retry; 4xx auth/validation errors are not.
_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


class NIMTranscriber:
    """Transcribes lecture WAV files through NVIDIA's NIM Whisper API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Prepare the OpenAI-compatible client for the NVIDIA endpoint.

        Args:
            api_key: NVIDIA API key. Defaults to ``NVIDIA_API_KEY`` env var.
                An explicit argument makes the class constructible in tests
                without touching the environment.
        """
        self.api_key = api_key or os.environ.get(NVIDIA_API_KEY_ENV)
        if not self.api_key:
            raise ValueError(
                f"{NVIDIA_API_KEY_ENV} is not set — pass api_key=... or "
                "export the variable."
            )
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=NVIDIA_BASE_URL,
        )

    # -- pre-processing ----------------------------------------------------

    def _strip_silence(
        self, audio: AudioSegment, keep_silence_ms: int = SILENCE_KEEP_MS
    ) -> AudioSegment:
        """Cut long silent gaps out of an :class:`AudioSegment`.

        Args:
            audio: The raw capture.
            keep_silence_ms: How much silence to keep on each side of a
                spoken chunk (0 = remove the gap completely).

        Returns:
            The concatenated non-silent audio.
        """
        spoken = split_on_silence(
            audio,
            min_silence_len=SILENCE_MIN_LEN_MS,
            silence_thresh=SILENCE_THRESH_DB,
            keep_silence=keep_silence_ms,
        )
        if not spoken:
            return audio
        return sum(spoken[1:], spoken[0])

    def _split_into_chunks(self, audio: AudioSegment) -> List[AudioSegment]:
        """Slice audio into <=10-minute segments (600,000 ms each).

        Args:
            audio: Silence-stripped audio.

        Returns:
            A list of non-overlapping chunks covering the whole segment.
        """
        return [
            audio[start : start + CHUNK_LENGTH_MS]
            for start in range(0, len(audio), CHUNK_LENGTH_MS)
        ]

    # -- API call ----------------------------------------------------------

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(5),
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        reraise=True,
    )
    def _transcribe_chunk(self, audio_buffer: io.BytesIO) -> str:
        """Send one in-memory MP3 buffer to Whisper; retry transient errors.

        Args:
            audio_buffer: A ``BytesIO`` holding MP3 audio, named e.g.
                ``chunk.mp3`` (the SDK keys uploaded files off ``.name``).

        Returns:
            The chunk's transcript text.
        """
        result = self.client.audio.transcriptions.create(
            model=MODEL_NAME,
            file=audio_buffer,
        )
        return result.text

    # -- public entry point ------------------------------------------------

    def transcribe_audio(self, audio_filepath: Union[str, Path]) -> str:
        """Transcribe a WAV file and persist ``raw_transcript.txt`` beside it.

        Args:
            audio_filepath: Path to the lecture WAV.

        Returns:
            The full transcript text.
        """
        audio_path = Path(audio_filepath)
        audio = AudioSegment.from_wav(str(audio_path))

        # 1. Silence stripping — log how much dead air was removed.
        original_ms = len(audio)
        audio = self._strip_silence(audio)
        stripped_ms = original_ms - len(audio)
        print(f"Stripped {stripped_ms / 1000:.1f} seconds of silence")

        # 2. Chunking — cap each payload at 10 minutes of audio.
        chunks = self._split_into_chunks(audio)
        print(f"Transcribing {len(chunks)} chunk(s)...")

        # 3/4. Per-chunk in-memory MP3 export + API call + explicit cleanup.
        parts: List[str] = []
        for index, chunk in enumerate(chunks, start=1):
            buffer = io.BytesIO()
            chunk.export(buffer, format="mp3")
            buffer.name = "chunk.mp3"

            print(
                f"  chunk {index}/{len(chunks)} "
                f"({len(chunk) / 1000:.1f}s) -> API"
            )
            text = self._transcribe_chunk(buffer)
            parts.append(text)

            del chunk, buffer
            gc.collect()

        transcript = "\n".join(part.strip() for part in parts if part.strip())

        # 5. Persist next to the source audio and return the text.
        transcript_path = audio_path.parent / TRANSCRIPT_FILENAME
        transcript_path.write_text(transcript, encoding="utf-8")
        print(f"Transcript saved to {transcript_path}")

        return transcript