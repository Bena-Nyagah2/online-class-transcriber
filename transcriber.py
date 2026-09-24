"""Phase 2 — Lecture audio transcription via NVIDIA NIM (Whisper).

Turns a recorded lecture WAV into plain text:

1.  silence is stripped with :mod:`pydub.silence`, so we never pay to send
    dead air to the model;
2.  the remaining audio is sliced into 2-minute chunks so each MP3 payload
    stays comfortably below the 25 MB request limit;
3.  every chunk is encoded to MP3 **in memory** (``io.BytesIO`` — no temp
    files) and POSTed as a raw multipart form to the Whisper container's
    direct NVCF invocation URL with ``openai/whisper-large-v3`` and
    exponential-backoff retries;
4.  the concatenated transcript is written next to the WAV as
    ``raw_transcript.txt`` and also returned to the caller.

Transport note: the generic NVIDIA gateways 404 on audio traffic (both the
OpenAI SDK's path construction and the load-balanced ``ai.api.nvidia.com``
gateway), so the request goes straight to the Whisper container's direct
NVCF invocation URL with a plain HTTP client. The available one in this
environment is ``httpx2`` — the HTTP client ``openai>=3`` already depends
on — which speaks the exact ``httpx`` multipart API.

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

import httpx2
from dotenv import load_dotenv
from pydub import AudioSegment
from pydub.silence import split_on_silence
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load NVIDIA_API_KEY from a local .env (if present) before any client reads
# the environment. Explicit shell exports still take precedence.
load_dotenv()

MODEL_NAME = "openai/whisper-large-v3"
# Direct NVCF invocation URL for the whisper-large-v3 backend container.
# The public ai.api.nvidia.com gateway load-balances audio traffic and 404s
# on this path, so we bypass it and POST straight to the container.
NVCF_TRANSCRIPTION_URL = (
    "https://b702f636-f60c-4a3d-a6f4-f3568c13bd7d.invocation.api.nvcf.nvidia.com"
    "/v1/audio/transcriptions"
)
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"
SILENCE_MIN_LEN_MS = 3000    # gaps shorter than this are kept as content
SILENCE_THRESH_DB = -40       # dBFS below which a window counts as silence
SILENCE_KEEP_MS = 500         # silence retained around each spoken chunk
CHUNK_LENGTH_MS = 120_000     # 2 minutes of audio per request (micro-chunking)
TRANSCRIPT_FILENAME = "raw_transcript.txt"

# Errors we are willing to retry (direct httpx transport, no OpenAI wrapper):
# * TransportError covers connection resets/timeouts (the Errno 10053 aborts);
# * HTTPStatusError covers non-2xx replies surfaced by raise_for_status().
_RETRYABLE_ERRORS = (httpx2.TransportError, httpx2.HTTPStatusError)


class NIMTranscriber:
    """Transcribes lecture WAV files through NVIDIA's NIM Whisper API."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Prepare the HTTP client for the direct NVCF invocation endpoint.

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
        # 120s timeout so slow container replies are not cut short mid-upload
        # (server timeouts manifest as Errno 10053 socket aborts otherwise).
        self.client = httpx2.Client(timeout=120.0)

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
        """Slice audio into <=2-minute segments (120,000 ms each).

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
        """POST one in-memory MP3 buffer to the Whisper backend container.

        Uses a plain multipart POST to the container's direct NVCF invocation
        URL, bypassing the public gateway load balancer (which 404s on audio
        traffic). Tenacity retries transport errors (resets, timeouts) and
        non-2xx replies with exponential backoff.

        Args:
            audio_buffer: A ``BytesIO`` holding MP3 audio, named e.g.
                ``chunk.mp3`` (the multipart filename below).

        Returns:
            The chunk's transcript text.
        """
        # Debug: confirm the exact endpoint + model the request goes to.
        print(f"  -> POST {NVCF_TRANSCRIPTION_URL}  model={MODEL_NAME}")
        headers = {
            "Authorization": (
                f"Bearer {self.api_key or os.environ.get(NVIDIA_API_KEY_ENV)}"
            )
        }
        response = self.client.post(
            NVCF_TRANSCRIPTION_URL,
            headers=headers,
            files={"file": ("chunk.mp3", audio_buffer, "audio/mpeg")},
            data={"model": MODEL_NAME, "response_format": "json"},
        )
        response.raise_for_status()
        return response.json().get("text", "")

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

        # 2. Chunking — cap each payload at 2 minutes of audio.
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