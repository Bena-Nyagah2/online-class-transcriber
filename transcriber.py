"""Phase 2 — Lecture audio transcription via faster-whisper (local).

Turns a recorded lecture WAV into plain text entirely on-device:

1.  faster-whisper runs ``openai/whisper``'s ``base`` model locally
    (CPU, int8-quantized) — no HTTP calls, no API keys, no request limits;
2.  ``vad_filter=True`` skips silence, so long pauses never inflate the
    transcript, and ``beam_size=5`` keeps the output accurate;
3.  the full transcript is written next to the WAV as
    ``raw_transcript.txt`` and the file's path is returned to the caller.

Model note: ``base`` is small and fast — ideal for testing. Swap it for
``"large-v3"`` in :meth:`LocalTranscriber.__init__` when production-grade
accuracy is needed.
"""

from __future__ import annotations

from pathlib import Path
from typing import Union

from faster_whisper import WhisperModel

TRANSCRIPT_FILENAME = "raw_transcript.txt"


class LocalTranscriber:
    """Transcribes lecture WAV files locally with faster-whisper."""

    def __init__(self) -> None:
        # "base" for speed during testing; the user can upgrade to
        # "large-v3" later without touching the rest of the pipeline.
        self.model = WhisperModel("base", device="cpu", compute_type="int8")

    # -- public entry point ------------------------------------------------

    def transcribe_audio(self, wav_path: Union[str, Path]) -> Path:
        """Transcribe a WAV file and persist ``raw_transcript.txt`` beside it.

        faster-whisper handles VAD and arbitrarily long audio natively, so
        no silence stripping or chunking is needed.

        Args:
            wav_path: Path to the lecture WAV.

        Returns:
            Path to the saved ``raw_transcript.txt``.
        """
        audio_path = Path(wav_path)
        # language="en" forces English output — Swahili segments are
        # auto-translated back to English. initial_prompt primes the model
        # with Kenyan regional context for better code-switch handling.
        segments, info = self.model.transcribe(
            str(audio_path),
            vad_filter=True,
            beam_size=5,
            language="en",
            initial_prompt=(
                "This is a university law and criminology lecture in Kenya, "
                "heavily mixing English and Swahili."
            ),
        )
        print(
            f"  -> transcribing {audio_path.name} locally "
            f"(lang={info.language}, {info.duration:.1f}s)"
        )

        # faster-whisper yields segments lazily as decoding progresses.
        parts = [segment.text.strip() for segment in segments]
        transcript = "\n".join(part for part in parts if part)

        transcript_path = audio_path.parent / TRANSCRIPT_FILENAME
        transcript_path.write_text(transcript, encoding="utf-8")
        print(f"Transcript saved to {transcript_path}")

        return transcript_path