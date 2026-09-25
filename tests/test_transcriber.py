"""Headless unit tests for ``transcriber.py`` (Phase 2) — ``LocalTranscriber``.

The whole suite runs **offline** — no model download, no CPU inference, no
audio hardware:

* ``faster_whisper.WhisperModel`` is replaced with a mock, so building
  ``LocalTranscriber()`` is instant and never touches disk or network;
* ``model.transcribe`` is mocked to yield a controlled list of segment
  objects, so the concat/priming/artifact logic is tested without decoding
  any audio;
* the ``raw_transcript.txt`` artifact is written next to the WAV in
  ``tmp_path``.
"""

from __future__ import annotations

from transcriber import LocalTranscriber, TRANSCRIPT_FILENAME


class MockSegment:
    """Duck-typed faster-whisper segment: only ``.text`` is needed."""

    def __init__(self, text: str) -> None:
        self.text = text


def test_transcribe_audio_concatenates_segments_and_priming(mocker, tmp_path) -> None:
    """Segments are joined into the transcript file; language and priming
    kwargs reach the model untouched."""
    wav_path = tmp_path / "lecture.wav"
    wav_path.write_bytes(b"\x00" * 44)  # placeholder — never decoded

    model_cls = mocker.patch("transcriber.WhisperModel")
    model_cls.return_value.transcribe.return_value = (
        iter([MockSegment("Hello"), MockSegment(" World")]),
        mocker.MagicMock(language="en", duration=10.0),
    )

    transcriber = LocalTranscriber()

    # -- model construction: local "base" model, CPU, int8 quantization -----
    model_cls.assert_called_once_with("base", device="cpu", compute_type="int8")

    transcript_path = transcriber.transcribe_audio(wav_path)

    # -- the WAV path and transcription config went to model.transcribe -----
    call = model_cls.return_value.transcribe.call_args
    assert call.args[0] == str(wav_path)
    assert call.kwargs["vad_filter"] is True
    assert call.kwargs["beam_size"] == 5
    assert call.kwargs["language"] == "en"
    priming = call.kwargs["initial_prompt"]
    assert "Kenya" in priming and "Swahili" in priming  # Swahili priming intact

    # -- return contract: the saved .txt artifact path -----------------------
    assert transcript_path == tmp_path / TRANSCRIPT_FILENAME
    assert transcript_path.exists()

    # -- artifact content: segments stripped + newline-joined -----------------
    # LocalTranscriber returns the raw-transcript *file* (the string is what
    # main.py hands to the synthesizer by reading the file).
    assert transcript_path.read_text(encoding="utf-8") == "Hello\nWorld"