"""Headless unit tests for ``transcriber.py`` (Phase 2) — ``LocalTranscriber``.

The whole suite runs **offline** — no model download, no CPU inference, no
audio hardware, no Hugging Face token:

* the whole ``whisperx`` module is replaced with a mock, so
  ``LocalTranscriber()`` never touches disk or network;
* the three pipeline stages (transcribe → align → diarize) are stubbed to
  return controlled segment dicts, so grouping, timestamping, and speaker
  labelling are tested without decoding any audio;
* the ``raw_transcript.txt`` artifact is written next to the WAV in
  ``tmp_path``.
"""

from __future__ import annotations

import pytest

from transcriber import (
    HF_TOKEN_ENV,
    LocalTranscriber,
    MAX_TURN_GAP_SECONDS,
    TRANSCRIPT_FILENAME,
    UNKNOWN_SPEAKER,
    _format_timestamp,
    _format_turns,
    resolve_hf_token,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_whisperx(mocker, segments, language="en", speakers=None):
    """Stub the three WhisperX stages and return the ``whisperx`` mock.

    Args:
        segments: Segment dicts returned by the align stage.
        language: Language code reported by the transcribe stage.
        speakers: Value returned by the diarization pipeline call.
    """
    whisperx = mocker.patch("transcriber.whisperx")
    whisperx.load_audio.return_value = [0.0] * 16000  # 1 s at 16 kHz

    model = whisperx.load_model.return_value
    model.transcribe.return_value = {"segments": list(segments), "language": language}

    whisperx.load_align_model.return_value = (
        mocker.MagicMock(name="align_model"),
        {"language": language},
    )
    whisperx.align.return_value = {"segments": list(segments)}
    whisperx.assign_word_speakers.return_value = {"segments": list(segments)}
    whisperx.DiarizationPipeline.return_value.return_value = (
        speakers if speakers is not None else mocker.MagicMock(name="diarize_df")
    )
    return whisperx


@pytest.fixture(autouse=True)
def _hf_token(monkeypatch):
    """Every test gets a token so construction never hits the real env."""
    monkeypatch.setenv(HF_TOKEN_ENV, "hf_dummy_token")


def _write_wav(tmp_path):
    wav_path = tmp_path / "lecture.wav"
    wav_path.write_bytes(b"\x00" * 44)  # placeholder — never decoded
    return wav_path


# ---------------------------------------------------------------------------
# HF token handling
# ---------------------------------------------------------------------------


def test_resolve_hf_token_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv(HF_TOKEN_ENV, "hf_abc123")
    assert resolve_hf_token() == "hf_abc123"


def test_missing_hf_token_raises_actionable_error(monkeypatch) -> None:
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)

    with pytest.raises(ValueError) as excinfo:
        resolve_hf_token()

    message = str(excinfo.value)
    assert HF_TOKEN_ENV in message
    assert "huggingface.co/settings/tokens" in message  # actionable


def test_transcriber_constructor_requires_token(monkeypatch) -> None:
    monkeypatch.delenv(HF_TOKEN_ENV, raising=False)
    with pytest.raises(ValueError):
        LocalTranscriber()


# ---------------------------------------------------------------------------
# timestamp + turn formatting
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("seconds", "expected"),
    [
        (0.0, "00:00"),
        (15.4, "00:15"),
        (61.0, "01:01"),
        (3599.0, "59:59"),
        (3600.0, "1:00:00"),   # lectures run long — hours must not wrap
        (7384.0, "2:03:04"),
        (-5.0, "00:00"),       # defensive: never render a negative clock
    ],
)
def test_format_timestamp(seconds, expected) -> None:
    assert _format_timestamp(seconds) == expected


def test_format_turns_merges_intra_sentence_splits() -> None:
    """Pyannote splitting one sentence mid-way must not split the turn.

    The diarizer labels *segments*, not sentences, so a single spoken
    sentence routinely arrives as three adjacent same-speaker segments. They
    must collapse into one turn. Grouping is driven purely by the speaker
    label, so an internal timestamp gap does not break the run — the turn
    still spans first-segment start to last-segment end.
    """
    segments = [
        {"start": 0.0, "end": 4.2, "text": "The doctrine", "speaker": "SPEAKER_00"},
        # Internal gap of 2.8 s: still the same speaker, still one turn.
        {"start": 7.0, "end": 11.5, "text": "of res judicata", "speaker": "SPEAKER_00"},
        {"start": 11.5, "end": 15.0, "text": "is central here.", "speaker": "SPEAKER_00"},
    ]

    assert _format_turns(segments).splitlines() == [
        "[SPEAKER_00] (00:00 - 00:15): The doctrine of res judicata is central here."
    ]


@pytest.mark.parametrize(
    ("gap", "expected_turns"),
    [
        # Sub-threshold silences are segmentation artefacts: keep the sentence.
        (0.0, 1),
        (0.4, 1),
        (2.8, 1),
        (29.9, 1),
        (30.0, 1),   # boundary is inclusive — merges
        # Beyond the threshold it is a real pause: prosody must survive.
        (30.1, 2),
        (45.0, 2),
        (180.0, 2),  # 3-minute gap
    ],
)
def test_format_turns_gap_threshold_splits_or_merges(gap, expected_turns) -> None:
    """Silence beyond MAX_TURN_GAP_SECONDS breaks a turn; at or below it, merges.

    The speaker label is identical on both sides either way — only the
    duration of the silence decides. Preserving these boundaries is what lets
    Stage 3 read lecturer pauses back out of the transcript.
    """
    # The parametrized boundaries are expressed against the documented 30 s, so
    # assert that contract directly rather than restating the number.
    assert MAX_TURN_GAP_SECONDS == 30.0

    segments = [
        {"start": 0.0, "end": 8.0, "text": "Good morning,", "speaker": "SPEAKER_00"},
        {
            "start": 8.0 + gap,
            "end": 8.0 + gap + 7.0,
            "text": "please sit down.",
            "speaker": "SPEAKER_00",
        },
    ]

    lines = _format_turns(segments).splitlines()

    assert len(lines) == expected_turns
    if expected_turns == 1:
        # One turn spanning first start to final end, the sub-threshold gap
        # legitimately included in the range.
        assert lines == [
            f"[SPEAKER_00] (00:00 - "
            f"{_format_timestamp(15.0 + gap)}): Good morning, please sit down."
        ]
    else:
        # Two turns, same label, and the first turn keeps its true 8 s span
        # rather than absorbing the silence.
        assert lines == [
            "[SPEAKER_00] (00:00 - 00:08): Good morning,",
            f"[SPEAKER_00] ({_format_timestamp(8.0 + gap)} - "
            f"{_format_timestamp(15.0 + gap)}): please sit down.",
        ]


def test_format_turns_splits_only_at_the_long_pause() -> None:
    """A sentence split twice survives unless one of the gaps is a real pause."""
    # 2.8 s and 1.1 s gaps — both sub-threshold, so one continuous turn.
    choppy = [
        {"start": 0.0, "end": 4.2, "text": "The doctrine", "speaker": "SPEAKER_00"},
        {"start": 7.0, "end": 11.5, "text": "of res judicata", "speaker": "SPEAKER_00"},
        {"start": 12.6, "end": 15.0, "text": "matters.", "speaker": "SPEAKER_00"},
    ]
    assert _format_turns(choppy).splitlines() == [
        "[SPEAKER_00] (00:00 - 00:15): The doctrine of res judicata matters."
    ]

    # Same sentence, but the lecturer now pauses 60 s before the last clause.
    paused = [
        {"start": 0.0, "end": 4.2, "text": "The doctrine", "speaker": "SPEAKER_00"},
        {"start": 7.0, "end": 11.5, "text": "of res judicata", "speaker": "SPEAKER_00"},
        {"start": 71.5, "end": 75.0, "text": "matters.", "speaker": "SPEAKER_00"},
    ]
    assert _format_turns(paused).splitlines() == [
        "[SPEAKER_00] (00:00 - 00:11): The doctrine of res judicata",
        "[SPEAKER_00] (01:11 - 01:15): matters.",
    ]


def test_format_turns_keeps_speaker_alternation_chronological() -> None:
    """A returning speaker starts a NEW turn — dialogue order is preserved.

    Merging is per speaker *label run*, not per speaker: collapsing
    SPEAKER_00's two appearances across SPEAKER_01 would reorder the
    conversation and hide the turn-taking the transcript exists to show.
    """
    segments = [
        {"start": 0.0, "end": 8.0, "text": "alpha", "speaker": "SPEAKER_00"},
        {"start": 8.0, "end": 15.0, "text": "beta", "speaker": "SPEAKER_01"},
        {"start": 15.0, "end": 31.0, "text": "gamma", "speaker": "SPEAKER_00"},
    ]

    assert _format_turns(segments).splitlines() == [
        "[SPEAKER_00] (00:00 - 00:08): alpha",
        "[SPEAKER_01] (00:08 - 00:15): beta",
        "[SPEAKER_00] (00:15 - 00:31): gamma",
    ]


def test_format_turns_blank_text_does_not_break_a_run() -> None:
    """An empty segment between two same-speaker segments is not a boundary."""
    segments = [
        {"start": 0.0, "end": 8.0, "text": "one", "speaker": "SPEAKER_00"},
        {"start": 8.0, "end": 9.0, "text": "   ", "speaker": "SPEAKER_00"},
        {"start": 9.0, "end": 15.0, "text": "two", "speaker": "SPEAKER_00"},
    ]

    assert _format_turns(segments).splitlines() == [
        "[SPEAKER_00] (00:00 - 00:15): one two"
    ]


def test_format_turns_groups_contiguous_speakers() -> None:
    """Consecutive segments from one speaker merge into a single turn."""
    segments = [
        {"start": 0.0, "end": 8.0, "text": "Good morning,", "speaker": "SPEAKER_00"},
        {"start": 8.0, "end": 15.0, "text": "please sit down.", "speaker": "SPEAKER_00"},
        {"start": 15.0, "end": 22.0, "text": "Is the recording on?", "speaker": "SPEAKER_01"},
        {"start": 22.0, "end": 31.0, "text": "Yes, it is running.", "speaker": "SPEAKER_00"},
    ]

    turns = _format_turns(segments)

    assert turns.splitlines() == [
        "[SPEAKER_00] (00:00 - 00:15): Good morning, please sit down.",
        "[SPEAKER_01] (00:15 - 00:22): Is the recording on?",
        "[SPEAKER_00] (00:22 - 00:31): Yes, it is running.",
    ]


def test_format_turns_handles_unattributed_and_empty_segments() -> None:
    """Unlabelled segments get a placeholder; blank ones are dropped."""
    segments = [
        {"start": 0.0, "end": 4.0, "text": "  ", "speaker": "SPEAKER_00"},
        {"start": 4.0, "end": 9.0, "text": "mumbled line", "speaker": None},
        {"start": 9.0, "end": 12.0, "text": "clearly spoken"},
        {"start": 40.0, "end": 44.0, "text": "and another", "speaker": "SPEAKER_01"},
    ]

    turns = _format_turns(segments)

    # Consecutive unattributed segments merge into ONE turn, and the blank
    # segment is dropped entirely without breaking the grouping.
    assert turns.splitlines() == [
        f"[{UNKNOWN_SPEAKER}] (00:04 - 00:12): mumbled line clearly spoken",
        "[SPEAKER_01] (00:40 - 00:44): and another",
    ]
    assert _format_turns([]) == ""
    assert _format_turns([{"start": 0.0, "end": 1.0, "text": "  "}]) == ""


# ---------------------------------------------------------------------------
# the three-stage pipeline
# ---------------------------------------------------------------------------


def test_transcribe_audio_runs_all_three_stages(mocker, tmp_path) -> None:
    """transcribe -> align -> diarize run in order against a loaded audio array."""
    wav_path = _write_wav(tmp_path)
    segments = [
        {"start": 0.0, "end": 12.5, "text": "Welcome to the lecture.", "speaker": "SPEAKER_00"},
    ]
    whisperx = _make_whisperx(mocker, segments)

    transcriber = LocalTranscriber()
    transcript_path = transcriber.transcribe_audio(wav_path)

    # -- Stage A: model loaded on CPU/int8, fed the decoded audio array -----
    whisperx.load_model.assert_called_once_with(
        "base", device="cpu", compute_type="int8"
    )
    transcribe_call = whisperx.load_model.return_value.transcribe.call_args
    assert transcribe_call.args[0] == whisperx.load_audio.return_value
    assert transcribe_call.kwargs["language"] == "en"
    assert "Swahili" in transcribe_call.kwargs["initial_prompt"]  # priming intact

    # -- Stage B: alignment keyed on the detected language ------------------
    whisperx.load_align_model.assert_called_once_with(language_code="en", device="cpu")
    align_call = whisperx.align.call_args
    assert align_call.args[0] == segments      # segments first
    assert align_call.args[3] == whisperx.load_audio.return_value  # audio fourth

    # -- Stage C: diarization then speaker assignment -----------------------
    whisperx.DiarizationPipeline.assert_called_once()
    diarize_call = whisperx.assign_word_speakers.call_args
    assert diarize_call.args[1]["segments"] == segments  # transcript_result 2nd

    # -- return contract + artifact ----------------------------------------
    assert transcript_path == tmp_path / TRANSCRIPT_FILENAME
    assert transcript_path.read_text(encoding="utf-8") == (
        "[SPEAKER_00] (00:00 - 00:12): Welcome to the lecture."
    )


def test_transcribe_audio_uses_injected_token(mocker, tmp_path) -> None:
    """An explicit hf_token overrides the environment for the diarizer."""
    wav_path = _write_wav(tmp_path)
    whisperx = _make_whisperx(mocker, [{"start": 0.0, "end": 1.0, "text": "hi"}])

    LocalTranscriber(hf_token="hf_injected").transcribe_audio(wav_path)

    kwargs = whisperx.DiarizationPipeline.call_args.kwargs
    assert kwargs["device"] == "cpu"
    assert "hf_injected" in kwargs.values()


def test_transcribe_audio_rejects_missing_wav(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        LocalTranscriber().transcribe_audio(tmp_path / "nope.wav")
