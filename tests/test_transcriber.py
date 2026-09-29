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

from pathlib import Path

import pytest

import transcriber
from transcriber import (
    BATCH_SIZE,
    BATCH_SIZE_ENV,
    COMPUTE_TYPE,
    COMPUTE_TYPE_ENV,
    DEFAULT_LANGUAGE,
    HF_TOKEN_ENV,
    LocalTranscriber,
    MAX_TURN_GAP_SECONDS,
    TRANSCRIPT_FILENAME,
    UNKNOWN_SPEAKER,
    _format_turns,
    format_timestamp,
    resolve_batch_size,
    resolve_compute_type,
    resolve_hf_token,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_whisperx(mocker, segments, language="en", speakers=None):
    """Stub the three WhisperX stages; return ``(whisperx, pipeline_cls, assign)``.

    The diarization pair is patched on ``transcriber`` itself, not on the
    ``whisperx`` module, because ``transcriber`` imports those two names
    directly from ``whisperx.diarize`` to sidestep the root namespace.

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

    assign = mocker.patch("transcriber.assign_word_speakers")
    assign.return_value = {"segments": list(segments)}

    pipeline_cls = mocker.patch("transcriber.DiarizationPipeline")
    pipeline_cls.return_value.return_value = (
        speakers if speakers is not None else mocker.MagicMock(name="diarize_df")
    )
    return whisperx, pipeline_cls, assign


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
    assert format_timestamp(seconds) == expected


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
            f"{format_timestamp(15.0 + gap)}): Good morning, please sit down."
        ]
    else:
        # Two turns, same label, and the first turn keeps its true 8 s span
        # rather than absorbing the silence.
        assert lines == [
            "[SPEAKER_00] (00:00 - 00:08): Good morning,",
            f"[SPEAKER_00] ({format_timestamp(8.0 + gap)} - "
            f"{format_timestamp(15.0 + gap)}): please sit down.",
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
    whisperx, pipeline_cls, assign = _make_whisperx(mocker, segments)

    transcriber = LocalTranscriber()
    transcript_path = transcriber.transcribe_audio(wav_path)

    # -- Stage A: model loaded on CPU/int8, fed the decoded audio array -----
    # The priming prompt rides in via asr_options at LOAD time. load_model
    # merges it into the TranscriptionOptions it builds; transcribe() has no
    # initial_prompt parameter, so passing it there is a TypeError.
    # The prompt lobotomy: initial_prompt removed to prevent hallucination bias.
    # condition_on_previous_text=False breaks Whisper's repetition loop, the
    # lowered no_speech_threshold biases ambiguous audio toward silence, and
    # the raised log_prob_threshold drops segments the decoder is unsure of.
    whisperx.load_model.assert_called_once_with(
        "base",
        device="cpu",
        compute_type=transcriber.compute_type,
        asr_options={
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.5,
            "log_prob_threshold": -0.8,
        },
        vad_options={"vad_onset": 0.8, "vad_offset": 0.6},
    )
    transcribe_call = whisperx.load_model.return_value.transcribe.call_args
    assert transcribe_call.args[0] == whisperx.load_audio.return_value
    # Regression guard: the prompt must NOT be passed to transcribe() again.
    assert "initial_prompt" not in transcribe_call.kwargs
    # The language is now supplied explicitly to skip detect_language().
    assert transcribe_call.kwargs["language"] == "en"
    assert transcribe_call.kwargs["batch_size"] == transcriber.batch_size
    # load_model() must still take no language: a preset there builds the
    # tokenizer eagerly, and the per-call value is the only place that needs it.
    assert "language" not in whisperx.load_model.call_args.kwargs

    # -- Stage B: alignment keyed on the caller's language -------------------
    whisperx.load_align_model.assert_called_once_with(language_code="en", device="cpu")
    align_call = whisperx.align.call_args
    assert align_call.args[0] == segments      # segments first
    assert align_call.args[3] == whisperx.load_audio.return_value  # audio fourth

    # -- Stage C: diarization then speaker assignment -----------------------
    # Both names are the directly-imported ones, patched on the transcriber
    # module rather than reached through the whisperx namespace.
    pipeline_cls.assert_called_once()
    assert assign.call_args.args[1]["segments"] == segments  # transcript_result 2nd
    assert assign.call_args.kwargs == {}  # the bounds do NOT belong here
    # Clustering bounds belong to the pyannote invocation, whose __call__ is
    # ``(audio, num_speakers, min_speakers, max_speakers, ...)``. min=1 allows
    # a mono-speaker recording, max=10 stops a short noisy exchange from
    # shattering into a dozen one-line speakers.
    diarize_call = pipeline_cls.return_value.call_args
    assert diarize_call.args[0] == whisperx.load_audio.return_value  # audio first
    assert diarize_call.kwargs == {"min_speakers": 1, "max_speakers": 10}

    # -- return contract + artifact ----------------------------------------
    assert transcript_path == tmp_path / TRANSCRIPT_FILENAME
    assert transcript_path.read_text(encoding="utf-8") == (
        "[SPEAKER_00] (00:00 - 00:12): Welcome to the lecture."
    )


def test_transcribe_audio_injects_prompt_via_asr_options(mocker, tmp_path) -> None:
    """The Swahili/English prompt reaches the model as an asr_option.

    ``load_model`` builds ``TranscriptionOptions`` once and merges this dict
    over its defaults (``initial_prompt`` defaults to ``None``). The
    alternative — ``transcribe(initial_prompt=...)`` — is a TypeError against
    a real ``FasterWhisperPipeline``, because that method has no such
    parameter. This test pins the correct channel so the bug cannot return.
    """
    wav_path = _write_wav(tmp_path)
    whisperx = _make_whisperx(mocker, [{"start": 0.0, "end": 1.0, "text": "hi"}])[0]

    LocalTranscriber().transcribe_audio(wav_path)

    load_kwargs = whisperx.load_model.call_args.kwargs
    # Prompt lobotomy: initial_prompt must NOT be present in asr_options.
    assert "initial_prompt" not in load_kwargs["asr_options"]
    assert load_kwargs["asr_options"] == {
        "condition_on_previous_text": False,
        "no_speech_threshold": 0.5,
        "log_prob_threshold": -0.8,
    }

    # And transcribe() must receive only arguments it actually accepts.
    transcribe_kwargs = whisperx.load_model.return_value.transcribe.call_args.kwargs
    assert set(transcribe_kwargs) <= {
        "batch_size",
        "num_workers",
        "language",
        "task",
        "chunk_size",
        "print_progress",
        "combined_progress",
        "verbose",
        "progress_callback",
    }, f"unsupported kwargs for FasterWhisperPipeline.transcribe: {transcribe_kwargs}"


def test_explicit_language_is_routed_to_asr_and_alignment(mocker, tmp_path) -> None:
    """A non-English code must reach *both* the decoder and the aligner.

    Passing a wrong language to Whisper does not raise — it silently returns
    fluent text in the wrong language, and every word timestamp is then
    anchored to a phoneme model for the wrong language too. So the code the
    operator typed has to arrive at both stages, not just the first.

    ``load_model`` is asserted separately: it must *not* receive the language,
    because a preset there builds the tokenizer eagerly and would override the
    per-call value.
    """
    wav_path = _write_wav(tmp_path)
    whisperx, _pipeline, _assign = _make_whisperx(
        mocker, [{"start": 0.0, "end": 1.0, "text": "habari"}], language="sw"
    )

    LocalTranscriber().transcribe_audio(wav_path, language="sw")

    transcribe_kwargs = whisperx.load_model.return_value.transcribe.call_args.kwargs
    assert transcribe_kwargs["language"] == "sw"
    whisperx.load_align_model.assert_called_once_with(
        language_code="sw", device="cpu"
    )
    assert "language" not in whisperx.load_model.call_args.kwargs


def test_alignment_language_is_not_laundered_from_the_result(mocker, tmp_path) -> None:
    """The aligner uses the requested code, not ``result["language"]``.

    WhisperX echoes the *requested* language back in its result dict, so
    reading it there is a no-op that looks like validation. Worse, it would
    silently "confirm" a bad guess: a Swahili lecture transcribed as ``en``
    reports ``language="en"``, and any future code trusting the result would
    bake the error in. The mock below reports a deliberately different code so
    a regression is visible.
    """
    wav_path = _write_wav(tmp_path)
    whisperx, _pipeline, _assign = _make_whisperx(
        mocker, [{"start": 0.0, "end": 1.0, "text": "habari"}], language="en"
    )

    LocalTranscriber().transcribe_audio(wav_path, language="sw")

    # Requested "sw" must win even though the result claims "en".
    whisperx.load_align_model.assert_called_once_with(
        language_code="sw", device="cpu"
    )


def test_batch_size_routes_to_transcribe(mocker, tmp_path) -> None:
    """Batching is the main throughput lever and must reach the decoder."""
    wav_path = _write_wav(tmp_path)
    whisperx, _pipeline, _assign = _make_whisperx(
        mocker, [{"start": 0.0, "end": 1.0, "text": "hi"}]
    )

    LocalTranscriber(batch_size=32).transcribe_audio(wav_path)

    kwargs = whisperx.load_model.return_value.transcribe.call_args.kwargs
    assert kwargs["batch_size"] == 32


# ---------------------------------------------------------------------------
# Compute type / batch size resolution
# ---------------------------------------------------------------------------


def test_compute_type_defaults_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv(COMPUTE_TYPE_ENV, raising=False)
    assert resolve_compute_type() == COMPUTE_TYPE


def test_compute_type_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv(COMPUTE_TYPE_ENV, "int8")
    assert resolve_compute_type() == "int8"


def test_blank_compute_type_env_falls_back_to_default(monkeypatch) -> None:
    monkeypatch.setenv(COMPUTE_TYPE_ENV, "   ")
    assert resolve_compute_type() == COMPUTE_TYPE


def test_invalid_compute_type_is_rejected_with_guidance(monkeypatch) -> None:
    """Bad quantization must fail before the WAV is decoded, not inside CTranslate2."""
    monkeypatch.setenv(COMPUTE_TYPE_ENV, "int4")
    with pytest.raises(ValueError) as excinfo:
        resolve_compute_type()
    message = str(excinfo.value)
    assert COMPUTE_TYPE_ENV in message
    assert "float16" in message  # lists the valid values


def test_batch_size_defaults_when_env_unset(monkeypatch) -> None:
    monkeypatch.delenv(BATCH_SIZE_ENV, raising=False)
    assert resolve_batch_size() == BATCH_SIZE


def test_batch_size_reads_environment(monkeypatch) -> None:
    monkeypatch.setenv(BATCH_SIZE_ENV, "32")
    assert resolve_batch_size() == 32


def test_non_numeric_batch_size_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv(BATCH_SIZE_ENV, "lots")
    with pytest.raises(ValueError) as excinfo:
        resolve_batch_size()
    assert BATCH_SIZE_ENV in str(excinfo.value)


@pytest.mark.parametrize("bad", ["0", "-4"])
def test_non_positive_batch_size_is_rejected(monkeypatch, bad) -> None:
    """0 and negatives are *accepted* by faster_whisper and fail deep in the
    decoder, so they are rejected up front with the value quoted."""
    monkeypatch.setenv(BATCH_SIZE_ENV, bad)
    with pytest.raises(ValueError) as excinfo:
        resolve_batch_size()
    assert ">= 1" in str(excinfo.value)


def test_constructor_argument_beats_environment(monkeypatch) -> None:
    """Explicit arguments win over env, so tests and callers need no env mutation."""
    monkeypatch.setenv(COMPUTE_TYPE_ENV, "int8")
    monkeypatch.setenv(BATCH_SIZE_ENV, "8")
    transcriber = LocalTranscriber(compute_type="float16", batch_size=64)
    assert transcriber.compute_type == "float16"
    assert transcriber.batch_size == 64


def test_constructor_picks_up_env_by_default(monkeypatch) -> None:
    monkeypatch.setenv(COMPUTE_TYPE_ENV, "int8")
    monkeypatch.setenv(BATCH_SIZE_ENV, "8")
    transcriber = LocalTranscriber()
    assert transcriber.compute_type == "int8"
    assert transcriber.batch_size == 8


def test_module_loads_dotenv_at_import() -> None:
    """``transcriber`` must read ``.env`` itself, not lean on ``summarizer``.

    The two tunables are documented as ``.env`` settings, so the module that
    reads them has to load that file. It previously depended on ``summarizer``
    loading it as an import side effect -- true under ``main.py``, false the
    moment this module is imported alone, which silently fell back to the code
    defaults (float16 rather than the configured int8).
    """
    source = Path(transcriber.__file__).read_text(encoding="utf-8")
    assert "load_dotenv()" in source, "transcriber never calls load_dotenv()"


def test_real_environment_beats_dotenv(monkeypatch) -> None:
    """An exported variable must win over the ``.env`` file.

    ``load_dotenv(override=False)`` is the python-dotenv default, and this
    pins it: without it, a re-load of ``.env`` would clobber a test's
    ``monkeypatch.setenv`` and reintroduce the real NVIDIA_API_KEY leak that
    ``test_summarizer.py`` already has to work around.
    """
    monkeypatch.setenv(COMPUTE_TYPE_ENV, "int8_float16")
    assert resolve_compute_type() == "int8_float16"


def test_compute_type_reaches_load_model(mocker, tmp_path) -> None:
    """The resolved value must actually be passed, not just stored."""
    wav_path = _write_wav(tmp_path)
    whisperx = _make_whisperx(
        mocker, [{"start": 0.0, "end": 1.0, "text": "hi"}]
    )[0]

    LocalTranscriber(compute_type="int8").transcribe_audio(wav_path)

    assert whisperx.load_model.call_args.kwargs["compute_type"] == "int8"


def test_asr_and_vad_options_use_the_field_names_whisperx_actually_has(
    mocker, tmp_path
) -> None:
    """The tuning keys must be real fields, or load_model raises TypeError.

    ``load_model`` merges this dict into its defaults and then splats the
    result into ``TranscriptionOptions(**...)``. Any key that is not a field
    name on that dataclass is a hard TypeError at load, not a silently ignored
    option. The one that bites is the log-probability gate: the field is
    ``log_prob_threshold`` (underscored), so the intuitive
    ``logprob_threshold`` looks right and silently crashes the pipeline. This
    test pins the exact key set so the spellings cannot drift.
    """
    wav_path = _write_wav(tmp_path)
    whisperx = _make_whisperx(mocker, [{"start": 0.0, "end": 1.0, "text": "hi"}])[0]

    LocalTranscriber().transcribe_audio(wav_path)

    load_kwargs = whisperx.load_model.call_args.kwargs

    asr_options = load_kwargs["asr_options"]
    assert "log_prob_threshold" in asr_options
    assert "logprob_threshold" not in asr_options  # not a TranscriptionOptions field
    assert asr_options["log_prob_threshold"] == -0.8

    vad_options = load_kwargs["vad_options"]
    assert vad_options == {"vad_onset": 0.8, "vad_offset": 0.6}
    # Both are genuine default_vad_options keys in whisperx.asr.load_model.
    assert set(vad_options) <= {"chunk_size", "vad_onset", "vad_offset"}


def test_transcribe_audio_uses_injected_token(mocker, tmp_path) -> None:
    """An explicit hf_token overrides the environment for the diarizer."""
    wav_path = _write_wav(tmp_path)
    _whisperx, pipeline_cls, _assign = _make_whisperx(
        mocker, [{"start": 0.0, "end": 1.0, "text": "hi"}]
    )

    LocalTranscriber(hf_token="hf_injected").transcribe_audio(wav_path)

    kwargs = pipeline_cls.call_args.kwargs
    assert kwargs["device"] == "cpu"
    assert "hf_injected" in kwargs.values()


def test_diarization_kwargs_fall_back_from_legacy_to_current(mocker, tmp_path) -> None:
    """The legacy ``use_auth_token`` is tried first, then ``token``.

    WhisperX renamed the constructor argument in 3.2, so a current install
    must reject the legacy name from the signature and fall through to
    ``token=``. The stand-in below is deliberately strict (no ``**kwargs``)
    so it reproduces that TypeError exactly as the real class does.
    """
    wav_path = _write_wav(tmp_path)
    _make_whisperx(mocker, [{"start": 0.0, "end": 1.0, "text": "hi"}])

    calls = []

    class ModernPipeline:
        """Signature-faithful stand-in for DiarizationPipeline(token=...)."""

        def __init__(self, token=None, device="cpu"):
            calls.append({"token": token, "device": device})
            if token == "bad_token":
                # A real auth failure surfaces as ValueError, NOT TypeError.
                raise ValueError("401 Unauthorized from pyannote")

        def __call__(self, audio, num_speakers=None, min_speakers=None,
                     max_speakers=None):
            return ["diarize_df_placeholder"]

    mocker.patch("transcriber.DiarizationPipeline", ModernPipeline)

    LocalTranscriber(hf_token="hf_good").transcribe_audio(wav_path)

    # Exactly one surviving call: the legacy attempt raised TypeError, so only
    # the modern token= form was constructed.
    assert calls == [{"token": "hf_good", "device": "cpu"}]


def test_diarization_auth_error_is_not_masked_by_the_fallback(mocker, tmp_path) -> None:
    """A non-signature failure must propagate, not be retried as a TypeError.

    The fallback catches ``TypeError`` only. If a bad token or an unreachable
    model raised something else it must reach the user verbatim, otherwise the
    retry would silently hide the real cause behind a second, equally opaque
    error.
    """
    wav_path = _write_wav(tmp_path)
    _make_whisperx(mocker, [{"start": 0.0, "end": 1.0, "text": "hi"}])

    class StrictPipeline:
        def __init__(self, token=None, device="cpu"):
            raise ValueError("401 Unauthorized from pyannote")

        def __call__(self, audio, **kwargs):
            return []

    mocker.patch("transcriber.DiarizationPipeline", StrictPipeline)

    with pytest.raises(ValueError, match="401 Unauthorized"):
        LocalTranscriber(hf_token="bad_token").transcribe_audio(wav_path)


def test_transcribe_audio_rejects_missing_wav(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        LocalTranscriber().transcribe_audio(tmp_path / "nope.wav")


def test_speaker_bounds_go_to_the_pipeline_not_the_assigner(mocker, tmp_path) -> None:
    """Regression guard: min/max_speakers are ``__call__`` args, not assign args.

    ``assign_word_speakers`` has the signature
    ``(diarize_df, transcript_result, speaker_embeddings, fill_nearest)`` and
    rejects ``min_speakers``/``max_speakers`` outright; those belong to
    ``DiarizationPipeline.__call__``. Both stand-ins below reproduce the real
    signatures *exactly* — no ``**kwargs`` — so putting the bounds on the wrong
    function raises TypeError here instead of surviving to a live run where the
    user meets it mid-lecture.
    """
    wav_path = _write_wav(tmp_path)
    whisperx = _make_whisperx(mocker, [{"start": 0.0, "end": 1.0, "text": "hi"}])[0]
    seen: dict[str, object] = {}

    class RealPipeline:
        """Mirrors ``DiarizationPipeline.__call__``'s real parameter list."""

        def __init__(self, model_name=None, token=None, device="cpu", cache_dir=None):
            seen["constructed_with"] = {"token": token, "device": device}

        def __call__(
            self,
            audio,
            num_speakers=None,
            min_speakers=None,
            max_speakers=None,
            return_embeddings=False,
            progress_callback=None,
        ):
            seen["call_kwargs"] = {
                "min_speakers": min_speakers,
                "max_speakers": max_speakers,
            }
            seen["audio"] = audio
            return "diarize_df"

    def real_assign_word_speakers(
        diarize_df, transcript_result, speaker_embeddings=None, fill_nearest=False
    ):
        seen["assign_kwargs"] = {
            "speaker_embeddings": speaker_embeddings,
            "fill_nearest": fill_nearest,
        }
        return transcript_result

    mocker.patch("transcriber.DiarizationPipeline", RealPipeline)
    mocker.patch("transcriber.assign_word_speakers", real_assign_word_speakers)

    LocalTranscriber().transcribe_audio(wav_path)

    # The bounds reached the pipeline invocation...
    assert seen["call_kwargs"] == {"min_speakers": 1, "max_speakers": 10}
    assert seen["audio"] == whisperx.load_audio.return_value
    # ...and the assigner got only its own four parameters.
    assert seen["assign_kwargs"] == {
        "speaker_embeddings": None,
        "fill_nearest": False,
    }
    # The audio is decoded once and shared across all three stages.
    assert whisperx.load_audio.call_count == 1
