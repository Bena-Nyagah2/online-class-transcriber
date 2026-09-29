"""Phase 2 — Lecture audio transcription + diarization via WhisperX (local).

Turns a recorded lecture WAV into a speaker-attributed transcript entirely
on-device, in three stages:

1.  **Transcribe** — ``whisperx.load_model("base", device="cpu",
    compute_type="int8")`` runs OpenAI Whisper locally (CTranslate2 backend).
    No HTTP calls, no API keys, no request limits.
2.  **Align** — ``whisperx.load_align_model`` + ``whisperx.align`` attach
    word-level timestamps using a wav2vec2 phoneme model. The output is
    segmented per sentence, which is what makes speaker turns crisp.
3.  **Diarize** — ``DiarizationPipeline`` (from ``whisperx.diarize``) runs
    pyannote to work out *who* spoke when, then ``assign_word_speakers``
    labels every segment and word with a speaker ID.

The final ``raw_transcript.txt`` groups contiguous turns from the same
speaker::

    [SPEAKER_00] (00:00 - 00:15): Good morning, please sit down.
    [SPEAKER_01] (00:15 - 00:31): Excuse me, is the recording on?

Configuration
-------------
* ``HF_TOKEN`` must name a Hugging Face access token that has accepted the
  ``pyannote/speaker-diarization-community-1`` model terms, otherwise stage 3
  cannot download the diarization weights. Read from the environment at
  construction time so tests can inject a dummy value.
* Model ``base`` is small and fast — ideal for testing. Swap
  :data:`WHISPER_MODEL` for ``"large-v3"`` when production-grade accuracy is
  needed.
* ``WHISPER_COMPUTE_TYPE`` and ``WHISPER_BATCH_SIZE`` tune throughput. See
  :data:`COMPUTE_TYPE` — the fast value is device-dependent, and ``int8`` beats
  ``float16`` on the CPU device this module pins by default.
* The spoken language is supplied per call (``transcribe_audio(..., language=)``)
  rather than auto-detected, which trades a little flexibility for a skipped
  detection pass. Callers prompt for it; the default is :data:`DEFAULT_LANGUAGE`.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional, Union

import whisperx
# The diarization classes are imported from their defining module rather than
# reached through the root ``whisperx`` namespace. That namespace is not
# stable across releases: v3.2.0 re-exports both names, while current main
# exports only ``assign_word_speakers``. Worse, if ``whisperx.diarize`` fails
# to import partway, Python leaves a partially initialised ``whisperx`` whose
# attributes were never bound, and the resulting AttributeError on the root
# namespace masks the real cause. Importing directly works on every version
# and surfaces the genuine error if the submodule itself is broken.
from whisperx.diarize import DiarizationPipeline, assign_word_speakers

from dotenv import load_dotenv

from recorder import SAMPLE_RATE

# ``WHISPER_COMPUTE_TYPE`` / ``WHISPER_BATCH_SIZE`` are documented as .env
# settings, so this module has to load that file itself. It previously relied
# on ``summarizer`` doing it as an import side effect -- which happens to work
# under main.py, since summarizer is imported before any transcriber is
# constructed, but silently yields the code defaults (notably float16) the
# moment this module is imported on its own. That is exactly the kind of
# difference between "configured" and "appears configured" that costs a
# multi-hour CPU transcription to discover.
#
# ``override`` stays False (the python-dotenv default) so a value already
# exported in the real environment always wins over the file, and tests that
# monkeypatch os.environ are never clobbered by re-loading .env.
load_dotenv()

TRANSCRIPT_FILENAME = "raw_transcript.txt"

HF_TOKEN_ENV: str = "HF_TOKEN"
"""Environment variable holding the Hugging Face access token."""

COMPUTE_TYPE_ENV: str = "WHISPER_COMPUTE_TYPE"
"""Environment variable overriding :data:`COMPUTE_TYPE`."""

BATCH_SIZE_ENV: str = "WHISPER_BATCH_SIZE"
"""Environment variable overriding :data:`BATCH_SIZE`."""

WHISPER_MODEL: str = "base"
"""Whisper model size. ``base`` for speed; ``large-v3`` for accuracy."""

DEVICE: str = "cpu"
"""Torch device for every stage. CPU-only keeps the pipeline dependency-free."""

COMPUTE_TYPE: str = "float16"
"""CTranslate2 quantization used when :data:`COMPUTE_TYPE_ENV` is unset.

CTranslate2 accepts ``float32`` / ``float16`` / ``int8_float16`` / ``int8``.
Which one is fastest is decided entirely by the *device*, not by taste:

* On **CUDA**, ``float16`` is the right pick and roughly doubles throughput
  over ``float32``.
* On **CPU** — which is what :data:`DEVICE` pins — ``float16`` buys nothing.
  CTranslate2 has to emulate half precision, so ``int8`` is normally the
  *faster* choice, not the slower one.

That asymmetry is why this is an environment override rather than a constant
baked into the call: the right value depends on the machine, and the wrong one
is not a crash but a silent multi-hour transcription. Set
``WHISPER_COMPUTE_TYPE=int8`` for CPU-only hosts, ``float16`` once
:data:`DEVICE` is switched to ``cuda``.
"""

DEFAULT_LANGUAGE: str = "en"
"""Spoken-language code used when the caller does not specify one.

Pinned rather than auto-detected. ``detect_language()`` scores the first 30 s
of audio with a separate model pass before the real transcription even starts,
which on a CPU-only host is a measurable slice of a multi-hour job. Supplying
the code also makes the run deterministic, so two passes over the same
recording cannot disagree about what language they are.
"""

BATCH_SIZE: int = 16
"""Number of audio chunks Whisper decodes in parallel.

Batching is the single largest throughput lever in WhisperX after the compute
type: a 90-minute lecture is thousands of short windows, and the default of 1
processes them strictly one at a time. 16 keeps peak memory modest while
amortising per-chunk overhead across a wide batch. Raise it for short models on
a GPU; lower it if a long lecture at ``large-v3`` exhausts RAM.
"""

UNKNOWN_SPEAKER: str = "SPEAKER_??"
"""Label used for segments pyannote could not attribute to a speaker."""

MAX_TURN_GAP_SECONDS: float = 30.0
"""Longest silence that still counts as *inside* one speaker turn.

Two same-speaker segments merge unless the silence between them exceeds this
value. The threshold separates the two cases that look identical in the data:

* **Intra-sentence splits.** Pyannote labels segments, not sentences, and
  routinely breaks one sentence at a sub-second hesitation. Merging these is
  the whole point — otherwise sentences are chopped at the one place a reader
  least expects it.
* **Genuine pauses.** A lecturer stopping to think, take a question, or wait
  for a slide is *proseody*, and Stage 3 of the pipeline detects lecturer
  emphasis from structural pauses in this file. Folding a long silence into
  the previous turn would erase that signal and report a bogus turn length —
  a 15-second remark inside a 3-minute span.

30 s sits far above a mid-sentence hesitation and far below a real pause in
discourse, so neither case is misclassified. Raise it only if transcripts
still fragment mid-sentence; lower it if pauses are still being swallowed.
"""


def resolve_hf_token() -> str:
    """Read the Hugging Face token used by the diarization stage.

    Returns:
        The token from ``HF_TOKEN``.

    Raises:
        ValueError: The variable is unset or empty, with instructions for
            where to get a working token.
    """
    token = os.environ.get(HF_TOKEN_ENV, "").strip()
    if not token:
        raise ValueError(
            f"{HF_TOKEN_ENV} is not set — speaker diarization needs a Hugging "
            "Face token. Create one at https://huggingface.co/settings/tokens, "
            "then accept the terms for "
            "https://huggingface.co/pyannote/speaker-diarization-community-1 "
            f"and export {HF_TOKEN_ENV}=hf_..."
        )
    return token


def resolve_compute_type() -> str:
    """Read the CTranslate2 compute type from the environment.

    Returns:
        ``WHISPER_COMPUTE_TYPE`` if set to a non-blank value, else
        :data:`COMPUTE_TYPE`.

    Raises:
        ValueError: The variable is set to a value CTranslate2 will reject.
            Failing here beats failing inside ``load_model`` after the WAV has
            already been decoded.
    """
    raw = os.environ.get(COMPUTE_TYPE_ENV, "").strip()
    if not raw:
        return COMPUTE_TYPE

    valid = {"float32", "float16", "int8_float32", "int8_float16", "int8"}
    if raw not in valid:
        raise ValueError(
            f"{COMPUTE_TYPE_ENV}={raw!r} is not a CTranslate2 compute type. "
            f"Valid values: {', '.join(sorted(valid))}."
        )
    return raw


def resolve_batch_size() -> int:
    """Read the Whisper decode batch size from the environment.

    Returns:
        ``WHISPER_BATCH_SIZE`` if set to a valid positive integer, else
        :data:`BATCH_SIZE`.

    Raises:
        ValueError: The variable is non-numeric, or not strictly positive. A
            zero or negative batch size is accepted by faster_whisper and
            produces a confusing failure deep inside the decoder, so it is
            rejected up front with the offending value quoted.
    """
    raw = os.environ.get(BATCH_SIZE_ENV, "").strip()
    if not raw:
        return BATCH_SIZE
    try:
        value = int(raw)
    except ValueError as exc:
        raise ValueError(
            f"{BATCH_SIZE_ENV}={raw!r} is not an integer."
        ) from exc
    if value < 1:
        raise ValueError(
            f"{BATCH_SIZE_ENV}={value} must be >= 1. A batch size of 0 or less "
            "is accepted by faster_whisper but fails inside the decoder."
        )
    return value


def format_timestamp(seconds: float) -> str:
    """Render an offset as ``MM:SS``, or ``H:MM:SS`` past an hour.

    Lectures routinely run past 60 minutes, so the hour component is only
    added when it is actually needed to keep the field width stable.

    This is the single canonical clock renderer for the project: the transcript
    turn headers and the slide manifest both go through it, which is what
    makes a manifest timestamp directly comparable to a transcript timestamp
    in the Stage 3 prompt.

    Args:
        seconds: Offset from the start of the recording.

    Returns:
        A zero-padded clock string.
    """
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, secs = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _build_diarization_pipeline(token: str) -> Any:
    """Construct a pyannote-backed diarization pipeline.

    WhisperX renamed this constructor's ``use_auth_token`` parameter to
    ``token`` in 3.2. Pinning the new name would break older installs, so the
    legacy name is tried first and only then the current one.

    Args:
        token: Hugging Face access token.

    Returns:
        An initialised ``whisperx.diarize.DiarizationPipeline``.

    Raises:
        TypeError: Neither keyword name was accepted, which means the
            installed WhisperX exposes an unrecognised constructor.
    """
    try:
        return DiarizationPipeline(use_auth_token=token, device=DEVICE)
    except TypeError:
        return DiarizationPipeline(token=token, device=DEVICE)


def _format_turns(segments: list[dict[str, Any]]) -> str:
    """Group same-speaker segments into timestamped turns.

    Pyannote emits a speaker label per *segment*, not per sentence, so one
    spoken sentence routinely arrives split across three or four adjacent
    segments. Those are merged so the transcript reads as dialogue instead of
    a stream of fragments.

    A new turn begins when **either** condition holds:

    1. the speaker label changes — so the chronological order of the
       conversation is never rewritten, and a returning speaker resumes as a
       distinct turn; or
    2. the silence since the previous segment exceeds
       :data:`MAX_TURN_GAP_SECONDS`.

    Rule 2 is what keeps prosody intact. A sub-second hesitation is an
    artefact of segmentation and must not chop a sentence in half, but a
    lecturer pausing to think is meaningful structure that Stage 3 detects
    from the line breaks in this file. Folding that pause into the preceding
    turn would erase the signal *and* misreport the turn's duration.

    Otherwise the segments merge and the turn spans from the first segment's
    start to the last segment's end, sub-threshold gaps included.

    Blank segments are skipped before grouping, so an empty segment cannot
    act as a spurious turn boundary.

    Args:
        segments: Aligned, diarized segments (``start``, ``end``, ``text``,
            and optionally ``speaker``).

    Returns:
        The formatted transcript, one turn per line, as
        ``[SPEAKER_00] (00:00 - 00:15): The text spoken...``.
    """
    # Each accumulated turn: speaker label, first start, latest end, and the
    # text fragments to be joined. A plain loop is used rather than
    # ``itertools.groupby`` because the gap rule needs the previous segment's
    # end time, which a key function never sees.
    turns: list[dict[str, Any]] = []

    for segment in segments:
        text = (segment.get("text") or "").strip()
        if not text:
            continue  # never a boundary: a blank cannot end or start a turn

        speaker = segment.get("speaker") or UNKNOWN_SPEAKER
        start = float(segment.get("start") or 0.0)
        end = float(segment.get("end", start))

        if turns:
            previous = turns[-1]
            same_speaker = previous["speaker"] == speaker
            gap = start - previous["end"]
            # Overlapping segments give a negative gap, which correctly merges.
            if same_speaker and gap <= MAX_TURN_GAP_SECONDS:
                previous["end"] = max(previous["end"], end)
                previous["text"].append(text)
                continue

        turns.append(
            {"speaker": speaker, "start": start, "end": end, "text": [text]}
        )

    lines: list[str] = []
    for turn in turns:
        # Collapse the doubled spaces left by joining pre-stripped fragments.
        text = " ".join(" ".join(turn["text"]).split())
        if not text:
            continue
        start = format_timestamp(turn["start"])
        end = format_timestamp(turn["end"])
        lines.append(f"[{turn['speaker']}] ({start} - {end}): {text}")
    return "\n".join(lines)


class LocalTranscriber:
    """Transcribes lecture WAV files locally with WhisperX + diarization."""

    def __init__(
        self,
        hf_token: Optional[str] = None,
        compute_type: Optional[str] = None,
        batch_size: Optional[int] = None,
    ) -> None:
        """Validate configuration up front so failures surface before recording.

        Every tunable resolves with the same three-step precedence — explicit
        argument, then environment variable, then the module default — so a
        caller can override without mutating ``os.environ``, and an operator can
        override without editing code.

        Args:
            hf_token: Hugging Face token override. Defaults to ``HF_TOKEN``.
            compute_type: CTranslate2 quantization override. Defaults to
                ``WHISPER_COMPUTE_TYPE``, then :data:`COMPUTE_TYPE`.
            batch_size: Whisper decode batch size override. Defaults to
                ``WHISPER_BATCH_SIZE``, then :data:`BATCH_SIZE`.

        Raises:
            ValueError: No token was supplied and ``HF_TOKEN`` is unset, or an
                override failed validation in its resolver.
        """
        self.hf_token = hf_token or resolve_hf_token()
        self.compute_type = compute_type or resolve_compute_type()
        self.batch_size = batch_size or resolve_batch_size()

    # -- public entry point ------------------------------------------------

    def transcribe_audio(
        self, wav_path: Union[str, Path], language: str = DEFAULT_LANGUAGE
    ) -> Path:
        """Transcribe and diarize a WAV, writing ``raw_transcript.txt`` beside it.

        Args:
            wav_path: Path to the lecture WAV.
            language: ISO-639-1 code of the spoken language. Supplied to both
                the ASR and the forced-alignment stage — see the note in the
                body on why this is pinned rather than auto-detected.

        Returns:
            Path to the saved ``raw_transcript.txt``.

        Raises:
            FileNotFoundError: ``wav_path`` does not exist.
        """
        audio_path = Path(wav_path)
        if not audio_path.is_file():
            raise FileNotFoundError(f"audio file not found: {audio_path}")

        # One decode, shared by all three stages. load_audio resamples to the
        # 16 kHz mono float32 array every WhisperX stage expects, which is
        # exactly the format recorder.py writes.
        audio = whisperx.load_audio(str(audio_path))
        duration = len(audio) / SAMPLE_RATE
        print(
            f"  -> transcribing {audio_path.name} locally "
            f"({duration:.1f}s of 16 kHz mono, lang={language}, "
            f"compute_type={self.compute_type}, batch_size={self.batch_size})"
        )

        # -- Stage A: transcribe ------------------------------------------
        # The Swahili/English priming prompt was removed (prompt lobotomy).
        # Passing an initial_prompt to Whisper biases the decoder toward
        # fluent, confident hallucinations — the very thing we are trying to
        # kill. The engine now relies solely on the acoustic signal and the
        # tuned VAD/log-prob thresholds to suppress invention.
        #
        # ``condition_on_previous_text=False`` is the single most effective
        # guard against Whisper's degenerate repetition loop: the decoder is
        # no longer primed with its own previous output, so a garbled or
        # music-only window cannot snowball into "thank you. thank you.
        # thank you." across the whole lecture. ``no_speech_threshold=0.5``
        # (default 0.6) additionally biases the model toward calling
        # ambiguous stretches silence rather than inventing speech, which is
        # what broadcast/weather audio triggers most often.
        #
        # ``log_prob_threshold`` is raised from the -1.0 default. A segment the
        # decoder is very unsure about is now dropped instead of being emitted
        # as fluent, wrong text — which is what turns one bad 30 s window into
        # a paragraph of confident nonsense. NOTE the underscore: this is the
        # field name on faster_whisper's ``TranscriptionOptions``, which
        # ``load_model`` splats this dict into. The un-underscored
        # ``logprob_threshold`` is not a field and raises TypeError at load.
        asr_options = {
            "condition_on_previous_text": False,
            "no_speech_threshold": 0.5,
            "log_prob_threshold": -0.8,
        }

        # Aggressive VAD. The defaults (onset 0.500 / offset 0.363) are tuned
        # for clean dictation and keep low-energy speech that opens and closes
        # softly. Room-mic plus laptop-loopback lecture audio has far more
        # borderline energy, so the default windowing fragments words across
        # chunk boundaries. Raising both values makes the detector commit to
        # speech earlier and hold the tail longer, giving the decoder fewer,
        # longer, more coherent chunks.
        vad_options = {"vad_onset": 0.8, "vad_offset": 0.6}

        # ``compute_type`` is threaded from the constructor (env-overridable)
        # rather than read from the module constant here, so a caller can pick
        # a quantization per run. NOTE ``load_model`` still takes no
        # ``language=``: a preset there would build the tokenizer eagerly and
        # pin the decode, whereas the per-call value below is the only place
        # that needs to know the language.
        model = whisperx.load_model(
            WHISPER_MODEL,
            device=DEVICE,
            compute_type=self.compute_type,
            asr_options=asr_options,
            vad_options=vad_options,
        )
        # ``language`` is passed explicitly to skip ``detect_language()``, a
        # whole extra model pass over the opening 30 s that dominated startup on
        # CPU. It must match the lecture: a wrong code does not fail loudly, it
        # silently returns fluent text in the wrong language.
        #
        # ``batch_size`` is the other half of the speed-up. faster_whisper
        # decodes one window at a time by default, and a 90-minute lecture is
        # thousands of them.
        result = model.transcribe(
            audio, language=language, batch_size=self.batch_size
        )
        segments: list[dict[str, Any]] = result["segments"]
        print(f"  -> transcribed {len(segments)} segments; aligning words...")

        # -- Stage B: forced alignment (word-level timestamps) -------------
        # Alignment models are language-specific, so this selects the right
        # wav2vec2 pipeline. It reads the caller's ``language`` rather than
        # ``result["language"]``: WhisperX echoes the *requested* code back
        # there, so trusting it would launder a wrong guess back in as if it
        # had been detected.
        align_model, metadata = whisperx.load_align_model(
            language_code=language, device=DEVICE
        )
        result = whisperx.align(
            segments, align_model, metadata, audio, DEVICE,
            return_char_alignments=False,
        )
        segments = result["segments"]

        # -- Stage C: speaker diarization ----------------------------------
        # The speaker-count bounds are arguments to the pyannote *invocation*,
        # not to assign_word_speakers (whose signature is
        # ``(diarize_df, transcript_result, speaker_embeddings, fill_nearest)``).
        # min_speakers=1 permits a mono-speaker recording, while max_speakers=10
        # stops a noisy short interaction from shattering into a dozen
        # one-line "speakers".
        print("  -> assigning speakers (this downloads pyannote on first run)...")
        diarize_pipeline = _build_diarization_pipeline(self.hf_token)
        diarize_segments = diarize_pipeline(
            audio, min_speakers=1, max_speakers=10
        )
        result = assign_word_speakers(diarize_segments, result)
        segments = result["segments"]

        transcript = _format_turns(segments)

        transcript_path = audio_path.parent / TRANSCRIPT_FILENAME
        transcript_path.write_text(transcript, encoding="utf-8")
        print(f"Transcript saved to {transcript_path}")

        return transcript_path
