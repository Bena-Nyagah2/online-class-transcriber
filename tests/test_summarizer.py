"""Headless unit tests for ``summarizer.py`` (Phase 3) — two-stage MapReduce.

The whole suite runs **offline** — no NVIDIA API calls, no network I/O:
* the ``openai`` client is always mocked out via ``mocker``,
* every file written by the code under test lands in ``tmp_path``,
* tenacity backoff is neutralised so tests run in milliseconds.
"""

from __future__ import annotations

import importlib
import json
import os
from unittest import mock

import pytest
from openai.types.chat import ChatCompletion
from PIL import Image

import summarizer
from summarizer import (
    LectureSynthesizer,
    MANIFEST_FILENAME,
    MAP_MODEL,
    MODEL_NAME,
    REDUCE_MODEL,
    SLIDE_EXTRACT_PROMPT,
    SYNTHESIS_INSTRUCTION,
    SYSTEM_PROMPT,
)
from transcriber import format_timestamp
from vision import SLIDE_TIMES_FILENAME

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_synthesizer(mocker) -> LectureSynthesizer:
    """A synthesizer whose OpenAI client is replaced by a mock."""
    synthesizer = LectureSynthesizer(api_key="test-key")
    synthesizer.client = mocker.MagicMock()
    return synthesizer


def _make_response(mocker, content: str, finish_reason: str = "stop"):
    """A duck-typed chat completion response returning ``content``.

    ``finish_reason`` defaults to ``"stop"``, the success case. It is named
    explicitly because a bare ``MagicMock`` attribute never compares equal to
    ``"length"``: a helper that left it unset would make Stage 3's truncation
    guard untriggerable in every test, and the guard would look covered while
    being dead code.
    """
    response = mocker.MagicMock()
    response.choices = [
        mocker.MagicMock(
            message=mocker.MagicMock(content=content),
            finish_reason=finish_reason,
        )
    ]
    return response


def _write_slides(tmp_path, count: int):
    """Write ``count`` tiny real JPEGs ``slide_001.jpg``... in ``tmp_path/slides``."""
    slides_dir = tmp_path / "slides"
    slides_dir.mkdir(exist_ok=True)
    for index in range(1, count + 1):
        Image.new("RGB", (16, 16), color=(200, index * 40, 100)).save(
            slides_dir / f"slide_{index:03d}.jpg", "JPEG"
        )
    return slides_dir


def _neutralise_backoff(mocker) -> None:
    """Make tenacity retries instant (no real sleeps) for both stages."""
    mocker.patch.object(
        LectureSynthesizer._extract_slide_data.retry, "sleep", lambda seconds: None
    )
    mocker.patch.object(
        LectureSynthesizer._call_synthesis.retry, "sleep", lambda seconds: None
    )


@pytest.fixture
def fresh_summarizer():
    """Yield a reloader that re-imports ``summarizer`` under a chosen environment.

    ``MAP_MODEL``/``REDUCE_MODEL`` are read from ``os.environ`` at *import*
    time, so their fallback behaviour cannot be tested by patching the
    already-imported module — only by importing it again. Two hazards have to
    be handled for that to be meaningful:

    * ``load_dotenv()`` runs on import and would re-inject this project's real
      ``.env`` (which sets ``MAP_MODEL``/``REDUCE_MODEL`` for live runs), so it
      is neutralised and the environment is fully caller-controlled. The patch
      is applied to ``dotenv.load_dotenv`` rather than to the attribute on
      ``summarizer`` because reloading re-executes ``from dotenv import
      load_dotenv``, which would rebind over a module-level patch;
    * reloading mutates the shared ``summarizer`` module object in place, so the
      module is reloaded once more during teardown to leave it as the rest of
      the suite expects to find it.

    Args:
        **env: Environment overrides. A value of ``None`` means *unset this
            variable*; any other value is set.

    Returns:
        The reloaded module.
    """
    missing = object()

    def _load(**env):
        saved = {key: os.environ.get(key, missing) for key in env}
        for key, value in env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            with mock.patch("dotenv.load_dotenv"):
                return importlib.reload(summarizer)
        finally:
            for key, value in saved.items():
                if value is missing:
                    os.environ.pop(key, None)
                else:
                    os.environ[key] = value

    yield _load

    with mock.patch("dotenv.load_dotenv"):
        importlib.reload(summarizer)


# ---------------------------------------------------------------------------
# null-transcript guard
# ---------------------------------------------------------------------------


def test_null_transcript_guard(tmp_path) -> None:
    """Transcripts under 50 words raise ValueError to prevent hallucination."""
    transcript_path = tmp_path / "short_transcript.txt"
    transcript_path.write_text("This is a very short transcript only ten words long.")

    synthesizer = LectureSynthesizer(api_key="test-key")
    slides_dir = tmp_path / "slides"  # may be missing/empty — guard fires first

    with pytest.raises(ValueError, match="Transcript too short"):
        synthesizer.generate_notes(
            transcript_path, tmp_path / "output.md", slides_dir
        )


# ---------------------------------------------------------------------------
# markdown cleaner
# ---------------------------------------------------------------------------


def test_markdown_cleaner() -> None:
    """The helper strips ```markdown ... ``` fences and bare ``` fences."""
    synthesizer = LectureSynthesizer(api_key="test-key")

    # Case 1: with "markdown" specifier
    fenced_md = "```markdown\n# Title\n\nContent here.\n```"
    cleaned = synthesizer._clean_markdown(fenced_md)
    assert cleaned == "# Title\n\nContent here."

    # Case 2: bare fences without language specifier
    bare_fenced = "```\n# Title\n\nContent.\n```"
    cleaned_bare = synthesizer._clean_markdown(bare_fenced)
    assert cleaned_bare == "# Title\n\nContent."

    # Case 3: no fences (already clean)
    clean_md = "# Title\n\nContent."
    assert synthesizer._clean_markdown(clean_md) == clean_md

    # Case 4: fences with surrounding whitespace
    ws_fenced = "   ```markdown\n# Title\n\nContent.\n```   "
    cleaned_ws = synthesizer._clean_markdown(ws_fenced)
    assert cleaned_ws == "# Title\n\nContent."


# ---------------------------------------------------------------------------
# Stage 1 — one-image vision extraction
# ---------------------------------------------------------------------------


def test_extract_slide_data_sends_single_image(mocker, tmp_path) -> None:
    """One slide -> exactly one image in the payload, capped at 800 tokens."""
    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.return_value = _make_response(
        mocker, "## Statutory\n- Theft Act 1968"
    )
    _neutralise_backoff(mocker)

    slide_path = _write_slides(tmp_path, 1) / "slide_001.jpg"
    result = synthesizer._extract_slide_data(slide_path)

    call = synthesizer.client.chat.completions.create.call_args
    assert call.kwargs["model"] == MAP_MODEL
    assert call.kwargs["max_tokens"] == 800

    messages = call.kwargs["messages"]
    assert len(messages) == 1  # no system prompt for the map stage
    content = messages[0]["content"]
    assert len(content) == 2  # exactly ONE image per request

    text_part, image_part = content
    assert text_part["type"] == "text"
    assert SLIDE_EXTRACT_PROMPT == text_part["text"]
    assert "statutory/case references" in text_part["text"]
    # The map stage must instruct the model to narrate visual elements.
    assert "infographics" in text_part["text"]

    assert image_part["type"] == "image_url"
    url = image_part["image_url"]["url"]
    assert url.startswith("data:image/jpeg;base64,")
    payload = url.split(",", 1)[1]
    assert payload  # real base64 of the on-disk JPEG

    assert result == "## Statutory\n- Theft Act 1968"  # fence-free extract


def test_stage_one_raises_rather_than_storing_a_truncated_extract(
    mocker, tmp_path
) -> None:
    """A slide that exhausted the 800-token budget is refused, not stored.

    The manifest is Stage 3's ground truth. A partial extract written into it
    does not announce itself as partial — Stage 3 reads "the extract stopped
    here" as "the slide ended here" and reasons confidently from the gap. The
    message must name the offending slide, because in a 90-slide deck the
    filename is the only thing that tells an operator where to look.
    """
    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.return_value = _make_response(
        mocker, "## Framework\n- Mens Rea\n- Actus Reus\n- Dolus Eventualis and then",
        finish_reason="length",
    )
    _neutralise_backoff(mocker)

    slide_path = _write_slides(tmp_path, 1) / "slide_001.jpg"

    with pytest.raises(RuntimeError) as excinfo:
        synthesizer._extract_slide_data(slide_path)

    message = str(excinfo.value)
    assert "Stage 1 hit the token cap" in message
    assert "slide_001.jpg" in message
    assert "would be truncated" in message


def test_stage_one_guard_does_not_disturb_the_request_budget(mocker, tmp_path) -> None:
    """The guard is a pure check: the outgoing request is untouched.

    Guards the requirement that adding the failsafe must not quietly relax the
    budget to "stop the crash" or inherit Stage 3's 4096. The same 800/0.2 pair
    has to go out on the request that trips the guard as on the one that
    doesn't.
    """
    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.return_value = _make_response(
        mocker, "cut off", finish_reason="length",
    )
    _neutralise_backoff(mocker)

    slide_path = _write_slides(tmp_path, 1) / "slide_001.jpg"
    with pytest.raises(RuntimeError):
        synthesizer._extract_slide_data(slide_path)

    call = synthesizer.client.chat.completions.create.call_args
    assert call.kwargs["max_tokens"] == 800
    assert call.kwargs["temperature"] == 0.2
    assert call.kwargs["model"] == MAP_MODEL
    # Still exactly one image, and no system prompt added for the map stage.
    assert len(call.kwargs["messages"]) == 1
    assert len(call.kwargs["messages"][0]["content"]) == 2


def test_stage_one_truncation_is_not_retried(mocker, tmp_path) -> None:
    """One truncated slide costs one call, not three.

    Vision calls are the most expensive thing in the pipeline — 90 of them on a
    full lecture. Retrying a deterministic cap hit triples the bill to produce
    the identical failure.
    """
    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.return_value = _make_response(
        mocker, "cut off", finish_reason="length",
    )
    _neutralise_backoff(mocker)

    slide_path = _write_slides(tmp_path, 1) / "slide_001.jpg"
    with pytest.raises(RuntimeError):
        synthesizer._extract_slide_data(slide_path)

    assert synthesizer.client.chat.completions.create.call_count == 1


def test_stage_one_guard_holds_against_the_real_openai_response_shape(
    tmp_path,
) -> None:
    """Stage 1's guard is bound to the real SDK type, not just to a MagicMock.

    Same reasoning as the Stage 3 equivalent. Every other test in this file
    fabricates responses, and ``MagicMock`` invents any attribute — so a suite
    built only on mocks stays green even if the SDK moves ``finish_reason``,
    while production quietly stops guarding.
    """
    chat_completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": MAP_MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": "## Framework\n- Mens Rea and then it stops",
                    },
                }
            ],
        }
    )

    synthesizer = LectureSynthesizer(api_key="test-key")
    synthesizer.client = mock.MagicMock()
    synthesizer.client.chat.completions.create.return_value = chat_completion

    slide_path = _write_slides(tmp_path, 1) / "slide_001.jpg"
    with pytest.raises(RuntimeError) as excinfo:
        synthesizer._extract_slide_data(slide_path)

    assert "slide_001.jpg" in str(excinfo.value)


def test_truncated_stage_one_preserves_the_slides_already_extracted(
    mocker, tmp_path
) -> None:
    """A mid-deck crash keeps every completed slide on disk and resumable.

    This is what makes raising the right call at Stage 1. ``_build_slide_manifest``
    flushes after each slide, so aborting on slide 2 of 3 must leave slide 1
    written and must not write a partial entry for the slide that failed. A
    re-run then resumes from slide 2 for the price of one vision call, not a
    full re-extraction of the deck.
    """
    slides_dir = _write_slides(tmp_path, 3)
    session_dir = tmp_path / "session"
    session_dir.mkdir()

    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "Extract of slide one"),
        _make_response(mocker, "Extract of slide two, truncated", finish_reason="length"),
    ]
    _neutralise_backoff(mocker)

    with pytest.raises(RuntimeError):
        synthesizer._build_slide_manifest(slides_dir, session_dir)

    # The completed slide survives; the failed one left no trace.
    manifest = (session_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert "slide_001.jpg" in manifest
    assert "Extract of slide one" in manifest
    assert "slide_002.jpg" not in manifest
    assert "truncated" not in manifest
    # Only two calls were spent: one good extract, one that failed.
    assert synthesizer.client.chat.completions.create.call_count == 2

    # And a re-run picks up exactly where it stopped. Only the two missing
    # slides are re-extracted — slide_001 is skipped via the resume cache, so
    # the delta is 2, not 3.
    calls_before_rerun = synthesizer.client.chat.completions.create.call_count
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "Extract of slide two"),
        _make_response(mocker, "Extract of slide three"),
    ]
    synthesizer._build_slide_manifest(slides_dir, session_dir)

    manifest = (session_dir / MANIFEST_FILENAME).read_text(encoding="utf-8")
    assert "Extract of slide one" in manifest
    assert "Extract of slide two" in manifest
    assert "Extract of slide three" in manifest
    # slide_001 was not re-extracted — only the two that were missing.
    rerun_calls = (
        synthesizer.client.chat.completions.create.call_count - calls_before_rerun
    )
    assert rerun_calls == 2


# ---------------------------------------------------------------------------
# Stage 2 — chronological manifest with resume
# ---------------------------------------------------------------------------


def test_build_slide_manifest_extracts_new_and_resumes(mocker, tmp_path) -> None:
    """New slides are extracted; already-present filenames skip the API."""
    synthesizer = _make_synthesizer(mocker)
    _neutralise_backoff(mocker)

    slides_dir = _write_slides(tmp_path, 2)
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "Content of slide one"),
        _make_response(mocker, "Content of slide two"),
    ]

    manifest = synthesizer._build_slide_manifest(slides_dir, tmp_path)
    assert manifest == (
        "### Slide: slide_001.jpg\nContent of slide one\n\n"
        "### Slide: slide_002.jpg\nContent of slide two"
    )
    assert synthesizer.client.chat.completions.create.call_count == 2

    # Resume: a brand-new slide_003 appears; only it triggers a 3rd API call.
    _write_slides(tmp_path, 3)
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "Content of slide three")
    ]

    resumed = synthesizer._build_slide_manifest(slides_dir, tmp_path)
    assert synthesizer.client.chat.completions.create.call_count == 3  # one new
    assert "### Slide: slide_003.jpg\nContent of slide three" in resumed
    assert "slide_001.jpg" in resumed and "slide_002.jpg" in resumed

    # The manifest file lives in the session root next to the transcript.
    manifest_path = tmp_path / MANIFEST_FILENAME
    assert manifest_path.exists()
    assert "slide_003.jpg" in manifest_path.read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# Stage 2 — time-domain injection into the manifest
# ---------------------------------------------------------------------------


def _write_slide_times(slides_dir, mapping) -> None:
    """Write the ``slide_times.json`` sidecar the vision catcher produces."""
    (slides_dir / SLIDE_TIMES_FILENAME).write_text(
        json.dumps(mapping), encoding="utf-8"
    )


def test_manifest_headers_carry_capture_timestamps(mocker, tmp_path) -> None:
    """Each slide header is prefixed with the MM:SS offset it was captured at.

    Stage 3's speaker-identification directive can only cross-reference a slide
    against a transcript turn if both carry comparable clocks, so the elapsed
    offset is written into the header rather than left implicit in the filename.
    """
    synthesizer = _make_synthesizer(mocker)
    _neutralise_backoff(mocker)

    slides_dir = _write_slides(tmp_path, 2)
    _write_slide_times(slides_dir, {"slide_001.jpg": 12.0, "slide_002.jpg": 104.0})
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "Content of slide one"),
        _make_response(mocker, "Content of slide two"),
    ]

    manifest = synthesizer._build_slide_manifest(slides_dir, tmp_path)

    assert manifest == (
        "### Slide: [00:12] slide_001.jpg\nContent of slide one\n\n"
        "### Slide: [01:44] slide_002.jpg\nContent of slide two"
    )


def test_manifest_timestamps_roll_over_into_hours(mocker, tmp_path) -> None:
    """Past the hour the header uses H:MM:SS, matching the transcript format.

    A 90-minute lecture would otherwise render "90:00" in the manifest while
    the transcript says "1:30:00" — two spellings of one instant, which the
    model has to reconcile by arithmetic instead of reading.
    """
    synthesizer = _make_synthesizer(mocker)
    _neutralise_backoff(mocker)

    slides_dir = _write_slides(tmp_path, 2)
    _write_slide_times(slides_dir, {"slide_001.jpg": 5999.0, "slide_002.jpg": 5430.0})
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "one"),
        _make_response(mocker, "two"),
    ]

    manifest = synthesizer._build_slide_manifest(slides_dir, tmp_path)

    assert "### Slide: [1:39:59] slide_001.jpg" in manifest
    assert "### Slide: [1:30:30] slide_002.jpg" in manifest


def test_manifest_resumes_across_timestamped_and_legacy_headers(mocker, tmp_path) -> None:
    """Both header spellings resume without re-paying for the API.

    Session directories captured before this feature have bare-filename
    headers. If the resume matcher only understood the new spelling, every
    existing session would re-run its whole deck on the next summarization.
    """
    synthesizer = _make_synthesizer(mocker)
    _neutralise_backoff(mocker)

    slides_dir = _write_slides(tmp_path, 3)
    # Written in the new spelling; the legacy spelling is covered separately
    # by the untouched ``test_build_slide_manifest_extracts_new_and_resumes``.
    (tmp_path / MANIFEST_FILENAME).write_text(
        "### Slide: [00:10] slide_001.jpg\nnew one", encoding="utf-8"
    )
    _write_slide_times(
        slides_dir,
        {"slide_001.jpg": 10.0, "slide_002.jpg": 20.0, "slide_003.jpg": 30.0},
    )
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "two"),
        _make_response(mocker, "three"),
    ]

    manifest = synthesizer._build_slide_manifest(slides_dir, tmp_path)

    # slide_001 was recognised through its "[00:10] " prefix and skipped.
    assert "new one" in manifest
    assert synthesizer.client.chat.completions.create.call_count == 2
    assert "### Slide: [00:20] slide_002.jpg\ntwo" in manifest
    assert "### Slide: [00:30] slide_003.jpg\nthree" in manifest


def test_legacy_untimestamped_manifest_still_resumes(mocker, tmp_path) -> None:
    """A pre-feature manifest is recognised, not re-billed.

    ``[MM:SS] `` is optional in the header matcher, so a session captured
    before timestamps existed resumes on its bare filenames. Without the
    optionality, every historical session would re-pay for its whole deck.
    """
    synthesizer = _make_synthesizer(mocker)
    _neutralise_backoff(mocker)

    slides_dir = _write_slides(tmp_path, 3)
    (tmp_path / MANIFEST_FILENAME).write_text(
        "### Slide: slide_001.jpg\nlegacy one\n\n"
        "### Slide: slide_002.jpg\nlegacy two",
        encoding="utf-8",
    )
    _write_slide_times(slides_dir, {"slide_003.jpg": 30.0})
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "three")
    ]

    manifest = synthesizer._build_slide_manifest(slides_dir, tmp_path)

    assert synthesizer.client.chat.completions.create.call_count == 1
    assert "legacy one" in manifest and "legacy two" in manifest
    # The new slide still gets stamped, even beside legacy neighbours.
    assert "### Slide: [00:30] slide_003.jpg\nthree" in manifest


@pytest.mark.parametrize(
    "sidecar_contents",
    [
        pytest.param(None, id="absent"),
        pytest.param("{not json", id="malformed"),
        pytest.param('["a", "b"]', id="wrong-shape"),
        pytest.param('{"slide_001.jpg": "soon"}', id="non-numeric"),
    ],
)
def test_unusable_sidecar_degrades_to_bare_headers(
    mocker, tmp_path, sidecar_contents
) -> None:
    """A missing or corrupt sidecar never fabricates a timestamp.

    The module advertises an offline re-run over an existing session
    directory, and those sessions predate the sidecar. Emitting "[00:00]" for
    an unknown capture time would be worse than omitting it: the model would
    read a confident, wrong instant and could anchor a speaker's name to the
    wrong part of the lecture.
    """
    synthesizer = _make_synthesizer(mocker)
    _neutralise_backoff(mocker)

    slides_dir = _write_slides(tmp_path, 1)
    if sidecar_contents is not None:
        (slides_dir / SLIDE_TIMES_FILENAME).write_text(sidecar_contents, encoding="utf-8")
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "content")
    ]

    manifest = synthesizer._build_slide_manifest(slides_dir, tmp_path)

    assert manifest == "### Slide: slide_001.jpg\ncontent"
    assert "[00:00]" not in manifest


def test_partial_sidecar_timestamps_only_the_slides_it_covers(
    mocker, tmp_path
) -> None:
    """A slide absent from the sidecar keeps a bare header; others are stamped."""
    synthesizer = _make_synthesizer(mocker)
    _neutralise_backoff(mocker)

    slides_dir = _write_slides(tmp_path, 2)
    _write_slide_times(slides_dir, {"slide_002.jpg": 65.0})  # slide_001 missing
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "one"),
        _make_response(mocker, "two"),
    ]

    manifest = synthesizer._build_slide_manifest(slides_dir, tmp_path)

    assert "### Slide: slide_001.jpg\none" in manifest
    assert "### Slide: [01:05] slide_002.jpg\ntwo" in manifest


def test_timestamps_reach_the_stage_three_prompt_on_the_transcript_clock(
    mocker, tmp_path
) -> None:
    """End-to-end: a slide offset arrives in the reduce prompt, matching the
    transcript's own clock for the same instant.

    This is the entire point of the feature. The Stage 3 speaker-identification
    directive instructs the model to cross-reference slide timestamps against
    transcript timestamps, which is only actionable if both are present *and*
    spelled the same way. The assertion pairs slide ``[01:44]`` with a
    transcript turn at 104.0s — the same moment — so a future change to either
    clock renderer that desynchronised the two would fail here.
    """
    # A turn header rendered by the transcriber at exactly 104.0 seconds.
    transcript_text = (
        f"[SPEAKER_01] ({format_timestamp(104.0)} - {format_timestamp(120.0)}): "
        "My name is Dr Amara Ochieng and I lead the forensics unit. "
    ) + " ".join(["filler"] * 60)

    transcript_path = tmp_path / "raw_transcript.txt"
    transcript_path.write_text(transcript_text, encoding="utf-8")

    slides_dir = _write_slides(tmp_path, 2)
    _write_slide_times(
        slides_dir, {"slide_001.jpg": 20.0, "slide_002.jpg": 104.0}
    )

    synthesizer = _make_synthesizer(mocker)
    _neutralise_backoff(mocker)
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "Intro title slide"),
        _make_response(mocker, "Lower third: DR AMARA OCHIENG - FORENSICS"),
        _make_response(mocker, "# Notes"),
    ]

    synthesizer.generate_notes(
        transcript_path, tmp_path / "notes.md", slides_dir
    )

    reduce_call = synthesizer.client.chat.completions.create.call_args_list[2]
    user_content = reduce_call.kwargs["messages"][1]["content"]

    # The lower-third slide is stamped at the same instant the speaker is
    # talking, so the model has everything the directive asks for.
    assert "### Slide: [01:44] slide_002.jpg" in user_content
    assert "DR AMARA OCHIENG" in user_content
    assert f"[SPEAKER_01] ({format_timestamp(104.0)} -" in user_content
    # Same spelling on both sides of the cross-reference.
    assert format_timestamp(104.0) in user_content


# ---------------------------------------------------------------------------
# Stage 3 — text-only final synthesis across the whole session
# ---------------------------------------------------------------------------


def test_generate_notes_synthesis_text_only(mocker, tmp_path) -> None:
    """generate_notes: map -> accumulate -> one text-only merge call."""
    transcript_path = tmp_path / "raw_transcript.txt"
    # 60 words > MIN_TRANSCRIPT_WORDS (50)
    transcript_text = (
        "This lecture covers the core concepts of criminal law including "
        "actus reus and mens rea requirements for various offences. "
        "The professor discussed the Theft Act 1968 and relevant case law "
        "such as R v Ghosh and R v Cunningham. Students should focus on "
        "the distinction between specific and basic intent crimes."
    )
    transcript_path.write_text(transcript_text, encoding="utf-8")

    slides_dir = _write_slides(tmp_path, 2)

    synthesizer = _make_synthesizer(mocker)
    _neutralise_backoff(mocker)
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "Content of slide one"),
        _make_response(mocker, "Content of slide two"),
        _make_response(
            mocker,
            "```markdown\n# Criminal Law Lecture\n\n## Core Concepts\n"
            "- Actus Reus\n- Mens Rea\n```",
        ),
    ]

    output_path = tmp_path / "study_notes.md"
    result = synthesizer.generate_notes(transcript_path, output_path, slides_dir)

    assert result is True
    assert output_path.exists()
    assert (
        output_path.read_text(encoding="utf-8")
        == "# Criminal Law Lecture\n\n## Core Concepts\n- Actus Reus\n- Mens Rea"
    )

    # 2 map calls + 1 reduce call.
    assert synthesizer.client.chat.completions.create.call_count == 3

    calls = synthesizer.client.chat.completions.create.call_args_list

    # Stage 1 calls: each carried exactly one image, capped at 800 tokens.
    for stage_one_call in calls[:2]:
        assert stage_one_call.kwargs["max_tokens"] == 800
        content = stage_one_call.kwargs["messages"][0]["content"]
        assert len(content) == 2
        assert content[1]["type"] == "image_url"

    # Stage 3 call: TEXT ONLY — a plain str, no image elements.
    reduce_call = calls[2]
    # 4096, not the old 4000: a reasoner model that leaks its scratchpad eats
    # the budget and the notes lose their final section to a length cut.
    assert reduce_call.kwargs["max_tokens"] == 4096
    assert reduce_call.kwargs["temperature"] == 0.2
    assert reduce_call.kwargs["model"] == REDUCE_MODEL
    assert reduce_call.kwargs["messages"][0]["role"] == "system"
    system_content = reduce_call.kwargs["messages"][0]["content"]
    assert system_content == SYSTEM_PROMPT
    # Zero-preamble guard. The directive must lead the prompt, before the
    # persona line, so it is the first thing the decoder is conditioned on.
    assert system_content.startswith("STRICT OUTPUT FORMAT DIRECTIVE:")
    assert (
        'Your response MUST begin immediately with the exact characters '
        '"## Executive Summary".' in system_content
    )
    # The new professor persona requires these structural elements:
    assert "## Executive Summary" in system_content
    assert "## Core Theories & Concepts" in system_content
    assert "## Statutes, Case Law, & References" in system_content
    assert "## Exam Preparation" in system_content
    # Anti-hallucination guard for broadcast/weather edge cases: no invented
    # URLs, code, or narrative filler; mismatched audio/slide pairs get
    # reported as two separate facts rather than reconciled into fiction.
    assert "STRICT GROUNDING DIRECTIVE" in system_content
    assert "NEVER invent, infer, or hallucinate" in system_content
    assert "report both facts neutrally" in system_content
    # Vision-audio fusion: slide OCR crossed with transcript timestamps so a
    # lower-third name replaces the raw [SPEAKER_XX] label in the notes.
    assert "SPEAKER IDENTIFICATION" in system_content
    assert "[SPEAKER_XX]" in system_content
    assert "lower-third graphic" in system_content
    # Three synthesis questions for exam preparation
    assert "exactly THREE" in system_content
    assert "final-year law school exam" in system_content

    user_content = reduce_call.kwargs["messages"][1]["content"]
    assert isinstance(user_content, str)
    assert SYNTHESIS_INSTRUCTION in user_content
    assert transcript_text in user_content
    assert "### Slide: slide_001.jpg" in user_content
    assert "Content of slide two" in user_content


# ---------------------------------------------------------------------------
# Stage 3 sampling budget
# ---------------------------------------------------------------------------


def test_synthesis_pins_token_budget_and_low_temperature(mocker) -> None:
    """``_call_synthesis`` sends max_tokens=4096 and temperature=0.2.

    Pinned directly against the call rather than through ``generate_notes``,
    because this is the one number that decides whether the notes survive at
    all: a model that spends the first few hundred tokens narrating its own
    planning is spending them out of the last section of the document. 4000 was
    not enough headroom; 4096 is.
    """
    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.return_value = _make_response(
        mocker, "## Executive Summary\nA lecture."
    )
    _neutralise_backoff(mocker)

    result = synthesizer._call_synthesis("transcript + manifest")

    assert result == "## Executive Summary\nA lecture."
    call = synthesizer.client.chat.completions.create.call_args
    assert call.kwargs["max_tokens"] == 4096
    assert call.kwargs["temperature"] == 0.2


def test_synthesis_token_budget_matches_the_module_constant(mocker) -> None:
    """The literal 4096 in the call must be the module's documented budget.

    Guards against the constant and the call site drifting apart, and records
    *why* the number is 4096 rather than a rounder, cheaper figure.
    """
    assert summarizer.MAX_TOKENS == 4096
    assert summarizer.TEMPERATURE == 0.2

    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.return_value = _make_response(
        mocker, "## Executive Summary"
    )
    _neutralise_backoff(mocker)

    synthesizer._call_synthesis("prompt")
    call = synthesizer.client.chat.completions.create.call_args
    assert call.kwargs["max_tokens"] == summarizer.MAX_TOKENS
    assert call.kwargs["temperature"] == summarizer.TEMPERATURE


def test_system_prompt_opens_with_the_zero_preamble_directive() -> None:
    """The no-scratchpad directive leads the prompt, ahead of the persona.

    Placement is the whole mechanism. A grounding rule buried at the end of a
    long prompt competes with everything above it for the decoder's attention;
    the first line is unconditionally the strongest conditioning the API
    offers, so the format ban goes there.
    """
    assert SYSTEM_PROMPT.startswith("STRICT OUTPUT FORMAT DIRECTIVE:")
    assert 'NEVER output "Here\'s a thinking process:"' in SYSTEM_PROMPT
    assert "Do NOT include preambles, drafting thoughts, scratchpads" in SYSTEM_PROMPT
    # It must precede the persona, and the grounding/fusion directives must
    # still be present after it.
    assert SYSTEM_PROMPT.index("Tenured University Professor") > 0
    assert "STRICT GROUNDING DIRECTIVE" in SYSTEM_PROMPT
    assert "SPEAKER IDENTIFICATION" in SYSTEM_PROMPT


def test_stage_one_keeps_its_own_small_budget(mocker, tmp_path) -> None:
    """Stage 1 is not swept up in the Stage 3 budget bump.

    A 90-slide lecture issues 90 vision calls. Lifting each from 800 to 4096
    tokens is a 5x cost increase on the stage that has no truncation problem
    -- a single slide never needs more than ~800 tokens of extract. The two
    budgets are deliberately different.
    """
    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.return_value = _make_response(
        mocker, "## Statutory\n- Theft Act 1968"
    )
    _neutralise_backoff(mocker)

    slide_path = _write_slides(tmp_path, 1) / "slide_001.jpg"
    synthesizer._extract_slide_data(slide_path)

    call = synthesizer.client.chat.completions.create.call_args
    assert call.kwargs["max_tokens"] == 800
    assert call.kwargs["temperature"] == 0.2


def test_synthesis_raises_rather_than_returning_a_truncated_document(mocker) -> None:
    """A ``finish_reason`` of "length" is a hard failure, not a short answer.

    The whole point of the failsafe: the model burned its budget and stopped
    mid-document, so the notes are missing their last section. Handing those
    partial notes back lets them reach disk and be read as complete.
    """
    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.return_value = _make_response(
        mocker,
        "## Executive Summary\nThe lecture opened with mens rea, and then it just\n"
        "## Core Theories & Concepts\n- Actus Reus\n- Mens Rea\n- Dolus Eventualis\n- A",
        finish_reason="length",
    )
    _neutralise_backoff(mocker)

    with pytest.raises(RuntimeError) as excinfo:
        synthesizer._call_synthesis("transcript + manifest")

    assert str(excinfo.value) == (
        "Stage 3 hit the token cap — notes would be truncated."
    )


def test_truncation_is_not_retried(mocker) -> None:
    """The failsafe must not burn three more full-length calls on the same wall.

    Truncation is deterministic for a fixed prompt, so a retry can only
    reproduce it while tripling the spend. ``RuntimeError`` is deliberately
    outside ``_RETRYABLE_ERRORS`` so tenacity re-raises it immediately; this
    test is what stops someone "helpfully" adding it to that tuple later.
    """
    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.return_value = _make_response(
        mocker, "## Executive Summary\ncut off here",
        finish_reason="length",
    )
    _neutralise_backoff(mocker)

    with pytest.raises(RuntimeError):
        synthesizer._call_synthesis("transcript + manifest")

    assert synthesizer.client.chat.completions.create.call_count == 1
    assert RuntimeError not in summarizer._RETRYABLE_ERRORS


def test_synthesis_returns_content_on_a_clean_stop(mocker) -> None:
    """The happy path is untouched: a "stop" response flows through as before.

    Guards the guard — a failsafe wired to the wrong condition would turn every
    successful run into a crash.
    """
    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.return_value = _make_response(
        mocker,
        "## Executive Summary\nComplete.\n\n## Exam Preparation\n1. Question?",
        finish_reason="stop",
    )
    _neutralise_backoff(mocker)

    result = synthesizer._call_synthesis("transcript + manifest")

    assert result.startswith("## Executive Summary")
    assert "## Exam Preparation" in result


def test_truncated_synthesis_writes_no_notes_file(mocker, tmp_path) -> None:
    """End-to-end: a truncated Stage 3 leaves no study notes on disk.

    ``generate_notes`` writes its output only after ``_call_synthesis``
    returns, so the crash has to happen before that write. Asserting the file
    is *absent* is the difference between "the run failed loudly" and "the run
    failed but left a plausible-looking broken artifact behind".
    """
    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        " ".join(["word"] * (summarizer.MIN_TRANSCRIPT_WORDS + 10)),
        encoding="utf-8",
    )
    slides_dir = _write_slides(tmp_path, 1)
    output_path = tmp_path / "study_notes.md"

    synthesizer = _make_synthesizer(mocker)
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "Content of slide one"),
        _make_response(mocker, "## Executive Summary\nTrunc", finish_reason="length"),
    ]
    _neutralise_backoff(mocker)

    with pytest.raises(RuntimeError):
        synthesizer.generate_notes(transcript_path, output_path, slides_dir)

    assert not output_path.exists()


def test_truncation_guard_holds_against_the_real_openai_response_shape() -> None:
    """The guard is pinned to the real SDK type, not only to our own mock.

    Everything else in this module fabricates responses with ``MagicMock``, so
    a test suite built on mocks would happily keep passing if the SDK renamed or
    moved ``finish_reason`` -- the production guard would stop firing and the
    tests would never notice. This builds an actual ``ChatCompletion`` and drives
    the real method, so an SDK contract change breaks the build instead of
    silently disarming the failsafe.

    Note the field lives on the ``Choice`` submodel, not on ``ChatCompletion``
    itself, so the guard has to read ``choices[0].finish_reason``.
    """
    chat_completion = ChatCompletion.model_validate(
        {
            "id": "chatcmpl-test",
            "object": "chat.completion",
            "created": 0,
            "model": REDUCE_MODEL,
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "length",
                    "message": {
                        "role": "assistant",
                        "content": "## Executive Summary\nThe lecture opened and then",
                    },
                }
            ],
        }
    )

    synthesizer = LectureSynthesizer(api_key="test-key")
    synthesizer.client = mock.MagicMock()
    synthesizer.client.chat.completions.create.return_value = chat_completion

    with pytest.raises(RuntimeError) as excinfo:
        synthesizer._call_synthesis("transcript + manifest")

    assert str(excinfo.value) == (
        "Stage 3 hit the token cap — notes would be truncated."
    )


# ---------------------------------------------------------------------------
# per-stage model selection
# ---------------------------------------------------------------------------


def test_map_and_reduce_route_to_their_own_model_variables(
    mocker, tmp_path, monkeypatch
) -> None:
    """Each stage reads its own module-level model, not a shared constant.

    The point of splitting ``MAP_MODEL``/``REDUCE_MODEL`` is that either stage
    can be retuned in isolation — e.g. a vision-tuned 70B for slide OCR and a
    cheaper 8B for the long-context merge. If either stage silently reverted to
    ``MODEL_NAME`` the split would be cosmetic.
    """
    monkeypatch.setattr(summarizer, "MAP_MODEL", "meta/map-specialised")
    monkeypatch.setattr(summarizer, "REDUCE_MODEL", "meta/reduce-specialised")

    transcript_path = tmp_path / "transcript.txt"
    transcript_path.write_text(
        "[SPEAKER_00] (00:00 - 00:31): " + " ".join(["word"] * 80)
    )
    slides_dir = _write_slides(tmp_path, 1)

    synthesizer = _make_synthesizer(mocker)
    _neutralise_backoff(mocker)
    synthesizer.client.chat.completions.create.side_effect = [
        _make_response(mocker, "Content of slide one"),
        _make_response(mocker, "# Notes"),
    ]

    synthesizer.generate_notes(
        transcript_path, tmp_path / "notes.md", slides_dir
    )

    calls = synthesizer.client.chat.completions.create.call_args_list
    assert len(calls) == 2  # 1 map + 1 reduce
    assert calls[0].kwargs["model"] == "meta/map-specialised"
    assert calls[1].kwargs["model"] == "meta/reduce-specialised"
    # The shared constant must not leak into either call.
    assert MODEL_NAME not in {"meta/map-specialised", "meta/reduce-specialised"}


def test_stage_models_default_to_the_shared_model_name(fresh_summarizer) -> None:
    """With no per-stage override, both stages inherit ``MODEL_NAME``.

    ``MAP_MODEL``/``REDUCE_MODEL`` fall back to ``MODEL_NAME`` rather than
    hardcoding a model id, so a single ``LLM_MODEL_NAME`` export still moves
    the whole pipeline.

    Asserted against a freshly reloaded module under a controlled environment.
    This project's ``.env`` genuinely sets ``MAP_MODEL``/``REDUCE_MODEL`` for
    real runs, so asserting the already-imported values would only be testing
    the developer's machine — and did pass or fail depending on their config.
    """
    module = fresh_summarizer(
        LLM_MODEL_NAME="meta/known-model",
        MAP_MODEL=None,  # absent -> exercise the fallback
        REDUCE_MODEL=None,
    )

    assert module.MODEL_NAME == "meta/known-model"
    assert module.MAP_MODEL == "meta/known-model"
    assert module.REDUCE_MODEL == "meta/known-model"


def test_per_stage_env_overrides_beat_the_shared_model(fresh_summarizer) -> None:
    """A per-stage export wins over ``LLM_MODEL_NAME`` independently per stage.

    This is the case that actually bit us: a developer who pins only
    ``REDUCE_MODEL`` to a bigger model for the reasoning-heavy merge while
    leaving the cheap vision model on the map stage.
    """
    module = fresh_summarizer(
        LLM_MODEL_NAME="meta/shared",
        MAP_MODEL="meta/map-only",
        REDUCE_MODEL="meta/reduce-only",
    )

    assert module.MAP_MODEL == "meta/map-only"
    assert module.REDUCE_MODEL == "meta/reduce-only"
    assert module.MODEL_NAME == "meta/shared"  # untouched


def test_only_one_stage_pinned_leaves_the_other_on_the_fallback(
    fresh_summarizer,
) -> None:
    """Pinning REDUCE_MODEL alone must not drag MAP_MODEL along with it."""
    module = fresh_summarizer(
        LLM_MODEL_NAME="meta/shared",
        MAP_MODEL=None,
        REDUCE_MODEL="meta/reduce-only",
    )

    assert module.REDUCE_MODEL == "meta/reduce-only"
    assert module.MAP_MODEL == "meta/shared"  # inherited, not dragged along