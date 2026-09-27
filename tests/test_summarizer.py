"""Headless unit tests for ``summarizer.py`` (Phase 3) — two-stage MapReduce.

The whole suite runs **offline** — no NVIDIA API calls, no network I/O:
* the ``openai`` client is always mocked out via ``mocker``,
* every file written by the code under test lands in ``tmp_path``,
* tenacity backoff is neutralised so tests run in milliseconds.
"""

from __future__ import annotations

import json

import pytest
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


def _make_response(mocker, content: str):
    """A duck-typed chat completion response returning ``content``."""
    response = mocker.MagicMock()
    response.choices = [
        mocker.MagicMock(message=mocker.MagicMock(content=content))
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
    assert reduce_call.kwargs["max_tokens"] == 4000
    assert reduce_call.kwargs["model"] == REDUCE_MODEL
    assert reduce_call.kwargs["messages"][0]["role"] == "system"
    system_content = reduce_call.kwargs["messages"][0]["content"]
    assert system_content == SYSTEM_PROMPT
    # The reduce stage must demand rich formatting and a references appendix.
    assert "Markdown Tables" in system_content
    assert "References, Statutes & Citations" in system_content
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

    user_content = reduce_call.kwargs["messages"][1]["content"]
    assert isinstance(user_content, str)
    assert SYNTHESIS_INSTRUCTION in user_content
    assert transcript_text in user_content
    assert "### Slide: slide_001.jpg" in user_content
    assert "Content of slide two" in user_content


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


def test_stage_models_default_to_the_shared_model_name() -> None:
    """With no per-stage override, both stages inherit ``MODEL_NAME``.

    ``MAP_MODEL``/``REDUCE_MODEL`` fall back to ``MODEL_NAME`` rather than
    hardcoding a model id, so a single ``LLM_MODEL_NAME`` export still moves
    the whole pipeline.
    """
    assert MAP_MODEL == MODEL_NAME
    assert REDUCE_MODEL == MODEL_NAME