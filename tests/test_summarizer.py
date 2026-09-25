"""Headless unit tests for ``summarizer.py`` (Phase 3) — two-stage MapReduce.

The whole suite runs **offline** — no NVIDIA API calls, no network I/O:
* the ``openai`` client is always mocked out via ``mocker``,
* every file written by the code under test lands in ``tmp_path``,
* tenacity backoff is neutralised so tests run in milliseconds.
"""

from __future__ import annotations

import pytest
from PIL import Image

from summarizer import (
    LectureSynthesizer,
    MANIFEST_FILENAME,
    MODEL_NAME,
    SLIDE_EXTRACT_PROMPT,
    SYNTHESIS_INSTRUCTION,
)

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
    assert call.kwargs["model"] == MODEL_NAME
    assert call.kwargs["max_tokens"] == 800

    messages = call.kwargs["messages"]
    assert len(messages) == 1  # no system prompt for the map stage
    content = messages[0]["content"]
    assert len(content) == 2  # exactly ONE image per request

    text_part, image_part = content
    assert text_part["type"] == "text"
    assert SLIDE_EXTRACT_PROMPT == text_part["text"]
    assert "statutory references" in text_part["text"]

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
    assert reduce_call.kwargs["model"] == MODEL_NAME
    assert reduce_call.kwargs["messages"][0]["role"] == "system"
    assert "Law and Criminology" in reduce_call.kwargs["messages"][0]["content"]

    user_content = reduce_call.kwargs["messages"][1]["content"]
    assert isinstance(user_content, str)
    assert SYNTHESIS_INSTRUCTION in user_content
    assert transcript_text in user_content
    assert "### Slide: slide_001.jpg" in user_content
    assert "Content of slide two" in user_content