"""Headless unit tests for ``summarizer.py`` (Phase 3).

The whole suite runs **offline** — no NVIDIA API calls, no network I/O:
* the ``openai`` client is always mocked out via ``mocker``,
* every file written by the code under test lands in ``tmp_path``.
"""

from __future__ import annotations

import pytest
from openai import RateLimitError

from summarizer import LectureSynthesizer

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


class _FakeResponse:
    """Duck-typed stand-in for ``httpx.Response`` used by openai errors."""

    status_code = 429
    headers: dict = {}
    request = None
    text = "rate limited"
    reason_phrase = "Too Many Requests"
    http_version = "HTTP/1.1"
    url = "https://integrate.api.nvidia.com/v1/chat/completions"


def _fake_rate_limit_error() -> RateLimitError:
    """A real :class:`RateLimitError` constructed without any HTTP I/O."""
    return RateLimitError(
        "NVIDIA rate limit exceeded", response=_FakeResponse(), body=None
    )


def _make_synthesizer(mocker) -> LectureSynthesizer:
    """A synthesizer whose OpenAI client is replaced by a mock."""
    synthesizer = LectureSynthesizer(api_key="test-key")
    client = mocker.MagicMock()
    synthesizer.client = client
    return synthesizer


# ---------------------------------------------------------------------------
# null-transcript guard
# ---------------------------------------------------------------------------


def test_null_transcript_guard(tmp_path) -> None:
    """Transcripts under 50 words raise ValueError to prevent hallucination."""
    transcript_path = tmp_path / "short_transcript.txt"
    transcript_path.write_text("This is a very short transcript only ten words long.")

    synthesizer = LectureSynthesizer(api_key="test-key")

    with pytest.raises(ValueError, match="Transcript too short"):
        synthesizer.generate_notes(transcript_path, tmp_path / "output.md")


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
# end-to-end success path (LLM fully mocked)
# ---------------------------------------------------------------------------


def test_generate_notes_success(mocker, tmp_path) -> None:
    """Valid transcript produces cleaned markdown output file and returns True."""
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

    synthesizer = _make_synthesizer(mocker)

    # Mock LLM response with markdown fences
    mock_response = mocker.MagicMock()
    mock_response.choices = [
        mocker.MagicMock(
            message=mocker.MagicMock(
                content="```markdown\n# Criminal Law Lecture\n\n## Core Concepts\n- Actus Reus\n- Mens Rea\n```"
            )
        )
    ]
    synthesizer.client.chat.completions.create.return_value = mock_response

    # Neutralize retry backoff for instant test execution
    mocker.patch.object(
        LectureSynthesizer._call_llm.retry, "sleep", lambda seconds: None
    )

    output_path = tmp_path / "study_notes.md"
    result = synthesizer.generate_notes(transcript_path, output_path)

    assert result is True
    assert output_path.exists()

    # Verify the fences were stripped
    content = output_path.read_text(encoding="utf-8")
    assert content == "# Criminal Law Lecture\n\n## Core Concepts\n- Actus Reus\n- Mens Rea"

    # Verify the LLM was called with correct parameters
    call_kwargs = synthesizer.client.chat.completions.create.call_args.kwargs
    assert call_kwargs["model"] == "meta/llama-3.1-70b-instruct"
    assert call_kwargs["temperature"] == 0.2
    assert call_kwargs["max_tokens"] == 4000
    assert len(call_kwargs["messages"]) == 2
    assert call_kwargs["messages"][0]["role"] == "system"
    assert "Law and Criminology" in call_kwargs["messages"][0]["content"]
    assert call_kwargs["messages"][1]["role"] == "user"
    assert call_kwargs["messages"][1]["content"] == transcript_text