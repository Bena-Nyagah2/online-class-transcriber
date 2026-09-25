"""Phase 3 — Multimodal lecture synthesis into structured study notes via NVIDIA NIM (Llama).

Reads the raw transcript **and** the deduplicated JPEG slides captured for
the lecture, then produces exam-focused Markdown notes through the
OpenAI-compatible NVIDIA endpoint. Slides are Base64-encoded into
``data:image/jpeg`` URLs in a multimodal chat payload so the model can fuse
visual content (case citations, diagrams, bullet points) with the
lecturer's spoken explanation.
"""

from __future__ import annotations

import base64
import os
import re
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

# Load NVIDIA_API_KEY from a local .env (if present) before any client reads
# the environment. Explicit shell exports still take precedence.
load_dotenv()

MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"

MIN_TRANSCRIPT_WORDS = 50
MAX_TOKENS = 4000
TEMPERATURE = 0.2

_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)

SYSTEM_PROMPT = (
    "You are an expert academic assistant for a university Law and Criminology "
    "student. Convert the following lecture transcript into highly structured, "
    "exam-focused Markdown notes. You must include these sections: "
    "Metadata (Topic/Date), Core Concepts & Definitions, "
    "Criminological/Legal Theories Discussed, Statutes & Case Law Cited, "
    "Lecturer Emphasis (Exam Hints), and Action Items. Do not invent facts."
)

# Engineered instruction fusing the visual and spoken channels. Placed as the
# leading text element of every multimodal user payload.
VISION_PROMPT = (
    "Integrate the provided slide images with the spoken transcript. "
    "Extract case citations, diagrams, and bullet points from the visual "
    "slides and place them in context with the lecturer's spoken explanation."
)


class LectureSynthesizer:
    """Synthesizes lecture transcripts + slides into structured Markdown notes."""

    def __init__(self, api_key: Optional[str] = None) -> None:
        """Prepare the OpenAI-compatible client for the NVIDIA endpoint.

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
        self.client = openai.OpenAI(
            api_key=self.api_key,
            base_url=NVIDIA_BASE_URL,
        )

    # -- image encoding ----------------------------------------------------

    def _encode_image(self, image_path: Path) -> str:
        """Base64-encode a JPEG slide into a ``data:`` URL for the vision model.

        Args:
            image_path: Path to a captured slide JPEG.

        Returns:
            ``data:image/jpeg;base64,...`` data URL.
        """
        with image_path.open("rb") as image_file:
            base64_str = base64.b64encode(image_file.read()).decode("utf-8")
        return f"data:image/jpeg;base64,{base64_str}"

    # -- markdown cleanup --------------------------------------------------

    @staticmethod
    def _clean_markdown(text: str) -> str:
        """Strip leading/trailing ```markdown and ``` fences if present.

        Args:
            text: Raw LLM output that may be wrapped in code fences.

        Returns:
            Clean Markdown without the outer fence block.
        """
        # Match optional leading whitespace, then ```markdown, then content,
        # then ``` with optional trailing whitespace.
        pattern = r"^\s*```markdown\s*\n(.*?)\n```\s*$"
        match = re.match(pattern, text, flags=re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        # Also handle bare ``` without "markdown" specifier
        pattern_bare = r"^\s*```\s*\n(.*?)\n```\s*$"
        match_bare = re.match(pattern_bare, text, flags=re.DOTALL)
        if match_bare:
            return match_bare.group(1).strip()
        return text.strip()

    # -- LLM call ----------------------------------------------------------

    @retry(
        wait=wait_exponential(multiplier=1, min=2, max=10),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        reraise=True,
    )
    def _call_llm(self, transcript_text: str, slide_paths: list[Path]) -> str:
        """Send the transcript + slide images to the multimodal LLM.

        Args:
            transcript_text: The lecture transcript text.
            slide_paths: Captured slide JPEGs, already sorted chronologically.

        Returns:
            Raw markdown response from the model.
        """
        # Multimodal payload: leading text element (engineered prompt +
        # transcript) followed by one image element per slide.
        content: list[dict] = [
            {"type": "text", "text": f"{VISION_PROMPT}\n\n{transcript_text}"}
        ]
        content.extend(
            {
                "type": "image_url",
                "image_url": {"url": self._encode_image(path)},
            }
            for path in slide_paths
        )

        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": content},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
        )
        return response.choices[0].message.content or ""

    # -- public entry point ------------------------------------------------

    def generate_notes(
        self, transcript_path: Path, notes_path: Path, slides_dir: Path
    ) -> bool:
        """Generate study notes from a transcript and its captured slides.

        Args:
            transcript_path: Path to the raw transcript text file.
            notes_path: Path where the cleaned Markdown notes will be written.
            slides_dir: Directory of deduplicated JPEG slides. Lexicographic
                ordering equals chronological order (``slide_001`` naming),
                so ``sorted()`` reproduces presentation order.

        Returns:
            True on success.

        Raises:
            ValueError: If transcript word count is below the minimum threshold.
        """
        transcript_text = transcript_path.read_text(encoding="utf-8")
        word_count = len(transcript_text.split())

        if word_count < MIN_TRANSCRIPT_WORDS:
            raise ValueError(
                "Transcript too short. Aborting synthesis to prevent hallucination."
            )

        # slide_001.jpg .. slide_0NN.jpg sort lexicographically; a missing or
        # empty slides dir yields an empty list and a text-only payload.
        slide_paths = sorted(slides_dir.glob("*.jpg"))

        raw_markdown = self._call_llm(transcript_text, slide_paths)
        cleaned_markdown = self._clean_markdown(raw_markdown)

        notes_path.write_text(cleaned_markdown, encoding="utf-8")
        return True