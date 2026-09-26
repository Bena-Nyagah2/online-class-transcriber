"""Phase 3 — Two-stage MapReduce lecture synthesis via NVIDIA NIM (Llama).

NVIDIA's endpoint caps multimodal requests at **one image per call**, so the
old "all slides in one payload" approach cannot scale to a full lecture deck.
This module runs a MapReduce pipeline and also works **offline** on any
already-captured session directory::

    python summarizer.py output/OCHIENG-FRIDAY_2026-09-25

Stage 1 — *Map*  ``_extract_slide_data``: each JPEG slide is sent to the
vision model alone (instructed to act as an "expert academic visual analyst"
and to narrate any infographics, charts, or diagrams) and reduced to a
compact Markdown block. NVIDIA rate limits are absorbed by tenacity retries
(2-5s jitter).

Stage 2 — *Accumulate*  ``_build_slide_manifest``: the per-slide extracts are
combined chronologically into ``slides_manifest.txt`` next to the raw
transcript. The manifest doubles as a **resume cache** — any slide filename
already present is skipped on re-runs, so an interrupted extraction continues
where it stopped and completed sessions never re-pay for API calls.

Stage 3 — *Reduce*  ``generate_notes``: a single **text-only** call merges the
raw transcript with the accumulated slide text into one cohesive academic
study guide, written to the session's ``<session-name>.md`` file.
"""

from __future__ import annotations

import base64
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_random,
)

from transcriber import TRANSCRIPT_FILENAME

# Load NVIDIA_API_KEY from a local .env (if present) before any client reads
# the environment. Explicit shell exports still take precedence.
load_dotenv()

MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "meta/llama-3.1-8b-instruct")
NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"

MIN_TRANSCRIPT_WORDS = 50
MAX_TOKENS = 4000  # Stage 3: final synthesis over the full transcript + deck
TEMPERATURE = 0.2

MANIFEST_FILENAME = "slides_manifest.txt"

_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)

SYSTEM_PROMPT = textwrap.dedent(
    """\
    You are an elite, highly engaging academic study assistant. You are receiving a chronological lecture transcript and a manifest of described visual slides.

    Your mandate is to fuse these into a comprehensive, deeply analytical, and fun-to-read study guide.

    CRITICAL RULES:
    1. **Read the Tone:** Look for semantic emphasis. If the lecturer repeats a concept, says "this is important", or if there are structural pauses (indicated by line breaks), highlight these as 🚨 **Core Concepts**.
    2. **Visual Fusion:** When the transcript aligns with a described infographic or diagram from the slides, explain the visual relationship in context.
    3. **Rich Formatting:** DO NOT output boring walls of text. You MUST use:
       - **Markdown Tables** for any comparisons, pros/cons, or chronological events.
       - **Blockquotes (>)** for major definitions or lecturer quotes.
       - **Bold text** for key terminology.
    4. **References:** Create a dedicated "📚 References, Statutes & Citations" appendix at the bottom containing every case, law, book, or paper mentioned.

    Make the formatting highly scannable, intellectually stimulating, and academically rigorous.
    """
)

# Stage 1: sent alongside exactly one slide image.
SLIDE_EXTRACT_PROMPT = (
    "You are an expert academic visual analyst. Extract all text, bullet points, "
    "and statutory/case references. IF the image contains infographics, charts, "
    "diagrams, or visual metaphors, you MUST explicitly describe the data, "
    "structural relationships, and meaning of the visual elements. Be highly "
    "detailed but format concisely in Markdown."
)

# Stage 3: leading instruction that fuses the two channels.
SYNTHESIS_INSTRUCTION = (
    "Here is the chronological transcript of the lecture, followed by the "
    "text extracted from the slides. Integrate them into a single, cohesive "
    "academic study guide."
)


class LectureSynthesizer:
    """Two-stage MapReduce synthesizer: per-slide vision map, then text merge."""

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

    # -- Stage 1 (Map): one-image vision extraction -------------------------

    @retry(
        wait=wait_random(min=2, max=5),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        reraise=True,
    )
    def _extract_slide_data(self, image_path: Path) -> str:
        """Extract the content of a single slide via a one-image vision call.

        NVIDIA's endpoint accepts one image per request, so each slide is
        mapped independently. ``max_tokens=800`` is a deliberate cap — a
        single slide's extract is short, and keeping Stage 1 cheap leaves
        room for the big Stage 3 synthesis.

        Args:
            image_path: Path to one captured slide JPEG.

        Returns:
            The extracted content as concise, fence-free Markdown.
        """
        content = [
            {"type": "text", "text": SLIDE_EXTRACT_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": self._encode_image(image_path)},
            },
        ]
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": content}],
            temperature=TEMPERATURE,
            max_tokens=800,
        )
        return self._clean_markdown(response.choices[0].message.content or "")

    # -- Stage 2 (Accumulate): chronological manifest with resume -----------

    def _build_slide_manifest(self, slides_dir: Path, session_dir: Path) -> str:
        """Combine per-slide extracts into a resumable ``slides_manifest.txt``.

        Slides are processed in chronological order (``slide_001.jpg``
        naming sorts lexicographically). Any slide whose ``### Slide:`` header
        already exists in the manifest is skipped — this is the resume
        checkpoint, so interrupted runs continue and re-runs are free.

        Args:
            slides_dir: Directory of deduplicated JPEG slides.
            session_dir: Session root where ``slides_manifest.txt`` is written
                (alongside ``raw_transcript.txt``).

        Returns:
            The full accumulated manifest text (all slide sections).
        """
        session_dir = Path(session_dir)
        slides_dir = Path(slides_dir)
        manifest_path = session_dir / MANIFEST_FILENAME

        slide_paths = sorted(slides_dir.glob("*.jpg"))
        existing = ""
        if manifest_path.exists():
            existing = manifest_path.read_text(encoding="utf-8")

        # Filenames already extracted — treat as processed even if a rerun.
        already_have = {
            match.group(1).strip()
            for match in re.finditer(
                r"^### Slide: (.+)$", existing, flags=re.MULTILINE
            )
        }
        pending = [slide for slide in slide_paths if slide.name not in already_have]

        total = len(slide_paths)
        for slide in pending:
            position = slide_paths.index(slide) + 1
            print(f"Extracting slide {position}/{total} ({slide.name})...")
            entry = self._extract_slide_data(slide)
            existing += f"### Slide: {slide.name}\n{entry}\n\n"
            # Persist after every slide so a crash mid-deck resumes cleanly.
            manifest_path.write_text(existing, encoding="utf-8")

        return existing.strip()

    # -- Stage 3 (Reduce): text-only final synthesis ------------------------

    @retry(
        wait=wait_random(min=2, max=5),
        stop=stop_after_attempt(3),
        retry=retry_if_exception_type(_RETRYABLE_ERRORS),
        reraise=True,
    )
    def _call_synthesis(self, prompt: str) -> str:
        """Send the merged transcript + slide text to the LLM (no images).

        Args:
            prompt: Fully assembled Stage 3 prompt.

        Returns:
            Raw markdown response from the model.
        """
        response = self.client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
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

        Stage 2 (``_build_slide_manifest``) accumulates the per-slide vision
        extracts into ``slides_manifest.txt`` beside the transcript, then
        Stage 3 merges that manifest with the raw transcript in a single
        text-only LLM call and writes the result to ``notes_path``.

        Args:
            transcript_path: Path to the raw transcript text file.
            notes_path: Path where the cleaned Markdown notes will be written.
            slides_dir: Directory of deduplicated JPEG slides.

        Returns:
            True on success.

        Raises:
            ValueError: If transcript word count is below the minimum threshold.
        """
        transcript_path = Path(transcript_path)
        notes_path = Path(notes_path)
        session_dir = transcript_path.parent

        transcript_text = transcript_path.read_text(encoding="utf-8")
        word_count = len(transcript_text.split())

        if word_count < MIN_TRANSCRIPT_WORDS:
            raise ValueError(
                "Transcript too short. Aborting synthesis to prevent hallucination."
            )

        slide_manifest = self._build_slide_manifest(slides_dir, session_dir)
        print(f"  -> slide manifest ready: {session_dir / MANIFEST_FILENAME}")

        prompt = (
            f"{SYNTHESIS_INSTRUCTION}\n\n"
            f"--- Transcript ---\n{transcript_text}\n\n"
            f"--- Slide content (chronological) ---\n{slide_manifest}"
        )

        print("Synthesizing study notes from transcript + slide manifest...")
        raw_markdown = self._call_synthesis(prompt)
        cleaned_markdown = self._clean_markdown(raw_markdown)

        notes_path.write_text(cleaned_markdown, encoding="utf-8")
        return True


# -- CLI ----------------------------------------------------------------------


def _run_cli(argv: list[str]) -> int:
    """Synthesize notes for an existing session directory from the command line.

    Args:
        argv: ``sys.argv`` (``argv[1]`` is the session directory).

    Returns:
        Process exit code (0 = success, 1 = error, 2 = usage error).
    """
    if len(argv) < 2:
        print("Usage: python summarizer.py <session_directory>")
        print("Example: python summarizer.py output/OCHIENG-FRIDAY_2026-09-25")
        return 2

    session_dir = Path(argv[1])
    transcript_path = session_dir / TRANSCRIPT_FILENAME
    slides_dir = session_dir / "slides"
    notes_path = session_dir / f"{session_dir.name}.md"

    if not session_dir.is_dir():
        print(f"Error: session directory not found: {session_dir}")
        return 1
    if not transcript_path.exists():
        print(f"Error: transcript not found: {transcript_path}")
        return 1
    if not slides_dir.is_dir():
        print(
            f"Warning: slides directory not found at {slides_dir} — "
            "synthesis will be text-only."
        )

    print(f"Session   : {session_dir}")
    print(f"Transcript: {transcript_path}")
    print(f"Slides    : {slides_dir}")
    print(f"Notes out : {notes_path}")

    try:
        synthesizer = LectureSynthesizer()
        synthesizer.generate_notes(transcript_path, notes_path, slides_dir)
    except ValueError as exc:
        print(f"Error: {exc}")
        return 1

    print(f"Done. Notes saved to {notes_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(_run_cli(sys.argv))