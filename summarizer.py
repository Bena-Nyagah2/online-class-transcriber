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
transcript. Each slide header is prefixed with the offset at which that frame
was captured (``### Slide: [01:44] slide_002.jpg``), read from the
``slide_times.json`` sidecar that :mod:`vision` writes. The manifest doubles as
a **resume cache** — any slide filename already present is skipped on re-runs,
so an interrupted extraction continues where it stopped and completed sessions
never re-pay for API calls.

Stage 3 — *Reduce*  ``generate_notes``: a single **text-only** call merges the
raw transcript with the accumulated slide text into one cohesive academic
study guide, written to the session's ``<session-name>.md`` file.
"""

from __future__ import annotations

import base64
import json
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

from transcriber import TRANSCRIPT_FILENAME, format_timestamp

# Read the slide timing sidecar written by vision.py. Reusing the constant
# keeps the two modules from drifting on the filename.
from vision import SLIDE_TIMES_FILENAME

# Load NVIDIA_API_KEY from a local .env (if present) before any client reads
# the environment. Explicit shell exports still take precedence.
load_dotenv()

MODEL_NAME = os.environ.get("LLM_MODEL_NAME", "meta/llama-3.1-8b-instruct")

# Per-stage model overrides. Stage 1 (Map) reads one small image per call and
# only needs OCR/description quality; Stage 3 (Reduce) is the reasoning-heavy
# merge over the whole transcript plus deck and benefits from a larger model.
# Keeping them separate lets either stage be retuned independently — e.g. a
# vision-tuned 70B for the map, a cheap 8B for the reduce — without editing
# call sites. ``MODEL_NAME`` remains the shared fallback for both.
MAP_MODEL = os.environ.get("MAP_MODEL", MODEL_NAME)
REDUCE_MODEL = os.environ.get("REDUCE_MODEL", MODEL_NAME)

# Nemotron 3.x (the current REDUCE_MODEL) reasons by default and writes that
# chain-of-thought into ``content`` ahead of the document, which starves the
# output budget and produces meta-commentary instead of notes. Off by default
# because this pipeline wants the document, not the deliberation. Set
# ``REDUCE_REASONING_ENABLED=1`` to turn it back on for a non-reasoning
# ``REDUCE_MODEL``, or if you ever want the model to think before writing.
#
# Stage 1 is unaffected by design: ``MAP_MODEL`` is a Llama vision model with
# no thinking mode, so sending the parameter there would be noise.
REDUCE_REASONING_ENABLED = os.environ.get(
    "REDUCE_REASONING_ENABLED", "0"
).strip().lower() in {"1", "true", "yes", "on"}

NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
NVIDIA_API_KEY_ENV = "NVIDIA_API_KEY"

MIN_TRANSCRIPT_WORDS = 50

# Stage 3 token budget: the final synthesis merges the whole transcript plus the
# whole slide deck into four mandated sections. 4000 was not enough -- a
# reasoner model (Llama 3.1 70B and friends) often spends its first few hundred
# tokens on a visible scratchpad before it starts the document, so the notes
# lost their entire "Exam Preparation" section to a `finish_reason: "length"`
# cut. 4096 is the round figure above that leak, and the STRICT OUTPUT FORMAT
# DIRECTIVE at the head of SYSTEM_PROMPT attacks the same failure from the
# other end by forbidding the scratchpad in the first place.
#
# If a real lecture still trips the cap, the failsafe in ``_call_synthesis``
# will now crash rather than write a partial document. That crash is the
# signal to raise this number -- or to switch REDUCE_MODEL to a model that
# does not narrate its planning -- not to remove the check.
MAX_TOKENS = 4096  # Stage 3: final synthesis over the full transcript + deck

# Low temperature keeps the merge extractive rather than inventive, which is
# what the grounding directive assumes. Applied to both stages: Stage 1 wants
# faithful OCR, Stage 3 wants faithful synthesis.
TEMPERATURE = 0.2

MANIFEST_FILENAME = "slides_manifest.txt"

# Matches a manifest slide header, tolerating the optional ``[MM:SS] `` timing
# prefix so manifests written before timestamps existed still resume correctly.
_SLIDE_HEADER_RE = re.compile(r"^### Slide: (?:\[[^\]]*\]\s*)?(.+)$", re.MULTILINE)

_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)

SYSTEM_PROMPT = textwrap.dedent(
    """
    STRICT OUTPUT FORMAT DIRECTIVE: Output ONLY the final Markdown document. Do NOT include preambles, drafting thoughts, scratchpads, or meta-commentary (e.g., NEVER output "Here's a thinking process:" or outline your planning steps). Your response MUST begin immediately with the exact characters "## Executive Summary".

    You are a Tenured University Professor in Law and Criminology. You are receiving a chronological lecture transcript and a manifest of described visual slides. Your mandate is to produce a rigorous, structured academic study guide -- not a blog post, not a summary, and certainly not a hallucinated narrative.

    CRITICAL RULES:
    1. **Structure is Mandatory.** The output MUST be organized into exactly these four Markdown sections, in this order, using ATX headings (##):
       - ## Executive Summary
       - ## Core Theories & Concepts
       - ## Statutes, Case Law, & References
       - ## Exam Preparation

    2. **Executive Summary** -- A concise TL;DR of the lecture (150-250 words). No fluff, no rhetorical questions. State the central thesis, the key statutes discussed, and the practical takeaway for a practitioner.

    3. **Core Theories & Concepts** -- For each major theory or concept lectured:
       - Name it precisely (e.g., "Actus Reus", "Mens Rea", "Dolus Eventualis").
       - Define it using the lecturer's own wording where possible, citing the transcript timestamp.
       - Note any academic debates or competing interpretations raised in the lecture.
       - Use **bold** for the concept name on first use.

    4. **Statutes, Case Law, & References** -- Extract *explicitly* from OCR and audio. Each entry must include:
       - Full citation, written as a *format template* with the real values substituted in: a statute as "<Act Name> <Year>, s. <section>"; a case as "<Case Name> [<Year>] <reporter> <page>". Never reproduce the template itself -- it contains no real authority.
       - The context in which it was discussed (one sentence).
       - Do NOT add citations the lecturer did not mention. This is absolute: if no authority was spoken or shown, the section says so and lists nothing. Never supply a citation from your own knowledge, however well known it is, and never reuse the format templates above as if they were real cases. If a slide shows an authority the lecturer skipped, note it as "[Slide reference only]".
       - Use a Markdown table with columns: **Citation | Context | Source (Audio/Slide/Both)**.

    5. **Exam Preparation** -- Generate exactly THREE high-level synthesis questions that:
       - Require integrating multiple concepts from the lecture.
       - Are answerable *only* from the provided material.
       - Mirror the style of a final-year law school exam (problem question or essay prompt).
       - Number them 1., 2., 3.

    6. **Visual Fusion:** When the transcript aligns with a described infographic or diagram from the slides, explain the visual relationship in context. Do not describe the visual in isolation -- anchor it to the spoken argument.

    7. **No Hallucinations, No Fluff.** The STRICT OUTPUT FORMAT DIRECTIVE, the STRICT GROUNDING DIRECTIVE and the SPEAKER IDENTIFICATION directives are all non-negotiable.

    STRICT GROUNDING DIRECTIVE: NEVER invent, infer, or hallucinate URLs, code snippets, or fictional narratives. If a slide shows code, transcribe exactly what is visible. If the audio topic entirely mismatches the visual slide, report both facts neutrally.

    SPEAKER IDENTIFICATION: Cross-reference the timestamps of the slides with the timestamps of the audio transcript. If a slide's OCR text reveals a person's name or title (e.g., a lower-third graphic) during the exact time a specific [SPEAKER_XX] is talking, deduce their identity and replace [SPEAKER_XX] with their real name in the final notes.

    Your output will be graded on precision, legal accuracy, and utility to a student preparing for examinations.
    """
).strip()  # .strip() so the first directive is character 0, not character 1

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
            timeout=120.0,
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
        room for the big Stage 3 synthesis. Because that budget is a cap
        rather than a target, the response is checked for exhaustion and a
        truncated extract is refused rather than written to the manifest.

        Args:
            image_path: Path to one captured slide JPEG.

        Returns:
            The extracted content as concise, fence-free Markdown.

        Raises:
            RuntimeError: If the model stopped at the token cap for this
                slide. The partial extract is discarded rather than stored.
        """
        content = [
            {"type": "text", "text": SLIDE_EXTRACT_PROMPT},
            {
                "type": "image_url",
                "image_url": {"url": self._encode_image(image_path)},
            },
        ]
        response = self.client.chat.completions.create(
            model=MAP_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=TEMPERATURE,
            max_tokens=800,
        )

        # Symmetric to the Stage 3 failsafe, and for the same reason. A slide
        # that ran out of budget at 800 tokens yields an extract that stops
        # mid-sentence, and that partial text is about to be written into the
        # manifest as though it were the whole slide. Stage 3 is then told to
        # treat the manifest as ground truth -- a truncated extract does not
        # read as "incomplete", it reads as "this is all the slide said", so
        # the model will reason from a gap it cannot see. Failing here is the
        # only honest option; the alternative launders missing content into
        # confident wrong notes.
        #
        # The blast radius is bounded: ``_build_slide_manifest`` persists the
        # manifest after every slide, so a crash here costs only the in-flight
        # slide and a re-run resumes from the last good one.
        #
        # As in Stage 3, RuntimeError is absent from ``_RETRYABLE_ERRORS`` so
        # tenacity re-raises immediately instead of spending two more calls to
        # hit the same wall. The remedy is a larger 800 or a tighter
        # SLIDE_EXTRACT_PROMPT, not a retry loop.
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError(
                f"Stage 1 hit the token cap on {image_path.name} — "
                "slide extract would be truncated."
            )

        return self._clean_markdown(choice.message.content or "")

    # -- Stage 2 (Accumulate): chronological manifest with resume -----------

    @staticmethod
    def _read_slide_times(slides_dir: Path) -> dict[str, float]:
        """Load the ``slide_times.json`` sidecar written by the vision catcher.

        Args:
            slides_dir: Directory holding the captured slide JPEGs.

        Returns:
            A filename-to-elapsed-seconds mapping, or an empty mapping when the
            sidecar is absent. The offline re-run path documented in the module
            header feeds this method sessions captured before timestamps were
            recorded, and a missing or unreadable sidecar must not fail the run —
            the manifest simply falls back to untimestamped headers.
        """
        times_path = Path(slides_dir) / SLIDE_TIMES_FILENAME
        if not times_path.is_file():
            return {}
        try:
            raw = json.loads(times_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return {}
        if not isinstance(raw, dict):
            return {}
        # Keep only real numeric offsets; a corrupt entry must not become a
        # bogus "[00:00]" that reads to the model as a confident claim.
        return {
            str(name): float(value)
            for name, value in raw.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }

    def _build_slide_manifest(self, slides_dir: Path, session_dir: Path) -> str:
        """Combine per-slide extracts into a resumable ``slides_manifest.txt``.

        Slides are processed in chronological order (``slide_001.jpg``
        naming sorts lexicographically). Any slide whose ``### Slide:`` header
        already exists in the manifest is skipped — this is the resume
        checkpoint, so interrupted runs continue and re-runs are free.

        Each header carries the slide's capture offset from
        ``slide_times.json``::

            ### Slide: [01:44] slide_002.jpg
            <extracted content>

        That ``[01:44]`` is rendered by the same :func:`format_timestamp` the
        transcript's turn headers use, so Stage 3 receives both channels on one
        clock and can align them. Slides with no recorded offset keep a bare
        filename instead of a fabricated time.

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

        slide_times = self._read_slide_times(slides_dir)

        # Filenames already extracted — treat as processed even if a rerun.
        # The regex swallows the optional ``[MM:SS] `` prefix, so a manifest
        # from before this feature (and one from after) both resume correctly.
        already_have = {
            match.group(1).strip() for match in _SLIDE_HEADER_RE.finditer(existing)
        }
        pending = [slide for slide in slide_paths if slide.name not in already_have]

        total = len(slide_paths)
        for slide in pending:
            position = slide_paths.index(slide) + 1
            print(f"Extracting slide {position}/{total} ({slide.name})...")
            entry = self._extract_slide_data(slide)
            # Prefix the header with the capture offset so Stage 3 can line the
            # slide up against the transcript's [SPEAKER_xx] (MM:SS) headers.
            # format_timestamp is shared with the transcript, so both channels
            # are guaranteed to be on the same clock. A slide with no recorded
            # time (a hand-dropped image, or a pre-timestamp session) simply
            # keeps the bare filename rather than inventing a time.
            offset = slide_times.get(slide.name)
            label = (
                f"[{format_timestamp(offset)}] {slide.name}"
                if offset is not None
                else slide.name
            )
            existing += f"### Slide: {label}\n{entry}\n\n"
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

        The call is deliberately pinned to ``max_tokens=4096`` and
        ``temperature=0.2``. The budget is the outer guard against a truncated
        document; the STRICT OUTPUT FORMAT DIRECTIVE is the inner one, since
        every token a reasoner model spends narrating its own planning is a
        token taken from the last section of the notes.

        Args:
            prompt: Fully assembled Stage 3 prompt.

        Returns:
            Raw markdown response from the model.

        Raises:
            RuntimeError: If the model stopped at the token cap. The partial
                document is discarded rather than returned.
        """
        response = self.client.chat.completions.create(
            model=REDUCE_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            # Nemotron 3.x ships as a *reasoning* model: thinking is on by
            # default, and the chain-of-thought is emitted inline in `content`
            # rather than in a separate `reasoning` field. Left on, it spends
            # the whole budget narrating its own plan -- the previous run of
            # this session produced 112 lines in which everything before the
            # document was "Here's a thinking process:", and not one heading.
            #
            # The STRICT OUTPUT FORMAT DIRECTIVE in SYSTEM_PROMPT cannot fix
            # this. A model whose training makes CoT the default path does not
            # reliably abandon it because a system prompt asked nicely; the
            # switch has to be flipped at the API level. Disabling reasoning
            # frees the entire budget for the four mandated sections, which is
            # what MAX_TOKENS was sized for in the first place.
            #
            # Stage 1 is deliberately untouched: MAP_MODEL is a Llama vision
            # model with no thinking mode, and the parameter is meaningless
            # there. See REDUCE_REASONING_ENABLED below.
            #
            # Both branches send the parameter explicitly. Omitting it is NOT
            # the same as disabling: Nemotron 3 defaults to thinking *on*, so
            # a bare request would silently reintroduce the scratchpad.
            extra_body={
                "chat_template_kwargs": {
                    "enable_thinking": REDUCE_REASONING_ENABLED
                }
            },
        )

        # A response that stopped because it ran out of budget is not a
        # response. Left unchecked, the caller receives a document that simply
        # stops mid-section -- normally losing "Exam Preparation", the last of
        # the four mandated sections -- writes it to disk, and
        # ``generate_notes`` still returns True. That is silent data loss
        # wearing the costume of a clean run, and the reader has no way to tell
        # the notes are incomplete.
        #
        # Raising rather than warning also keeps this out of the retry loop:
        # RuntimeError is absent from ``_RETRYABLE_ERRORS``, so tenacity
        # re-raises it on the first pass instead of spending three more
        # full-length calls to hit the identical wall. Truncation is
        # deterministic for a given prompt and transcript, so retrying is pure
        # cost. The fix belongs at the budget (raise MAX_TOKENS) or the
        # directive (stop the model leaking a scratchpad), not in a loop.
        choice = response.choices[0]
        if choice.finish_reason == "length":
            raise RuntimeError("Stage 3 hit the token cap — notes would be truncated.")

        return choice.message.content or ""

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