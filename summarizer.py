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
import logging
import os
import re
import sys
import textwrap
from pathlib import Path
from typing import Optional

import openai
from dotenv import load_dotenv
from openai import APIConnectionError, APITimeoutError, RateLimitError
from rich.console import Console
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

# Warnings from the engine layers that cannot reach main.py's console. Rich
# resolves ``file`` lazily on every write, so pytest's capture still works on
# this instance and tests can patch ``summarizer.console`` directly.
console = Console()

# The standard-library logger carries pipeline warnings that belong to the
# process's logging stream rather than the operator-facing console -- the deck
# downsampling circuit breaker in particular. Same namespacing convention the
# gdrive/ledger services use, so every warning points at its module.
logger = logging.getLogger(__name__)

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

# Stage 1's per-slide budget. Raised from the original 800 because 800 was
# sized for a title slide, and the decks that overflow it are the dense ones:
# a statute table, a case-list slide, a full framework diagram. Those are
# exactly the slides a student most needs extracted, so the cap was failing
# hardest where it mattered most -- one bad slide cost the whole lecture's notes.
#
# 2048 rather than the more modest 1500: the vision model is
# ``meta/llama-3.2-11b-vision-instruct``, whose output ceiling sits far above
# this, and Stage 1's prompt is tiny (one image plus a short instruction), so
# the large-``max_tokens`` ceiling that drops connections in Stage 3 does not
# apply here. Cost is bounded either way, since generation halts at the model's
# natural stopping point regardless of what the ceiling permits.
SLIDE_MAX_TOKENS = 2048

# Appended to an extract that ran out of budget. Stage 3 reads the manifest as
# ground truth, so a partial extract is otherwise indistinguishable from a
# complete one -- the marker is what makes the gap visible to the consumer. The
# wording is deliberately content-free so it cannot be mistaken for something
# the slide actually said.
SLIDE_TRUNCATION_NOTE = "\n\n[Note: Extraction truncated due to slide density.]"

# Low temperature keeps the merge extractive rather than inventive, which is
# what the grounding directive assumes. Applied to both stages: Stage 1 wants
# faithful OCR, Stage 3 wants faithful synthesis.
TEMPERATURE = 0.2

MANIFEST_FILENAME = "slides_manifest.txt"

# Circuit breaker on the deck: at most this many slides Stage 2 will extract.
# A dense build-step deck or a long capture can yield hundreds of JPEGs, each
# of which is a vision round-trip (OCR cost) and a chunk of the Stage 3 prompt
# (context tokens). Past the cap the deck is uniformly downsampled to exactly
# this many slides -- first and last always kept -- so runtime and token spend
# stay bounded no matter how long the lecture ran.
MAX_SLIDES = 60

# Matches a manifest slide header, tolerating the optional ``[MM:SS] `` timing
# prefix so manifests written before timestamps existed still resume correctly.
_SLIDE_HEADER_RE = re.compile(r"^### Slide: (?:\[[^\]]*\]\s*)?(.+)$", re.MULTILINE)

_RETRYABLE_ERRORS = (RateLimitError, APIConnectionError, APITimeoutError)


def _uniform_slide_sample(slides: list[Path], limit: int = MAX_SLIDES) -> list[Path]:
    """Return ``limit`` slides spread evenly from the first to the last.

    The downsample is deterministic -- no randomness -- so the same deck always
    selects the same set, which keeps the manifest's resume checkpoint exact
    across re-runs and between a preview and its full run. The first and last
    slides are always kept: the opening framing and the closing review are the
    slides a student is most likely to need afterwards.

    Args:
        slides: Chronologically sorted slide paths.
        limit: Target sample size. Defaults to :data:`MAX_SLIDES`.

    Returns:
        ``slides`` itself when it already fits within the limit; otherwise a
        new list of exactly ``limit`` paths, order preserved, starting with
        ``slides[0]`` and ending with ``slides[-1]``.
    """
    if len(slides) <= limit:
        return slides
    # Walk a fixed step across the index range instead of drawing random picks:
    # pure Python, no dependencies, and exactly ``limit`` distinct slides
    # because a step > 1 separates every neighbouring pair by more than one.
    # (``round`` only ever lands on the ``len - 1`` last index at the final
    # point of the walk; the ``min`` is a free guard against float fuzz.)
    step = (len(slides) - 1) / (limit - 1)
    return [slides[min(round(step * index), len(slides) - 1)] for index in range(limit)]

_CRIMINOLOGY_SYSTEM_PROMPT = textwrap.dedent(
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

    TRUNCATED SLIDE HANDLING: A slide extract may end with the literal marker "[Note: Extraction truncated due to slide density.]" That marker is an annotation added by the extraction pipeline, NOT content from the slide. When you see it, that slide's extract is incomplete: cover the material that IS present, never guess at what the cut-off remainder said, and never quote or repeat the marker itself in the notes. Where a truncated slide covers material central to the lecture, say in the relevant section that the slide's detail was only partially captured.


    SPEAKER IDENTIFICATION: Cross-reference the timestamps of the slides with the timestamps of the audio transcript. If a slide's OCR text reveals a person's name or title (e.g., a lower-third graphic) during the exact time a specific [SPEAKER_XX] is talking, deduce their identity and replace [SPEAKER_XX] with their real name in the final notes.

    Your output will be graded on precision, legal accuracy, and utility to a student preparing for examinations.
    """
).strip()  # .strip() so the first directive is character 0, not character 1

# Stage 1: sent alongside exactly one slide image (criminology profile).
_CRIMINOLOGY_SLIDE_PROMPT = (
    "You are an expert academic visual analyst. Extract all text, bullet points, "
    "and statutory/case references. IF the image contains infographics, charts, "
    "diagrams, or visual metaphors, you MUST explicitly describe the data, "
    "structural relationships, and meaning of the visual elements. Be highly "
    "detailed but format concisely in Markdown."
)

# ---------------------------------------------------------------------------
# Domain profiles
# ---------------------------------------------------------------------------
#
# The synthesis prompts are the only domain-coupled part of this pipeline, so
# they live here as data rather than being smeared through the code. A profile
# supplies the two prompts the pipeline is parameterised by:
#
#   SYSTEM_PROMPT         -- Stage 3 system role: persona, mandated sections,
#                            and the domain-agnostic directives (grounding,
#                            truncated-slide handling, speaker identification).
#   SLIDE_EXTRACT_PROMPT  -- Stage 1 vision instruction sent with each slide.
#
# ``generate_notes`` looks its ``domain`` argument up in this table and falls
# back to the "general" profile when the key is unknown, so a bad value from a
# caller degrades to neutral meeting notes instead of crashing a run that has
# already spent money capturing and transcribing the session.

PROFILES: dict[str, dict[str, str]] = {
    "criminology": {
        "SYSTEM_PROMPT": _CRIMINOLOGY_SYSTEM_PROMPT,
        "SLIDE_EXTRACT_PROMPT": _CRIMINOLOGY_SLIDE_PROMPT,
    },
    "software": {
        "SYSTEM_PROMPT": textwrap.dedent(
            """
            STRICT OUTPUT FORMAT DIRECTIVE: Output ONLY the final Markdown document. Do NOT include preambles, drafting thoughts, scratchpads, or meta-commentary (e.g., NEVER output "Here's a thinking process:" or outline your planning steps). Your response MUST begin immediately with the exact characters "## Executive Summary".

            You are a Senior Technical Architect. You are receiving a chronological meeting transcript and a manifest of described visual slides. Your mandate is to produce a rigorous, structured engineering record -- not a blog post, not a summary, and certainly not a hallucinated narrative.

            CRITICAL RULES:
            1. **Structure is Mandatory.** The output MUST be organized into exactly these four Markdown sections, in this order, using ATX headings (##):
               - ## Executive Summary
               - ## Core Architectures & System Design
               - ## API & Code References
               - ## Action Items

            2. **Executive Summary** -- A concise TL;DR of the discussion (150-250 words). No fluff, no rhetorical questions. State the central problem, the decisions taken, and the practical takeaway for the engineering team.

            3. **Core Architectures & System Design** -- For each major component, service, or design decision discussed:
               - Name it precisely (e.g., "ingest gateway", "event ledger", "read-through cache").
               - Describe it in the speaker's own wording where possible, citing the transcript timestamp.
               - Note any trade-offs, rejected alternatives, or open questions raised in the discussion.
               - Use **bold** for the component name on first use.

            4. **API & Code References** -- Extract *explicitly* from OCR and audio. Each entry must include:
               - The reference exactly as given: an endpoint as "<METHOD> <path>"; a symbol as "<module>.<name>"; a snippet transcribed exactly as the slide shows it.
               - The context in which it was discussed (one sentence).
               - Do NOT add endpoints, symbols, or code the speaker did not mention. This is absolute: if no reference was spoken or shown, the section says so and lists nothing. Never supply a signature from your own knowledge, however well known it is, and never reuse the format templates above as if they were real references. If a slide shows a reference the speaker skipped, note it as "[Slide reference only]".
               - Use a Markdown table with columns: **Reference | Context | Source (Audio/Slide/Both)**.

            5. **Action Items** -- Capture every follow-up agreed in the session:
               - Number them 1., 2., 3.
               - Attribute each item to its owner when a name was given, otherwise write "Unassigned".
               - Include a deadline only when one was spoken or shown; never invent one.

            6. **Visual Fusion:** When the transcript aligns with a described diagram, screenshot, or architecture sketch from the slides, explain the visual relationship in context. Do not describe the visual in isolation -- anchor it to the spoken argument.

            7. **No Hallucinations, No Fluff.** The STRICT OUTPUT FORMAT DIRECTIVE, the STRICT GROUNDING DIRECTIVE and the SPEAKER IDENTIFICATION directives are all non-negotiable.

            STRICT GROUNDING DIRECTIVE: NEVER invent, infer, or hallucinate URLs, code snippets, or fictional narratives. If a slide shows code, transcribe exactly what is visible. If the audio topic entirely mismatches the visual slide, report both facts neutrally.

            TRUNCATED SLIDE HANDLING: A slide extract may end with the literal marker "[Note: Extraction truncated due to slide density.]" That marker is an annotation added by the extraction pipeline, NOT content from the slide. When you see it, that slide's extract is incomplete: cover the material that IS present, never guess at what the cut-off remainder said, and never quote or repeat the marker itself in the notes. Where a truncated slide covers material central to the discussion, say in the relevant section that the slide's detail was only partially captured.

            SPEAKER IDENTIFICATION: Cross-reference the timestamps of the slides with the timestamps of the transcript. If a slide's OCR text reveals a person's name or title (e.g., a lower-third graphic) during the exact time a specific [SPEAKER_XX] is talking, deduce their identity and replace [SPEAKER_XX] with their real name in the final notes.

            Your output will be graded on precision, technical accuracy, and utility to the engineers picking this work up.
            """
        ).strip(),
        "SLIDE_EXTRACT_PROMPT": (
            "Extract all text, code snippets, system diagrams, and architectures. "
            "IF the image contains code, schemas, API signatures, sequence or "
            "flow diagrams, or infrastructure topology, you MUST explicitly "
            "describe the data flow, structural relationships, and meaning of "
            "the visual elements. Be highly detailed but format concisely in "
            "Markdown."
        ),
    },
    "general": {
        "SYSTEM_PROMPT": textwrap.dedent(
            """
            STRICT OUTPUT FORMAT DIRECTIVE: Output ONLY the final Markdown document. Do NOT include preambles, drafting thoughts, scratchpads, or meta-commentary (e.g., NEVER output "Here's a thinking process:" or outline your planning steps). Your response MUST begin immediately with the exact characters "## Executive Summary".

            You are a professional meeting-notes facilitator. You are receiving a chronological transcript of a meeting and a manifest of described visual slides. Your mandate is to produce a rigorous, structured record of the discussion -- not a blog post, not a summary, and certainly not a hallucinated narrative.

            CRITICAL RULES:
            1. **Structure is Mandatory.** The output MUST be organized into exactly these four Markdown sections, in this order, using ATX headings (##):
               - ## Executive Summary
               - ## Key Points & Discussion
               - ## Decisions & References
               - ## Action Items

            2. **Executive Summary** -- A concise TL;DR of the meeting (150-250 words). No fluff, no rhetorical questions. State the purpose of the meeting, the central conclusions reached, and the practical takeaway for the attendees.

            3. **Key Points & Discussion** -- For each major topic discussed:
               - Name it precisely.
               - Summarise it using the speaker's own wording where possible, citing the transcript timestamp.
               - Note any competing views or unresolved questions raised in the discussion.
               - Use **bold** for the topic name on first use.

            4. **Decisions & References** -- Extract *explicitly* from OCR and audio. Each entry must include:
               - The item exactly as it was given: a decision as "<what was agreed> (<owner/date if given>)"; a referenced document, link, or artifact transcribed as the slide shows it.
               - The context in which it was raised (one sentence).
               - Do NOT add decisions, documents, or references nobody raised. This is absolute: if nothing was decided or referenced, the section says so and lists nothing. Never supply a reference from your own knowledge, and never reuse the format templates above as if they were real. If a slide shows a reference the speaker skipped, note it as "[Slide reference only]".
               - Use a Markdown table with columns: **Item | Context | Source (Audio/Slide/Both)**.

            5. **Action Items** -- Capture every follow-up agreed in the meeting:
               - Number them 1., 2., 3.
               - Attribute each item to its owner when a name was given, otherwise write "Unassigned".
               - Include a deadline only when one was spoken or shown; never invent one.

            6. **Visual Fusion:** When the transcript aligns with a described chart, diagram, or document from the slides, explain the visual relationship in context. Do not describe the visual in isolation -- anchor it to the spoken discussion.

            7. **No Hallucinations, No Fluff.** The STRICT OUTPUT FORMAT DIRECTIVE, the STRICT GROUNDING DIRECTIVE and the SPEAKER IDENTIFICATION directives are all non-negotiable.

            STRICT GROUNDING DIRECTIVE: NEVER invent, infer, or hallucinate URLs, code snippets, or fictional narratives. If a slide shows code, transcribe exactly what is visible. If the audio topic entirely mismatches the visual slide, report both facts neutrally.

            TRUNCATED SLIDE HANDLING: A slide extract may end with the literal marker "[Note: Extraction truncated due to slide density.]" That marker is an annotation added by the extraction pipeline, NOT content from the slide. When you see it, that slide's extract is incomplete: cover the material that IS present, never guess at what the cut-off remainder said, and never quote or repeat the marker itself in the notes. Where a truncated slide covers material central to the meeting, say in the relevant section that the slide's detail was only partially captured.

            SPEAKER IDENTIFICATION: Cross-reference the timestamps of the slides with the timestamps of the transcript. If a slide's OCR text reveals a person's name or title (e.g., a lower-third graphic) during the exact time a specific [SPEAKER_XX] is talking, deduce their identity and replace [SPEAKER_XX] with their real name in the final notes.

            Your output will be graded on precision, fidelity to the discussion, and utility to the attendees.
            """
        ).strip(),
        "SLIDE_EXTRACT_PROMPT": (
            "You are an expert meeting-notes visual analyst. Extract all text, "
            "bullet points, and referenced documents or links. IF the image "
            "contains charts, diagrams, or other visuals, you MUST explicitly "
            "describe the data, structural relationships, and meaning of the "
            "visual elements. Be highly detailed but format concisely in Markdown."
        ),
    },
}

# The module-level names predate the profile table and are imported by tests
# and by direct callers. They are -- and remain -- the criminology profile's
# entries, which is what they always were. ``generate_notes`` no longer reads
# them; it resolves the caller's ``domain`` through :func:`resolve_profile`.
SYSTEM_PROMPT = PROFILES["criminology"]["SYSTEM_PROMPT"]
SLIDE_EXTRACT_PROMPT = PROFILES["criminology"]["SLIDE_EXTRACT_PROMPT"]


def resolve_profile(domain: str) -> dict[str, str]:
    """Return the profile for ``domain``, falling back to the general one.

    Args:
        domain: Profile key, e.g. ``"criminology"`` or ``"software"``.

    Returns:
        The matching profile mapping, or the generic meeting-notes profile
        when the key is unknown -- a typo or a key added by a newer client
        must degrade to neutral notes, not fail a run that is already paid
        for end to end.
    """
    return PROFILES.get(domain, PROFILES["general"])


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
    def _extract_slide_data(
        self, image_path: Path, slide_prompt: str = SLIDE_EXTRACT_PROMPT
    ) -> str:
        """Extract the content of a single slide via a one-image vision call.

        NVIDIA's endpoint accepts one image per request, so each slide is
        mapped independently at :data:`SLIDE_MAX_TOKENS` -- kept well below
        Stage 3's budget, because a 90-slide lecture issues 90 of these calls.

        A slide that exhausts its budget degrades rather than aborts: the
        partial extract is returned with :data:`SLIDE_TRUNCATION_NOTE`
        appended, so the remaining slides are still processed and the operator
        sees one warning per affected slide. The alternative -- raising -- meant
        a single dense slide destroyed the notes for the entire lecture,
        including every slide that had already extracted cleanly.

        The marker is what keeps this safe. Stage 3 treats the manifest as
        ground truth, so an unmarked partial extract reads as "this slide said
        only this", and the model then reasons confidently from a gap it cannot
        see. Flagging the gap moves the failure from silent to visible.

        Args:
            image_path: Path to one captured slide JPEG.
            slide_prompt: Stage 1 instruction sent ahead of the image. Defaults
                to the criminology profile's prompt so a direct call behaves
                as it always has; ``generate_notes`` passes the selected
                profile's own.

        Returns:
            The extracted content as concise, fence-free Markdown, carrying
            :data:`SLIDE_TRUNCATION_NOTE` when the token cap was hit.
        """
        content = [
            {"type": "text", "text": slide_prompt},
            {
                "type": "image_url",
                "image_url": {"url": self._encode_image(image_path)},
            },
        ]
        response = self.client.chat.completions.create(
            model=MAP_MODEL,
            messages=[{"role": "user", "content": content}],
            temperature=TEMPERATURE,
            max_tokens=SLIDE_MAX_TOKENS,
        )

        # Cleaning happens *before* the note is appended: _clean_markdown is
        # built to strip the model's own ``` fences, and the note is our
        # annotation rather than model output, so it must not be fed through a
        # parser aimed at the model's formatting habits.
        choice = response.choices[0]
        extract = self._clean_markdown(choice.message.content or "")

        # Truncation is deterministic, so it stays out of the retry path --
        # tenacity re-raises immediately rather than spending two more vision
        # calls (the pipeline's most expensive line item) to hit the same wall.
        if choice.finish_reason == "length":
            console.print(
                f"[bold yellow]⚠ Warning: Slide {image_path.name} truncated "
                f"due to token limits.[/bold yellow]"
            )
            return f"{extract}{SLIDE_TRUNCATION_NOTE}"

        return extract

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

    def _build_slide_manifest(
        self,
        slides_dir: Path,
        session_dir: Path,
        slide_prompt: str = SLIDE_EXTRACT_PROMPT,
    ) -> str:
        """Combine per-slide extracts into a resumable ``slides_manifest.txt``.

        Slides are processed in chronological order (``slide_001.jpg``
        naming sorts lexicographically). Any slide whose ``### Slide:`` header
        already exists in the manifest is skipped — this is the resume
        checkpoint, so interrupted runs continue and re-runs are free. The one
        exception is an entry carrying :data:`SLIDE_TRUNCATION_NOTE`: a partial
        extract is not a finished one, so its stale entry is discarded and that
        slide is re-attempted on the next run.

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
            slide_prompt: Stage 1 instruction forwarded to
                :meth:`_extract_slide_data` for every pending slide. Defaults
                to the criminology profile's prompt for direct callers;
                ``generate_notes`` passes the selected profile's own.

        Returns:
            The full accumulated manifest text (all slide sections).
        """
        session_dir = Path(session_dir)
        slides_dir = Path(slides_dir)
        manifest_path = session_dir / MANIFEST_FILENAME

        slide_paths = sorted(slides_dir.glob("*.jpg"))

        # Circuit breaker: a deck past MAX_SLIDES is downsampled to exactly
        # MAX_SLIDES here, before any OCR or LLM work, so the per-slide cost
        # (a vision round-trip plus a manifest write) and the Stage 3 context
        # stay bounded however long the capture ran. Sampling the *full* list --
        # not just the not-yet-extracted remainder -- keeps the choice
        # deterministic across re-runs, so a resume selects identically and the
        # checkpoint never shifts under the operator.
        if len(slide_paths) > MAX_SLIDES:
            logger.warning(
                "Slide count %d exceeds MAX_SLIDES. Downsampling to %d.",
                len(slide_paths),
                MAX_SLIDES,
            )
            slide_paths = _uniform_slide_sample(slide_paths, MAX_SLIDES)

        existing = ""
        if manifest_path.exists():
            existing = manifest_path.read_text(encoding="utf-8")

        slide_times = self._read_slide_times(slides_dir)

        # Filenames already extracted — treat as processed even if a rerun.
        # The regex swallows the optional ``[MM:SS] `` prefix, so a manifest
        # from before this feature (and one from after) both resume correctly.
        #
        # One exception: a slide whose extract was truncated is NOT finished,
        # so its stale entry is dropped and the slide is re-attempted. Under the
        # old raise-on-truncation behaviour the failed slide was simply never
        # written, so a re-run retried it for free. Persisting the partial
        # extract would silently forfeit that retry — the deck would be stuck
        # on the truncated text until someone hand-edited the manifest. The
        # entry is removed rather than skipped so the fresh extract replaces it
        # instead of appending a second ``### Slide:`` header for the same
        # image, which would have Stage 3 reading one slide twice.
        already_have: set[str] = set()
        headers = list(_SLIDE_HEADER_RE.finditer(existing))
        if headers:
            spans: list[tuple[int, int]] = [(0, headers[0].start())]
            for index, match in enumerate(headers):
                start = match.end()
                end = (
                    headers[index + 1].start()
                    if index + 1 < len(headers)
                    else len(existing)
                )
                if SLIDE_TRUNCATION_NOTE.strip() in existing[start:end]:
                    continue
                spans.append((match.start(), end))
                already_have.add(match.group(1).strip())
            kept = "".join(existing[begin:stop] for begin, stop in spans)
            if kept != existing:
                existing = kept
                manifest_path.write_text(existing, encoding="utf-8")

        pending = [slide for slide in slide_paths if slide.name not in already_have]

        total = len(slide_paths)
        for slide in pending:
            position = slide_paths.index(slide) + 1
            print(f"Extracting slide {position}/{total} ({slide.name})...")
            entry = self._extract_slide_data(slide, slide_prompt)
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
    def _call_synthesis(self, prompt: str, system_prompt: str = SYSTEM_PROMPT) -> str:
        """Send the merged transcript + slide text to the LLM (no images).

        The call is deliberately pinned to ``max_tokens=4096`` and
        ``temperature=0.2``. The budget is the outer guard against a truncated
        document; the STRICT OUTPUT FORMAT DIRECTIVE is the inner one, since
        every token a reasoner model spends narrating its own planning is a
        token taken from the last section of the notes.

        Args:
            prompt: Fully assembled Stage 3 prompt.
            system_prompt: The domain profile's system role. Defaults to the
                criminology profile's prompt so a direct call behaves as it
                always has; ``generate_notes`` passes the selected profile's.

        Returns:
            Raw markdown response from the model.

        Raises:
            RuntimeError: If the model stopped at the token cap. The partial
                document is discarded rather than returned.
        """
        response = self.client.chat.completions.create(
            model=REDUCE_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
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
        self,
        transcript_path: Path,
        notes_path: Path,
        slides_dir: Path,
        domain: str = "general",
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
            domain: Key into :data:`PROFILES` selecting the persona, the
                mandated section headings, and the Stage 1 slide instruction.
                An unknown key falls back to the "general" profile via
                :func:`resolve_profile` rather than raising: a bad value must
                degrade to neutral notes, not throw away a session that has
                already been captured and transcribed.

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

        profile = resolve_profile(domain)

        slide_manifest = self._build_slide_manifest(
            slides_dir, session_dir, profile["SLIDE_EXTRACT_PROMPT"]
        )
        print(f"  -> slide manifest ready: {session_dir / MANIFEST_FILENAME}")

        prompt = (
            f"{SYNTHESIS_INSTRUCTION}\n\n"
            f"--- Transcript ---\n{transcript_text}\n\n"
            f"--- Slide content (chronological) ---\n{slide_manifest}"
        )

        print("Synthesizing study notes from transcript + slide manifest...")
        raw_markdown = self._call_synthesis(prompt, profile["SYSTEM_PROMPT"])
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