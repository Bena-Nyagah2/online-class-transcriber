"""Offline re-synthesis smoke test for an already-captured session directory.

Runs Stage 3 only. Stage 1/2 are skipped because the session already has a
``slides_manifest.txt``; ``_build_slide_manifest`` treats every filename in it
as done, so no vision calls are made and this costs exactly one LLM request.

The API is path-based and does the writing itself:

    generate_notes(transcript_path: Path, notes_path: Path, slides_dir: Path) -> bool

It returns a bool, not the Markdown. Reading the notes back off disk is the
only way to see what was produced.
"""

import sys
from pathlib import Path

from summarizer import LectureSynthesizer

SESSION_DIR = Path(r"output\YT-TEST-1_2026-09-28")
NOTES_PATH = SESSION_DIR / "YT-TEST-1_FIXED.md"

transcript_path = SESSION_DIR / "raw_transcript.txt"
slides_dir = SESSION_DIR / "slides"

for required in (transcript_path, slides_dir):
    if not required.exists():
        print(f"Error loading artifacts: missing {required}")
        sys.exit(1)

print(f"Session:   {SESSION_DIR}")
print(f"Transcript: {transcript_path.stat().st_size:,} bytes")
print(f"Slides:     {len(list(slides_dir.glob('*.jpg')))} captured")
print(f"Manifest:   {SESSION_DIR / 'slides_manifest.txt'}")

print("\nSynthesizing notes using Academic Persona...")
synthesizer = LectureSynthesizer()

try:
    ok = synthesizer.generate_notes(transcript_path, NOTES_PATH, slides_dir)
except Exception as exc:  # surface the failsafes instead of swallowing them
    print(f"Synthesis failed: {type(exc).__name__}: {exc}")
    sys.exit(1)

if not ok:
    print("Synthesis failed: generate_notes returned False")
    sys.exit(1)

print(f"\nDone! {NOTES_PATH} ({NOTES_PATH.stat().st_size:,} bytes)")
