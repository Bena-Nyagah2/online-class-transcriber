import sys
from pathlib import Path
from summarizer import LectureSynthesizer

if len(sys.argv) < 2:
    print("Usage: python quick_test.py <path_to_output_dir>")
    sys.exit(1)

output_dir = Path(sys.argv[1])
transcript_path = output_dir / "raw_transcript.txt"
slides_dir = output_dir / "slides"
notes_path = output_dir / f"{output_dir.name}_RECOVERED.md"

if not transcript_path.exists():
    print(f"FATAL: Transcript not found at {transcript_path}")
    sys.exit(1)

print(f"🚀 Recovering synthesis for {output_dir.name}...")
synthesizer = LectureSynthesizer()

try:
    success = synthesizer.generate_notes(transcript_path, notes_path, slides_dir)
    if success:
        print(f"✅ Recovery Complete! Notes saved to: {notes_path}")
    else:
        print("❌ Synthesis failed silently (returned False).")
except Exception as e:
    print(f"❌ Synthesis exception: {e}")