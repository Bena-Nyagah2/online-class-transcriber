"""Phase 5 — "Saccade & Settle" screen capture with perceptual deduplication.

Grabs the primary display on a poll timer from a daemon thread and only
writes a slide when the screen *settles* on a stable, distinctly new frame:

1. **Saccade (motion) detection** — while the display keeps changing (page
   scroll, lecture video, drag), the settle stopwatch keeps resetting, so
   transient frames are never written;
2. **Settle detection** — once the screen holds still for ``settle_delay``
   seconds, the frame becomes a save candidate;
3. **Deduplication** — a candidate is written only when it differs from the
   last saved frame by more than ``hash_threshold``, so a repeated or
   near-identical slide costs nothing.

Every saved slide also records **when** it was captured, as an elapsed offset
from the start of the capture session, in ``slides/slide_times.json``. Stage 3
of the summarizer needs that offset to line a slide up against the audio
transcript: the transcript's turn headers and the manifest's slide headers must
be expressed on the same clock before a model can cross-reference them.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path

import imagehash
import mss
from PIL import Image

SLIDE_TIMES_FILENAME = "slide_times.json"
"""Sidecar in the slides directory mapping each slide filename to its offset.

Raw seconds are stored rather than a preformatted string so the summarizer can
render them with the *same* clock helper the transcript uses. Preformatting here
would let the two clocks drift apart and break the comparison.
"""


class SlideCatcher:
    """Captures settled, distinct lecture slides while ignoring scrolling."""

    def __init__(
        self,
        output_dir: Path | str,
        poll_interval: float = 1.0,
        settle_delay: float = 3.0,
        hash_threshold: int = 5,
    ) -> None:
        """Configure the catcher and prepare the ``slides`` subdirectory.

        Args:
            output_dir: Session directory; slides land in ``<output_dir>/slides``.
            poll_interval: Seconds between screen grabs.
            settle_delay: Seconds the screen must stay still before its
                frame counts as settled.
            hash_threshold: Largest pHash distance still considered the same
                slide (lower = stricter deduplication).
        """
        self.output_dir = Path(output_dir)
        self.slides_dir = self.output_dir / "slides"
        os.makedirs(self.slides_dir, exist_ok=True)
        self.times_path = self.slides_dir / SLIDE_TIMES_FILENAME

        self.poll_interval = poll_interval
        self.settle_delay = settle_delay
        self.hash_threshold = hash_threshold

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Origin for every elapsed offset below. Set in start() so the clock
        # begins when capture does, not at construction.
        self.start_time: float = 0.0

        # pHash of the most recent frame (drives the motion check).
        self.last_frame_hash: imagehash.ImageHash | None = None
        # pHash of the last file actually written (drives deduplication).
        self.last_saved_hash: imagehash.ImageHash | None = None
        # Wall-clock timestamp of the last detected motion.
        self.stable_since: float = 0.0
        self.slide_index: int = 0
        # Append-only from the capture thread, read after stop(); safe
        # under the GIL, no extra locking needed.
        self.captured_slides: list[Path] = []
        # Elapsed seconds from start_time at the instant each frame was
        # written, keyed by filename. This is the time-domain signal Stage 3
        # needs to align a slide against the transcript.
        self.slide_times: dict[str, float] = {}

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Spawn and start the daemon capture thread.

        The elapsed-time origin is stamped here rather than in ``__init__`` so
        that a catcher constructed early (or restarted) does not accumulate
        dead time into every slide offset. In ``main.py`` this runs just before
        the recorder starts, so the slide clock and the transcript clock share
        an origin to within the thread-spawn delay.
        """
        self.start_time = time.time()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Capture loop: grab -> motion/settle state machine -> poll wait.

        ``mss`` is instantiated inside the thread because COM/thread
        affinity on Windows breaks if it is created elsewhere.
        """
        with mss.mss() as sct:
            # monitors[0] is the virtual bounding box; [1] is the primary.
            monitor = sct.monitors[1]

            while not self._stop_event.is_set():
                raw_frame = sct.grab(monitor)
                img = Image.frombytes(
                    "RGB", raw_frame.size, raw_frame.bgra, "raw", "BGRX"
                )
                current_hash = imagehash.phash(img)

                # Motion check: a large delta means the screen is still
                # scrolling/playing, so reset the settle stopwatch.
                if (
                    self.last_frame_hash is None
                    or (current_hash - self.last_frame_hash) > 2
                ):
                    self.stable_since = time.time()
                else:
                    # Screen is still — has it been still long enough?
                    if (time.time() - self.stable_since) >= self.settle_delay:
                        # Dedup: only write frames distinct from the last file.
                        if (
                            self.last_saved_hash is None
                            or (current_hash - self.last_saved_hash)
                            > self.hash_threshold
                        ):
                            self.slide_index += 1
                            filename = f"slide_{self.slide_index:03d}.jpg"
                            filepath = self.slides_dir / filename
                            # Stamp the moment the frame is written. Taken
                            # before the (slow) JPEG encode so the offset
                            # reflects when the slide was actually on screen,
                            # not when encoding finished.
                            elapsed = time.time() - self.start_time
                            img.save(
                                filepath, format="JPEG", quality=85, optimize=True
                            )
                            self.last_saved_hash = current_hash
                            self.captured_slides.append(filepath)
                            self.slide_times[filename] = elapsed
                            self._persist_slide_times()
                            # Guard the log: cp1252 stdout (e.g. redirected
                            # Windows output) cannot encode the emoji and
                            # would kill the thread.
                            try:
                                print(
                                    "📸 Vision: Settled frame captured "
                                    f"({filename})"
                                )
                            except UnicodeEncodeError:
                                print(
                                    "Vision: Settled frame captured "
                                    f"({filename})"
                                )
                        # Aggressively reset after a settle cycle so a frame
                        # can only be saved once per period of stillness.
                        self.stable_since = time.time()

                self.last_frame_hash = current_hash

                # Poll wait; the event wakes it immediately on stop().
                self._stop_event.wait(timeout=self.poll_interval)

    def _persist_slide_times(self) -> None:
        """Write :attr:`slide_times` to the sidecar JSON, never fatally.

        Called from the capture thread after every save, so the timestamps
        survive an abrupt shutdown (the process being killed mid-lecture) and
        not just a clean ``stop()``. A write failure must not kill the capture
        thread: the slides themselves are already safely on disk, so losing
        the timing sidecar is a degradation, not a failure.
        """
        try:
            self.times_path.write_text(
                json.dumps(self.slide_times, indent=2, sort_keys=True),
                encoding="utf-8",
            )
        except OSError:
            pass

    def stop(self) -> list[Path]:
        """Stop the capture daemon and return the settled slides captured."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        # Final flush so a session that ends without a new save (e.g. the last
        # slide is the only one) still has a complete sidecar.
        self._persist_slide_times()
        return self.captured_slides