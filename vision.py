"""Phase 5 — Visual screen capture with perceptual deduplication.

Periodically grabs the primary display from a daemon background thread,
computes a perceptual hash (``imagehash.phash``) of each frame, and saves a
JPEG only when the screen changes beyond a small threshold — long stretches
of a static lecture slide cost nothing while every genuine transition is
kept.
"""

from __future__ import annotations

import os
import threading
import time
from pathlib import Path
from typing import Optional

import imagehash
import mss
from PIL import Image


class SlideCatcher:
    """Thread-safe periodic screen capture with perceptual deduplication."""

    def __init__(
        self,
        output_dir: Path | str,
        interval_seconds: float = 10.0,
        hash_threshold: int = 5,
    ) -> None:
        """Configure the catcher and prepare the ``slides`` subdirectory.

        Args:
            output_dir: Session directory; slides land in ``<output_dir>/slides``.
            interval_seconds: Seconds between capture attempts.
            hash_threshold: Largest pHash distance still considered the same
                slide (lower = stricter deduplication).
        """
        self.output_dir = Path(output_dir)
        self.slides_dir = self.output_dir / "slides"
        os.makedirs(self.slides_dir, exist_ok=True)

        self.interval_seconds = interval_seconds
        self.hash_threshold = hash_threshold

        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_hash: Optional[imagehash.ImageHash] = None
        self.slide_index: int = 0
        self._started_at: Optional[float] = None
        # Append-only from the capture thread and read only after stop();
        # safe under the GIL, no extra locking needed.
        self.captured_slides: list[Path] = []

    # -- lifecycle ---------------------------------------------------------

    def start(self) -> None:
        """Spawn and start the daemon capture thread."""
        self._started_at = time.monotonic()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Capture loop: grab -> hash -> save-on-change -> interval wait.

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
                curr_hash = imagehash.phash(img)

                # First frame is always saved; later frames only when they
                # differ from the last saved slide by more than the threshold.
                if (
                    self.last_hash is None
                    or (curr_hash - self.last_hash) > self.hash_threshold
                ):
                    self.slide_index += 1
                    filename = f"slide_{self.slide_index:03d}.jpg"
                    filepath = self.slides_dir / filename
                    img.save(filepath, format="JPEG", quality=85, optimize=True)
                    self.last_hash = curr_hash
                    self.captured_slides.append(filepath)
                    # Guard the log: cp1252 stdout (e.g. redirected Windows
                    # output) cannot encode the camera emoji and would kill
                    # the thread, so fall back to plain ASCII.
                    try:
                        print(f"📸 Slide captured: {filename}")
                    except UnicodeEncodeError:
                        print(f"Slide captured: {filename}")

                # Interval wait; the event wakes it immediately on stop().
                self._stop_event.wait(timeout=self.interval_seconds)

    def stop(self) -> list[Path]:
        """Signal the capture thread to stop and return the saved slides."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)
        return self.captured_slides
