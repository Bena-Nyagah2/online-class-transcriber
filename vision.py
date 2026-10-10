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
   near-identical slide costs nothing;
4. **Continuous-motion safety valve** — a screen that never settles (playing
   video, a live demo, an endless scroll) would otherwise yield *zero* slides
   for the whole session. Once motion has run unbroken for
   ``forced_capture_timeout`` seconds the current frame is written anyway, so
   a busy screen still contributes evidence.

Every saved slide also records **when** it was captured, as an elapsed offset
from the start of the capture session, in ``slides/slide_times.json``. Stage 3
of the summarizer needs that offset to line a slide up against the audio
transcript: the transcript's turn headers and the manifest's slide headers must
be expressed on the same clock before a model can cross-reference them.

By default the **primary monitor** is captured. When a ``window_title`` is
passed instead the catcher tracks that window, re-reading its bounding box
every poll so a moved or resized window stays framed; a window that is
minimised or closed mid-lecture degrades to full-screen capture rather than
silently producing no slides. :func:`get_active_windows` enumerates the on-screen
windows for the dashboard's capture picker.
"""

from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from typing import Any

import imagehash
import mss
import pygetwindow as gw
from PIL import Image

SLIDE_TIMES_FILENAME = "slide_times.json"
"""Sidecar in the slides directory mapping each slide filename to its offset.

Raw seconds are stored rather than a preformatted string so the summarizer can
render them with the *same* clock helper the transcript uses. Preformatting here
would let the two clocks drift apart and break the comparison.
"""

_PRIMARY_MONITOR_INDEX = 1
"""Index of the primary monitor in an ``mss`` monitor list.

``monitors[0]`` is the virtual bounding box spanning every display; ``[1]`` is
the primary one, and has been the capture target since Phase 5.
"""

_SYSTEM_WINDOW_TITLES = frozenset({"Program Manager"})
"""Shell windows that are visible and titled but never a lecture surface.

The Windows desktop shell reports itself as an ordinary visible window named
"Program Manager"; without this it would sit in the dashboard's picker next to
the actual slides. Kept tiny and explicit on purpose -- a broad denylist would
eventually hide a real lecture window.
"""


def _is_captureable_window(window: Any) -> bool:
    """Whether ``window`` is visible, upright, and not the shell.

    Every attribute read is guarded: a window can be torn down between the
    enumeration that yielded it and this call, and a dying handle raises rather
    than returning a sentinel. A window that cannot be inspected is treated as
    uncaptureable, never as an error.
    """
    try:
        return bool(window.visible) and not window.isMinimized
    except Exception:  # noqa: BLE001 - a dead handle must not kill the picker
        return False


def get_active_windows() -> list[str]:
    """Enumerate on-screen window titles for the dashboard's capture picker.

    Returns the deduplicated titles of every top-level window that is currently
    visible and not minimised, in enumeration order, with empty titles and the
    desktop shell filtered out. Only titles cross this boundary -- window
    handles are not stable across a session and process ids mean nothing to the
    operator choosing a target.

    Enumeration is best-effort: a platform where ``pygetwindow`` cannot reach a
    window server yields an empty list rather than raising, so the endpoint
    degrades to "nothing to pick" instead of a 500.
    """
    try:
        windows = gw.getAllWindows()
    except Exception:  # noqa: BLE001 - no display / no window-server access
        return []

    titles: list[str] = []
    seen: set[str] = set()
    for window in windows:
        try:
            title = (window.title or "").strip()
        except Exception:  # noqa: BLE001 - a dying window cannot be titled
            continue
        if not title or title in _SYSTEM_WINDOW_TITLES:
            continue
        if title in seen or not _is_captureable_window(window):
            continue
        seen.add(title)
        titles.append(title)
    return titles


class SlideCatcher:
    """Captures settled, distinct lecture slides while ignoring scrolling."""

    def __init__(
        self,
        output_dir: Path | str,
        poll_interval: float = 1.0,
        settle_delay: float = 3.0,
        hash_threshold: int = 5,
        forced_capture_timeout: float = 15.0,
        window_title: str | None = None,
    ) -> None:
        """Configure the catcher and prepare the ``slides`` subdirectory.

        Args:
            output_dir: Session directory; slides land in ``<output_dir>/slides``.
            poll_interval: Seconds between screen grabs.
            settle_delay: Seconds the screen must stay still before its
                frame counts as settled.
            hash_threshold: Largest pHash distance still considered the same
                slide (lower = stricter deduplication).
            forced_capture_timeout: Seconds of unbroken motion after which a
                frame is captured anyway, even though the screen never settled.
                Set to ``0`` to disable the safety valve entirely.
            window_title: Title of the window whose bounding box is captured.
                ``None`` (the default) captures the primary monitor. When the
                window is minimised or closed mid-lecture, capture falls back
                to the primary monitor until the window is usable again.
        """
        self.output_dir = Path(output_dir)
        self.slides_dir = self.output_dir / "slides"
        os.makedirs(self.slides_dir, exist_ok=True)
        self.times_path = self.slides_dir / SLIDE_TIMES_FILENAME

        self.poll_interval = poll_interval
        self.settle_delay = settle_delay
        self.hash_threshold = hash_threshold
        self.forced_capture_timeout = forced_capture_timeout
        self.window_title = window_title

        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Origin for every elapsed offset below. Set in start() so the clock
        # begins when capture does, not at construction.
        self.start_time: float = 0.0

        # True while a tracked window is minimised/closed and capture has
        # fallen back to the primary monitor; a latch so the transition is
        # announced once rather than once per poll.
        self._window_missing = False
        # Same latch for a run of failed grabs (see _announce_grab_failed).
        self._grab_failed = False

        # pHash of the most recent frame (drives the motion check).
        self.last_frame_hash: imagehash.ImageHash | None = None
        # pHash of the last file actually written (drives deduplication).
        self.last_saved_hash: imagehash.ImageHash | None = None
        # Wall-clock timestamp of the last detected motion.
        self.stable_since: float = 0.0
        # When the current unbroken run of motion began; 0.0 means the screen
        # is currently considered still. Drives forced_capture_timeout.
        self.motion_since: float = 0.0
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
        if self.window_title:
            self._announce(f"Tracking window {self.window_title!r}")
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        """Capture loop: grab -> motion/settle state machine -> poll wait.

        ``mss`` is instantiated inside the thread because COM/thread
        affinity on Windows breaks if it is created elsewhere.

        The grab box is re-resolved every poll: when tracking a ``window_title``
        the window's current rect is looked up fresh, so the box follows a moved
        or resized window, while a window that is minimised or closed falls back
        to the primary monitor -- announced once per outage -- so an alt-tabbed
        presenter still produces slides instead of a silent gap. A grab that
        fails outright (window teardown racing our rect read, monitor unplug,
        resolution switch) skips that frame and retries next poll.
        """
        with mss.mss() as sct:
            while not self._stop_event.is_set():
                box = self._resolve_grab_box(sct)
                try:
                    raw_frame = sct.grab(box)
                except Exception:  # noqa: BLE001 - see _announce_grab_failed
                    # Dropping one frame costs nothing; letting the exception
                    # kill the daemon thread costs the rest of the lecture.
                    self._announce_grab_failed()
                    self._stop_event.wait(timeout=self.poll_interval)
                    continue

                self._grab_failed = False
                img = Image.frombytes(
                    "RGB", raw_frame.size, raw_frame.bgra, "raw", "BGRX"
                )
                current_hash = imagehash.phash(img)
                now = time.time()

                # Motion check: a large delta means the screen is still
                # scrolling/playing, so reset the settle stopwatch.
                if (
                    self.last_frame_hash is None
                    or (current_hash - self.last_frame_hash) > 2
                ):
                    self.stable_since = now
                    # Start (or continue) the unbroken-motion run. Sentinel 0.0
                    # means "not currently in motion", so only the first frame
                    # of a run stamps it.
                    if self.motion_since == 0.0:
                        self.motion_since = now
                else:
                    # Screen is still — has it been still long enough?
                    if (now - self.stable_since) >= self.settle_delay:
                        # Dedup: only write frames distinct from the last file.
                        if (
                            self.last_saved_hash is None
                            or (current_hash - self.last_saved_hash)
                            > self.hash_threshold
                        ):
                            self._save_frame(img, current_hash, now)
                        # Aggressively reset after a settle cycle so a frame
                        # can only be saved once per period of stillness.
                        self.stable_since = now
                    # The screen is still, so any motion run is over.
                    self.motion_since = 0.0

                # Safety valve: a screen that never settles (playing video,
                # continuous scroll, a live demo) would otherwise produce zero
                # slides for the entire session. After the timeout, take the
                # frame we have rather than nothing, then start the clock
                # over. This deliberately bypasses the settle requirement but
                # NOT deduplication — see _save_frame's docstring.
                if (
                    self.forced_capture_timeout
                    and self.motion_since
                    and (now - self.motion_since) >= self.forced_capture_timeout
                ):
                    if (
                        self.last_saved_hash is None
                        or (current_hash - self.last_saved_hash)
                        > self.hash_threshold
                    ):
                        self._save_frame(img, current_hash, now, forced=True)
                    # Reset either way: a forced capture that deduped away
                    # still consumed this window, and letting the timeout
                    # re-fire every poll would busy-loop the log.
                    self.motion_since = now

                self.last_frame_hash = current_hash

                # Poll wait; the event wakes it immediately on stop().
                self._stop_event.wait(timeout=self.poll_interval)

    def _resolve_grab_box(self, sct: Any) -> dict[str, int]:
        """Return the ``mss.grab`` rectangle for the next frame.

        Without a ``window_title`` this is the primary monitor, exactly as
        before. With one, the target's *current* bounding box is looked up
        fresh every poll, so following a window while it is dragged or resized
        is free of any cached rect to invalidate. If the window is minimised,
        closed, or otherwise unavailable, the primary monitor is captured
        instead (announced once per outage) rather than letting the slide
        stream go silent.
        """
        if self.window_title is None:
            return sct.monitors[_PRIMARY_MONITOR_INDEX]

        window = self._find_tracked_window()
        if window is not None:
            if self._window_missing:
                self._window_missing = False
                self._announce(f"Resumed tracking window {self.window_title!r}")
            return {
                "left": window.left,
                "top": window.top,
                "width": window.width,
                "height": window.height,
            }

        if not self._window_missing:
            self._window_missing = True
            self._announce(
                f"Target window {self.window_title!r} unavailable "
                "(minimised or closed); captured the primary monitor instead"
            )
        return sct.monitors[_PRIMARY_MONITOR_INDEX]

    def _find_tracked_window(self) -> Any | None:
        """Return the live window matching :attr:`window_title`, or ``None``.

        ``pygetwindow`` matches titles case-insensitively as substrings, and
        the picker hands back exact titles from :func:`get_active_windows`, so
        in practice this is a 1:1 match. Each candidate is re-checked here
        rather than trusted from enumeration: a handle that was valid a moment
        ago can be gone by the time its rect is read.
        """
        if self.window_title is None:
            return None
        try:
            candidates = gw.getWindowsWithTitle(self.window_title)
        except Exception:  # noqa: BLE001 - enumeration is best-effort
            return None

        for window in candidates:
            try:
                if not window.title:
                    continue
                if not _is_captureable_window(window):
                    continue
                if window.width <= 0 or window.height <= 0:
                    continue
                if window.left is None or window.top is None:
                    continue
                return window
            except Exception:  # noqa: BLE001 - skip a window mid-teardown
                continue
        return None

    def _announce(self, message: str) -> None:
        """Print a capture-lifecycle message, never fatally.

        Shares :meth:`_save_frame`'s encoding guard: cp1252 stdout cannot
        encode the camera emoji, and raising here would kill the capture thread.
        """
        try:
            print(f"📸 Vision: {message}")
        except UnicodeEncodeError:
            print(f"Vision: {message}")

    def _announce_grab_failed(self) -> None:
        """Log a failed grab once per outage, not once per poll."""
        if not self._grab_failed:
            self._grab_failed = True
            self._announce("Frame grab failed; skipping until the next poll")

    def _save_frame(
        self, img: Image.Image, current_hash: Any, now: float, forced: bool = False
    ) -> None:
        """Write one settled (or forced) frame and record its capture offset.

        Shared by the settle path and the continuous-motion safety valve so the
        two cannot drift apart in how they name, stamp, persist, or log a
        slide. Callers are responsible for having already cleared the
        deduplication check.

        ``forced`` selects the log line only. A forced capture still honours
        deduplication upstream: during genuinely continuous motion consecutive
        frames are genuinely different, so dedup costs nothing there, while
        honouring it stops a near-static screen (a blinking cursor, a
        chatty UI) from emitting a near-duplicate slide every timeout window
        and flooding the summarizer with vision API calls.

        Args:
            img: The decoded PIL frame to encode.
            current_hash: pHash of ``img``, cached as the dedup reference.
            now: Wall-clock value stamped as the capture instant.
            forced: True when triggered by the motion timeout rather than a
                genuine settle.
        """
        self.slide_index += 1
        filename = f"slide_{self.slide_index:03d}.jpg"
        filepath = self.slides_dir / filename
        # Stamp the moment the frame is written. Taken before the (slow) JPEG
        # encode so the offset reflects when the slide was actually on screen,
        # not when encoding finished.
        elapsed = now - self.start_time
        img.save(filepath, format="JPEG", quality=85, optimize=True)
        self.last_saved_hash = current_hash
        self.captured_slides.append(filepath)
        self.slide_times[filename] = elapsed
        self._persist_slide_times()

        message = (
            f"Forced capture due to continuous motion ({filename})"
            if forced
            else f"Settled frame captured ({filename})"
        )
        # Guard the log: cp1252 stdout (e.g. redirected Windows output) cannot
        # encode the emoji and would kill the thread.
        try:
            print(f"📸 Vision: {message}")
        except UnicodeEncodeError:
            print(f"Vision: {message}")

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