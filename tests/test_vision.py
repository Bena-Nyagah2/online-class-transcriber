"""Headless unit tests for ``vision.py`` (Phase 5) — the Saccade & Settle daemon.

The capture state machine is driven **synchronously** (``_run()`` called
directly on the calling thread — no real daemon thread, zero race conditions)
with everything that touches physical hardware mocked away:

* ``mss.mss`` yields a fake capture device whose ``grab`` returns a fixed
  4x4 BGRA frame, so no display or screen capture is ever accessed;
* ``PIL.Image.frombytes`` stays **real**, so the catcher's ``img.save``
  writes genuine JPEG files into ``tmp_path`` that can be asserted on disk;
* ``imagehash.phash`` is mocked to hand back controlled hashes built with
  ``imagehash.hex_to_hash`` (real ``ImageHash`` objects with numeric
  Hamming distances), so the motion / settle / dedup branches are hit
  deterministically;
* ``time.time`` is replayed frame-by-frame (fast-forwarded) to drive the
  ``settle_delay`` logic without sleeping.

The elapsed-offset tests set ``catcher.start_time`` directly, which is exactly
what ``start()`` does; ``test_start_stamps_the_elapsed_time_origin`` covers that
stamping itself.

All three original scenarios correspond one-to-one to the documented behaviours:
scroll = motion storm (0 slides), hold = settle (exactly 1 slide), and a
transition = a *second* slide once the new frame settles.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest
from imagehash import hex_to_hash

from vision import SLIDE_TIMES_FILENAME, SlideCatcher

# Two hashes 64 bits apart: every motion (>2) and dedup (>5) check triggers.
HASH_A = "0" * 16
HASH_B = "f" * 16

FRAME_SIZE = (4, 4)


class _FakeSCT:
    """Duck-typed stand-in for ``mss.mss`` — no screen access at all."""

    def __init__(self) -> None:
        # 4x4 image, 4 bytes/pixel BGRA ("BGRX" raw decode) -> 64 bytes.
        self.raw = bytes(4 * 4 * 4)
        self.monitors = [
            {"left": 0, "top": 0, "width": 4, "height": 4},  # virtual box
            {"left": 0, "top": 0, "width": 4, "height": 4},  # primary
        ]

    def __enter__(self) -> "_FakeSCT":
        return self

    def __exit__(self, *exc) -> bool:
        return False

    def grab(self, monitor):
        return SimpleNamespace(size=FRAME_SIZE, bgra=self.raw)


def _run(catcher: SlideCatcher, mocker, frames: list[tuple[float, str]]) -> None:
    """Drive one synchronous ``_run`` pass with a fully mocked environment.

    Args:
        catcher: The catcher under test (fresh per test).
        mocker: pytest-mock fixture.
        frames: ``(timestamp, phash_hex)`` pairs, one per simulated poll.
            The timestamp is the absolute wall-clock value *of that frame*
            — fast-forwarding past ``settle_delay`` is done by choosing a
            later timestamp for a later poll. It is installed just before the
            poll's motion/settle logic reads ``time.time()``, so both time
            reads in the settle branch (the ``>=`` check and the reset) see
            the same frame timestamp.
    """
    clock = {"now": 100.0}
    frame_iter = iter(frames)

    def _time() -> float:
        return clock["now"]

    def _phash(_img):
        timestamp, hex_str = next(frame_iter)
        clock["now"] = timestamp  # this poll is captured "at" this time
        return hex_to_hash(hex_str)

    mocker.patch("vision.time.time", side_effect=_time)
    mocker.patch("vision.imagehash.phash", side_effect=_phash)
    mocker.patch("vision.mss.mss", return_value=_FakeSCT())

    polls_left = len(frames)

    def _auto_stop(timeout=None) -> bool:
        # After the final simulated poll, terminate the loop cleanly.
        nonlocal polls_left
        polls_left -= 1
        if polls_left <= 0:
            catcher._stop_event.set()
        return True

    mocker.patch.object(catcher._stop_event, "wait", side_effect=_auto_stop)

    catcher._run()


# ---------------------------------------------------------------------------
# Test Case 1 — scrolling document: never settles, zero slides
# ---------------------------------------------------------------------------


def test_scrolling_document_saves_nothing(mocker, tmp_path) -> None:
    """Rapid high-delta frames keep resetting the settle stopwatch."""
    catcher = SlideCatcher(output_dir=tmp_path)

    # Three frames, all >2 pHash units apart, spread over <1 second.
    _run(
        catcher,
        mocker,
        frames=[
            (100.0, HASH_A),
            (100.3, HASH_B),
            (100.6, HASH_A),
        ],
    )

    assert catcher.captured_slides == []
    assert list(catcher.slides_dir.glob("*.jpg")) == []
    assert catcher.slide_index == 0


# ---------------------------------------------------------------------------
# Test Case 2 — settled slide: exactly one capture
# ---------------------------------------------------------------------------


def test_settled_slide_saves_exactly_once(mocker, tmp_path) -> None:
    """A frame held still past ``settle_delay`` is captured once."""
    catcher = SlideCatcher(output_dir=tmp_path, settle_delay=3.0)

    # Same frame repeated; the second poll arrives 3.1s later (> 3.0).
    _run(
        catcher,
        mocker,
        frames=[
            (100.0, HASH_A),
            (103.1, HASH_A),
        ],
    )

    assert len(catcher.captured_slides) == 1
    slide = catcher.captured_slides[0]
    assert slide.name == "slide_001.jpg"
    assert slide.exists()
    assert slide.read_bytes()[:2] == b"\xff\xd8"  # genuine JPEG, not a mock
    assert list(catcher.slides_dir.glob("*.jpg")) == [slide]


# ---------------------------------------------------------------------------
# Test Case 3 — slide transition: a second, distinct slide is captured
# ---------------------------------------------------------------------------


def test_transition_captures_second_slide(mocker, tmp_path) -> None:
    """A settled slide, a motion spike, then a new settled frame -> slide_002."""
    catcher = SlideCatcher(output_dir=tmp_path)

    # t=100.0  frame A          (first motion onset)
    # t=103.1  frame A repeated (settles 3.1s > 3.0 -> slide_001)
    # t=103.5  frame B          (drastic delta -> transition motion)
    # t=106.6  frame B repeated (settles again -> slide_002)
    _run(
        catcher,
        mocker,
        frames=[
            (100.0, HASH_A),
            (103.1, HASH_A),
            (103.5, HASH_B),
            (106.6, HASH_B),
        ],
    )

    assert [p.name for p in catcher.captured_slides] == [
        "slide_001.jpg",
        "slide_002.jpg",
    ]
    assert catcher.slide_index == 2
    assert all(p.exists() for p in catcher.captured_slides)
    assert all(p.read_bytes()[:2] == b"\xff\xd8" for p in catcher.captured_slides)


# ---------------------------------------------------------------------------
# Test Case 4 — capture timestamps (the time-domain signal Stage 3 consumes)
# ---------------------------------------------------------------------------


def test_slide_times_are_recorded_as_elapsed_offsets(mocker, tmp_path) -> None:
    """Each save records how long into the session the frame was captured.

    The offset is measured from ``start_time`` — the value ``start()`` stamps —
    so it lines up with the transcript's own elapsed timestamps. Here the
    origin is 100.0, matching the harness clock, so slide_001 lands 3.1s in and
    slide_002 6.6s in.
    """
    catcher = SlideCatcher(output_dir=tmp_path)
    catcher.start_time = 100.0  # what start() would stamp

    _run(
        catcher,
        mocker,
        frames=[
            (100.0, HASH_A),
            (103.1, HASH_A),  # -> slide_001 at t=103.1
            (103.5, HASH_B),
            (106.6, HASH_B),  # -> slide_002 at t=106.6
        ],
    )

    assert catcher.slide_times == pytest.approx(
        {"slide_001.jpg": 3.1, "slide_002.jpg": 6.6}
    )


def test_slide_times_sidecar_is_persisted_for_the_summarizer(mocker, tmp_path) -> None:
    """The sidecar lands on disk, keyed by filename, in the slides directory.

    ``_build_slide_manifest`` only ever receives the slides directory, so the
    timestamps have to survive as a file rather than live in memory.
    """
    catcher = SlideCatcher(output_dir=tmp_path)
    catcher.start_time = 100.0

    _run(
        catcher,
        mocker,
        frames=[(100.0, HASH_A), (104.0, HASH_A)],
    )

    sidecar = catcher.slides_dir / SLIDE_TIMES_FILENAME
    assert sidecar.is_file()
    assert json.loads(sidecar.read_text(encoding="utf-8")) == pytest.approx(
        {"slide_001.jpg": 4.0}
    )
    # The .json sidecar must not be mistaken for a slide by any *.jpg glob.
    assert [p.name for p in catcher.slides_dir.glob("*.jpg")] == ["slide_001.jpg"]


def test_stop_flushes_the_sidecar_even_without_a_new_save(mocker, tmp_path) -> None:
    """A session that captures nothing still leaves a valid (empty) sidecar.

    Writing it unconditionally means a downstream reader never has to
    distinguish "never captured" from "sidecar missing".
    """
    catcher = SlideCatcher(output_dir=tmp_path)
    sidecar = catcher.slides_dir / SLIDE_TIMES_FILENAME
    assert not sidecar.exists()

    catcher.stop()  # no start(), no _run() — must not raise

    assert json.loads(sidecar.read_text(encoding="utf-8")) == {}


def test_sidecar_write_failure_does_not_kill_the_capture(mocker, tmp_path) -> None:
    """An unwritable sidecar degrades timing, never the slides themselves.

    ``_persist_slide_times`` runs on the capture thread after every save. If it
    raised, the thread would die and the rest of the lecture would go
    unrecorded — a much worse outcome than losing the timing metadata.
    """
    catcher = SlideCatcher(output_dir=tmp_path)
    catcher.start_time = 100.0
    # Point the sidecar at a parent directory that does not exist, so
    # write_text raises FileNotFoundError (an OSError) on every attempt.
    catcher.times_path = tmp_path / "no_such_dir" / SLIDE_TIMES_FILENAME

    _run(catcher, mocker, frames=[(100.0, HASH_A), (104.0, HASH_A)])

    # The slide is still on disk and still tracked in memory.
    assert [p.name for p in catcher.captured_slides] == ["slide_001.jpg"]
    assert catcher.slide_times == pytest.approx({"slide_001.jpg": 4.0})
    assert catcher.captured_slides[0].exists()


def test_start_stamps_the_elapsed_time_origin(mocker, tmp_path) -> None:
    """``start()`` sets the clock origin, not ``__init__``.

    main.py constructs the catcher and starts it back-to-back, but a caller
    that builds a catcher early (or restarts one) must not accumulate dead
    time into every offset.
    """
    catcher = SlideCatcher(output_dir=tmp_path)
    assert catcher.start_time == 0.0  # unset until capture begins

    stamped = {"now": 500.0}
    mocker.patch("vision.time.time", side_effect=lambda: stamped["now"])
    mocker.patch("vision.mss.mss", return_value=_FakeSCT())
    # Replace the thread body so start() never touches a capture device.
    mocker.patch.object(catcher, "_run", lambda: stamped.setdefault("ran", True))

    catcher.start()
    catcher._thread.join(timeout=5.0)

    assert catcher.start_time == 500.0
    assert stamped.get("ran") is True