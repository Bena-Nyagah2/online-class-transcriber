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

import vision
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


class _RecordingSCT(_FakeSCT):
    """``_FakeSCT`` that records every successful grab box.

    ``fail_first`` makes the first N grabs raise, simulating a window tearing
    down mid-capture or a monitor switching resolution; the loop is expected to
    skip those frames and survive.
    """

    def __init__(self, fail_first: int = 0) -> None:
        super().__init__()
        self.grabbed: list[dict] = []
        self._fail_left = fail_first

    def grab(self, monitor):
        box = dict(monitor)
        if self._fail_left > 0:
            self._fail_left -= 1
            raise RuntimeError("simulated grab failure")
        self.grabbed.append(box)
        return super().grab(monitor)


def _window(
    title: str,
    visible: int = 1,
    minimized: bool = False,
    left: int = 0,
    top: int = 0,
    width: int = 800,
    height: int = 600,
) -> SimpleNamespace:
    """A duck-typed stand-in for a ``pygetwindow`` window object."""
    return SimpleNamespace(
        title=title,
        visible=visible,
        isMinimized=minimized,
        left=left,
        top=top,
        width=width,
        height=height,
    )


def _window_sequence(*results):
    """Return a ``getWindowsWithTitle`` fake replaying ``results``, last forever.

    One entry per poll: the first call yields ``results[0]``, the second
    ``results[1]``, and once ``results`` is exhausted the final entry repeats --
    which is how "the window restored mid-lecture" stays restored.
    """
    poll = {"i": 0}

    def _next(*_args, **_kwargs):
        item = results[min(poll["i"], len(results) - 1)]
        poll["i"] += 1
        return item

    return _next


def _run(
    catcher: SlideCatcher,
    mocker,
    frames: list[tuple[float, str]],
    sct_instance: _FakeSCT | None = None,
) -> None:
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
        sct_instance: Optional pre-built capture device, for tests that need to
            inspect the grab boxes or inject failures. Defaults to a fresh
            :class:`_FakeSCT`.
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
    mocker.patch(
        "vision.mss.mss",
        return_value=sct_instance if sct_instance is not None else _FakeSCT(),
    )

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


@pytest.mark.allow_capture_start
def test_start_stamps_the_elapsed_time_origin(mocker, tmp_path) -> None:
    """``start()`` sets the clock origin, not ``__init__``.

    main.py constructs the catcher and starts it back-to-back, but a caller
    that builds a catcher early (or restarts one) must not accumulate dead
    time into every offset.

    This is the one test that opts out of the suite-wide capture guard, because
    the thread spawn *is* the subject here. The opt-out only re-permits
    ``start()``; it grants no access to the display, and the two patches below
    are what keep that true -- the thread body is replaced outright, and the
    fake ``mss`` is the second layer in case that replacement is ever dropped.
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


# ---------------------------------------------------------------------------
# Test Case 5 — continuous-motion safety valve
# ---------------------------------------------------------------------------


def test_continuous_motion_force_captures_after_the_timeout(mocker, tmp_path) -> None:
    """A screen that never settles still yields slides.

    Every poll here is a motion spike, so the settle branch never runs. Without
    the safety valve the whole session would produce zero slides and the
    summarizer would have nothing visual to work with — the exact failure that
    appears when the lecturer shares a playing video or a live demo.
    """
    catcher = SlideCatcher(output_dir=tmp_path, settle_delay=3.0)
    catcher.start_time = 100.0

    # Alternating A/B is a delta of 64 every poll, so every poll is motion.
    _run(
        catcher,
        mocker,
        frames=[
            (100.0, HASH_A),
            (105.0, HASH_B),
            (110.0, HASH_A),
            (115.0, HASH_B),  # 15.0s of unbroken motion -> forced capture
            (120.0, HASH_A),
        ],
    )

    assert [p.name for p in catcher.captured_slides] == ["slide_001.jpg"]
    # Stamped at the moment of the forced grab, not at session start.
    assert catcher.slide_times["slide_001.jpg"] == pytest.approx(15.0)
    assert catcher.captured_slides[0].exists()


def test_forced_capture_logs_the_continuously_moving_reason(
    mocker, tmp_path, capsys
) -> None:
    """The log distinguishes a forced grab from a genuine settle.

    A forced frame is a mid-motion blur, not a slide the lecturer chose to
    show, and the operator reading the console needs to be able to tell which
    is which when they review the deck.
    """
    catcher = SlideCatcher(output_dir=tmp_path, settle_delay=3.0)
    catcher.start_time = 100.0

    _run(
        catcher,
        mocker,
        frames=[(100.0, HASH_A), (105.0, HASH_B), (110.0, HASH_A), (115.0, HASH_B)],
    )

    output = capsys.readouterr().out
    assert "Forced capture due to continuous motion (slide_001.jpg)" in output
    assert "Settled frame captured" not in output


def test_settled_capture_does_not_claim_continuous_motion(
    mocker, tmp_path, capsys
) -> None:
    """The ordinary settle path must not emit the forced-capture wording."""
    catcher = SlideCatcher(output_dir=tmp_path, settle_delay=3.0)
    catcher.start_time = 100.0

    _run(catcher, mocker, frames=[(100.0, HASH_A), (103.1, HASH_A)])

    output = capsys.readouterr().out
    assert "Settled frame captured (slide_001.jpg)" in output
    assert "continuous motion" not in output


def test_motion_timer_resets_when_the_screen_settles(mocker, tmp_path) -> None:
    """Settling clears the motion run, so the timeout cannot leak across it.

    Without the reset, a burst of motion early in the lecture would leave the
    timer primed and force a spurious capture on the very next still frame.
    """
    catcher = SlideCatcher(output_dir=tmp_path, settle_delay=3.0)
    catcher.start_time = 100.0

    _run(
        catcher,
        mocker,
        frames=[
            (100.0, HASH_A),  # motion onset -> motion_since = 100
            (104.0, HASH_B),  # motion, 4.0s in
            (108.0, HASH_B),  # still -> settled capture, motion run cleared
            (130.0, HASH_B),  # long gap; must NOT force-capture
        ],
    )

    assert catcher.motion_since == 0.0
    # Exactly the one settled slide; no forced capture after the 22s gap.
    assert [p.name for p in catcher.captured_slides] == ["slide_001.jpg"]


def test_timeout_does_not_re_fire_every_poll_after_a_forced_capture(
    mocker, tmp_path, capsys
) -> None:
    """One forced capture per timeout window, not one per poll.

    The dedup check would absorb most repeats, but the timer must still be
    rearmed unconditionally: otherwise a busy screen either floods the
    summarizer with near-duplicate slides or busy-loops the log.
    """
    catcher = SlideCatcher(output_dir=tmp_path, settle_delay=3.0)
    catcher.start_time = 100.0

    # Four polls inside one 15s window, all motion: 100, 105, 110, 115 forces
    # at 115, then 116..120 stay inside the rearmed window.
    _run(
        catcher,
        mocker,
        frames=[
            (100.0, HASH_A),
            (105.0, HASH_B),
            (110.0, HASH_A),
            (115.0, HASH_B),  # forced capture here
            (116.0, HASH_A),
            (117.0, HASH_B),
            (118.0, HASH_A),
        ],
    )

    forced_lines = [
        line
        for line in capsys.readouterr().out.splitlines()
        if "Forced capture" in line
    ]
    assert len(forced_lines) == 1
    assert [p.name for p in catcher.captured_slides] == ["slide_001.jpg"]


def test_forced_capture_still_honours_deduplication(mocker, tmp_path) -> None:
    """A near-static but jittering screen does not flood the deck.

    Frames drifting by 4 pHash units read as motion (threshold 2) but fall
    under the dedup threshold of 5. Saving them unconditionally would emit a
    slide every 15s for a screen that has not really changed — each one costing
    a vision API call in Stage 1.
    """
    catcher = SlideCatcher(output_dir=tmp_path, settle_delay=3.0)
    catcher.start_time = 100.0

    near = "000000000000000f"  # 4 bits from HASH_A: motion, but dedup-blocked
    _run(
        catcher,
        mocker,
        frames=[
            (100.0, HASH_A),
            (103.1, HASH_A),  # settles -> slide_001, dedup reference is HASH_A
            (108.1, near),  # delta 4 -> motion, motion_since = 108.1
            (113.1, HASH_A),  # delta 4 -> motion
            (118.1, near),  # delta 4 -> motion
            (123.1, HASH_A),  # 15.0s of motion -> forced, but dedup blocks
            (128.1, near),
            (133.1, HASH_A),
            (138.1, near),  # forced again, dedup blocks again
        ],
    )

    # Only the genuine settled slide was written.
    assert [p.name for p in catcher.captured_slides] == ["slide_001.jpg"]


def test_forced_capture_timeout_of_zero_disables_the_safety_valve(
    mocker, tmp_path
) -> None:
    """``0`` turns the valve off, restoring strict settle-only behaviour."""
    catcher = SlideCatcher(
        output_dir=tmp_path, settle_delay=3.0, forced_capture_timeout=0
    )
    catcher.start_time = 100.0

    _run(
        catcher,
        mocker,
        frames=[
            (100.0, HASH_A),
            (115.0, HASH_B),
            (130.0, HASH_A),
            (145.0, HASH_B),
        ],
    )

    assert catcher.captured_slides == []
    # The motion run is still tracked; only the forced capture is suppressed.
    assert catcher.motion_since == 100.0


def test_forced_capture_timeout_is_configurable(mocker, tmp_path) -> None:
    """A shorter timeout fires sooner, so the knob actually does something."""
    catcher = SlideCatcher(
        output_dir=tmp_path, settle_delay=3.0, forced_capture_timeout=4.0
    )
    catcher.start_time = 100.0

    _run(
        catcher,
        mocker,
        frames=[
            (100.0, HASH_A),
            (102.0, HASH_B),
            (104.0, HASH_A),  # 4.0s of motion -> forced under this timeout
        ],
    )

    assert [p.name for p in catcher.captured_slides] == ["slide_001.jpg"]
    assert catcher.slide_times["slide_001.jpg"] == pytest.approx(4.0)


def test_the_suite_guard_blocks_an_unpatched_capture(tmp_path) -> None:
    """An unpatched ``start()`` must fail loudly, in this thread.

    The failure this prevents is silent and delayed: a test that constructs a
    ``SlideCatcher`` without patching would otherwise screenshot the real
    display and write the frames into the lecture output directory, and nothing
    would look wrong until someone opened the archive and found their own
    desktop in it.

    Asserted on ``start()`` rather than on ``mss.mss`` because of *where* the
    failure lands. ``start()`` spawns a daemon thread; anything raised inside it
    is swallowed and the test would pass while proving nothing. Raising here, in
    the test's own thread, is what makes the mistake visible as a named error
    instead of an empty ``captured_slides`` list.
    """
    catcher = SlideCatcher(output_dir=tmp_path)

    with pytest.raises(RuntimeError, match="real screen"):
        catcher.start()

    assert catcher.captured_slides == []


def test_the_guard_also_blocks_a_bare_mss_grab() -> None:
    """``mss`` is blocked independently, for a grab that skips ``start()``."""
    with pytest.raises(RuntimeError, match="real screen"):
        vision.mss.mss()


@pytest.mark.allow_capture_start
def test_a_marked_test_may_drive_the_capture_loop(mocker, tmp_path) -> None:
    """The guard is a default, not a wall: a marked test can still capture.

    Without this, the guard could be "fixed" by deleting the tests that trip it
    and nothing would object.

    Both layers are visible here. The marker is what re-permits ``start()`` to
    spawn -- it cannot be substituted for by a patch, because the guard
    replaces the method itself. In exchange, the marker also declines to patch
    ``mss``, so this test must supply its own fake device; that is the trade,
    and it is why the marker is safe to hand out.
    """
    mocker.patch("vision.mss.mss", return_value=_FakeSCT())
    mocker.patch.object(SlideCatcher, "_run", lambda self: None)

    catcher = SlideCatcher(output_dir=tmp_path)
    catcher.start()
    catcher._thread.join(timeout=5.0)

    assert not catcher._thread.is_alive()


# ---------------------------------------------------------------------------
# Window enumeration (get_active_windows)
# ---------------------------------------------------------------------------


def test_get_active_windows_lists_visible_deduplicated_titles(mocker) -> None:
    """Only visible, upright windows with a real title reach the picker."""
    fake_gw = SimpleNamespace(
        getAllWindows=lambda: [
            _window("Slides - Chrome"),  # kept
            _window(""),  # empty title -> system internals, dropped
            _window("Hidden Proc", visible=0),  # hidden -> dropped
            _window("Minimized", minimized=True),  # dropped
            _window("Program Manager"),  # desktop shell -> dropped
            _window("Slides - Chrome"),  # duplicate -> deduplicated away
        ]
    )
    mocker.patch.object(vision, "gw", fake_gw)

    assert vision.get_active_windows() == ["Slides - Chrome"]


def test_get_active_windows_survives_a_dying_window(mocker) -> None:
    """A window torn down between enumeration and title read is skipped, not fatal."""
    dead = SimpleNamespace()
    fake_gw = SimpleNamespace(
        getAllWindows=lambda: [_window("Alive"), _window("Alive"), dead]
    )
    mocker.patch.object(vision, "gw", fake_gw)

    # `.title` on the dead object raises AttributeError -> skipped;
    # the duplicate "Alive" is the dedup exercised on the list.
    assert vision.get_active_windows() == ["Alive"]


def test_get_active_windows_degrades_to_empty_when_enumeration_fails(mocker) -> None:
    """No window server / headless box -> [], never an exception."""
    fake_gw = SimpleNamespace(getAllWindows=lambda: (_ for _ in ()).throw(RuntimeError))
    mocker.patch.object(vision, "gw", fake_gw)

    assert vision.get_active_windows() == []


# ---------------------------------------------------------------------------
# Targeted window capture (SlideCatcher(window_title=...))
# ---------------------------------------------------------------------------


def test_window_capture_grabs_the_windows_own_bounding_box(mocker, tmp_path) -> None:
    """A ``window_title`` narrows ``mss.grab`` to that window's rect."""
    catcher = SlideCatcher(output_dir=tmp_path, window_title="Lecture Slides")
    mocker.patch.object(
        vision,
        "gw",
        SimpleNamespace(
            getWindowsWithTitle=_window_sequence(
                [_window("Lecture Slides", left=100, top=50, width=800, height=600)]
            ),
            getAllWindows=lambda: [],
        ),
    )
    sct = _RecordingSCT()
    _run(
        catcher,
        mocker,
        frames=[(100.0, HASH_A), (101.0, HASH_A)],
        sct_instance=sct,
    )

    assert sct.grabbed == [
        {"left": 100, "top": 50, "width": 800, "height": 600},
        {"left": 100, "top": 50, "width": 800, "height": 600},
    ]


def test_window_capture_follows_a_moved_window_between_polls(mocker, tmp_path) -> None:
    """The rect is re-read every poll, so a dragged window stays framed."""
    catcher = SlideCatcher(output_dir=tmp_path, window_title="Lecture Slides")
    mocker.patch.object(
        vision,
        "gw",
        SimpleNamespace(
            getWindowsWithTitle=_window_sequence(
                [_window("Lecture Slides", left=10, top=10, width=400, height=300)],
                [_window("Lecture Slides", left=90, top=70, width=640, height=480)],
            ),
            getAllWindows=lambda: [],
        ),
    )
    sct = _RecordingSCT()
    _run(catcher, mocker, frames=[(100.0, HASH_A), (101.0, HASH_A)], sct_instance=sct)

    assert sct.grabbed[0] == {"left": 10, "top": 10, "width": 400, "height": 300}
    assert sct.grabbed[1] == {"left": 90, "top": 70, "width": 640, "height": 480}


def test_minimised_window_falls_back_to_fullscreen_then_resumes(
    mocker, tmp_path, capsys
) -> None:
    """Minimised -> primary monitor; restored -> window box again.

    The fallback is announced exactly once and the latch clears on recovery,
    so the resume is observable (and testable) rather than silent.
    """
    catcher = SlideCatcher(output_dir=tmp_path, window_title="Lecture Slides")
    mocker.patch.object(
        vision,
        "gw",
        SimpleNamespace(
            getWindowsWithTitle=_window_sequence(
                [_window("Lecture Slides", minimized=True)],  # minimised at start
                [_window("Lecture Slides", left=120, top=30, width=700, height=500)],
            ),
            getAllWindows=lambda: [],
        ),
    )
    sct = _RecordingSCT()
    _run(catcher, mocker, frames=[(100.0, HASH_A), (101.0, HASH_A)], sct_instance=sct)

    # Poll 1: window minimised -> emulated-primary-monitor box.
    assert sct.grabbed[0] == sct.monitors[1]
    # Poll 2: window restored -> its own box again.
    assert sct.grabbed[1] == {"left": 120, "top": 30, "width": 700, "height": 500}
    assert catcher._window_missing is False  # latch cleared on resume

    output = capsys.readouterr().out
    assert "unavailable" in output
    assert "Resumed tracking window" in output


def test_closed_window_stays_on_fullscreen_without_spamming(mocker, tmp_path, capsys) -> None:
    """A window that never comes back keeps capturing, announcing the fallback once."""
    catcher = SlideCatcher(output_dir=tmp_path, window_title="Lecture Slides")
    mocker.patch.object(
        vision,
        "gw",
        SimpleNamespace(
            getWindowsWithTitle=_window_sequence([]),
            getAllWindows=lambda: [],
        ),
    )
    sct = _RecordingSCT()
    _run(catcher, mocker, frames=[(100.0, HASH_A), (101.0, HASH_A)], sct_instance=sct)

    assert catcher._window_missing is True
    assert sct.grabbed == [sct.monitors[1], sct.monitors[1]]
    output = capsys.readouterr().out
    assert output.count("unavailable") == 1  # announced once per outage


def test_a_failed_grab_skips_the_frame_and_the_daemon_survives(
    mocker, tmp_path, capsys
) -> None:
    """A grab racing a window teardown costs one frame, not the whole lecture."""
    catcher = SlideCatcher(output_dir=tmp_path)
    sct = _RecordingSCT(fail_first=1)
    _run(
        catcher,
        mocker,
        frames=[(100.0, HASH_A), (101.0, HASH_A)],
        sct_instance=sct,
    )

    # The failed grab consumed a poll but never yielded a frame.
    assert sct.grabbed == [sct.monitors[1]]
    assert catcher._grab_failed is False  # latch reset once a grab succeeds
    assert "skipping" in capsys.readouterr().out