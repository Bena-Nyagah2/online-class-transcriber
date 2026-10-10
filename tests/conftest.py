"""Shared pytest configuration for the headless Phase 1/2 test suites."""

from __future__ import annotations

import os
import sys
import types
import warnings
import importlib.util

import pytest

# ---------------------------------------------------------------------------
# Boot-time environment validation
# ---------------------------------------------------------------------------
# ``config.Settings()`` runs at import time and raises ``ValidationError`` if
# ``NVIDIA_API_KEY`` or ``HF_TOKEN`` are missing. Several test modules import
# ``server`` (which imports ``config``) or ``summarizer``/``transcriber``
# directly, so those variables must exist *before* the first import happens.
# We set them here -- the earliest hook pytest offers -- using ``setdefault``
# so a developer's real shell exports still win. The values are dummy; no test
# makes a real API call.
os.environ.setdefault("NVIDIA_API_KEY", "test-nvidia-key")
os.environ.setdefault("HF_TOKEN", "test-hf-token")


def _stub_whisperx_if_absent() -> None:
    """Register placeholder ``whisperx`` modules when the package is missing.

    ``transcriber.py`` imports ``whisperx`` *and* ``whisperx.diarize`` at
    module scope, and importing either for real pulls in torch, pyannote, and
    the CTranslate2 runtime — hundreds of megabytes that the suite is
    explicitly designed not to touch. Every test mocks what it needs anyway,
    so empty placeholders are enough to satisfy the imports.

    The stub is registered as a package (``__path__`` plus an explicit
    ``diarize`` attribute) so that ``from whisperx.diarize import ...``
    resolves without a real distribution on disk.

    The real package is used whenever it is present, so a developer with a
    working install still exercises the true modules.
    """
    if importlib.util.find_spec("whisperx") is not None:
        return

    whisperx_stub = types.ModuleType("whisperx")
    # Empty __path__ marks it as a package so submodule imports resolve.
    whisperx_stub.__path__ = []  # type: ignore[attr-defined]
    # Attribute access on a bare module raises; tests patch what they need.
    whisperx_stub.__getattr__ = lambda name: None  # type: ignore[attr-defined]

    diarize_stub = types.ModuleType("whisperx.diarize")
    diarize_stub.DiarizationPipeline = type(  # type: ignore[attr-defined]
        "DiarizationPipeline", (), {"__init__": lambda self, *a, **k: None}
    )
    diarize_stub.assign_word_speakers = (  # type: ignore[attr-defined]
        lambda *args, **kwargs: None
    )
    # Bind the child onto the parent so `import whisperx.diarize` resolves it.
    whisperx_stub.diarize = diarize_stub  # type: ignore[attr-defined]

    sys.modules.setdefault("whisperx", whisperx_stub)
    sys.modules.setdefault("whisperx.diarize", diarize_stub)


_stub_whisperx_if_absent()


def pytest_configure(config) -> None:  # noqa: ARG001 - standard pytest hook
    """Keep pydub's import-time "ffmpeg not found" warning out of test output.

    pydub discovers its audio encoder at import time and warns when ffmpeg
    is missing from PATH. The transcriber tests never invoke the encoder
    (the MP3 export is mocked), so the warning is pure noise here.
    """
    warnings.filterwarnings(
        "ignore",
        message=r"Couldn't find ffmpeg.*",
        category=RuntimeWarning,
    )
    config.addinivalue_line(
        "markers",
        "allow_capture_start: permit SlideCatcher.start() to spawn its real "
        "capture thread. Only for a test whose subject *is* the thread spawn; "
        "such a test must still inject a fake mss device.",
    )


@pytest.fixture(autouse=True)
def no_real_screen_capture(request, monkeypatch) -> None:
    """Make real screen capture impossible for the whole suite.

    ``SlideCatcher`` grabs the primary monitor through ``mss`` and writes the
    frames into the pipeline's output directory as JPEGs. That is the whole
    point of the feature, and it is also the most destructive thing a test can
    do by accident: a test that constructs a ``SlideCatcher`` and forgets to
    patch it photographs whatever is actually on the developer's screen -- open
    documents, chat windows, anything -- and those frames land in the lecture
    archive under ``output/``.

    **The guard is on ``start()``, not on ``mss``, and that placement is the
    whole design.** ``SlideCatcher.start`` spawns a *daemon* thread, so an
    exception raised deeper down is swallowed: the thread dies, the test sees an
    empty ``captured_slides`` list, and the test passes. Guarding ``mss.mss``
    alone prevents the screenshot but produces exactly the silent, confusing
    failure it was meant to replace. Blocking ``start()`` raises in the test's
    own thread, so pytest reports a named error pointing at the test that
    forgot to patch. ``mss.mss`` is still blocked as a second line of defence,
    for a grab reached without going through ``start`` at all.

    The opt-out marker exists for the one case that genuinely must spawn the
    real thread: a test whose subject is ``start()``'s thread-spawning
    behaviour. It is deliberately narrow, it is not something a test should
    reach for to make a failure go away, and it does not by itself permit a
    screenshot -- a marked test still has to inject a fake ``mss``.

    The import is deferred and import failures are swallowed: ``mss`` is a
    native dependency absent on some machines, and a guard that made collection
    depend on it would turn "this feature cannot be tested here" into "the whole
    suite errors before it runs".
    """
    try:
        import vision
    except Exception:  # noqa: BLE001 - a missing native dep must not break collection
        return

    if request.node.get_closest_marker("allow_capture_start"):
        return

    def _refuse(*args, **kwargs):
        raise RuntimeError(
            "A test tried to capture the real screen via SlideCatcher.start(). "
            "Inject a fake capture device instead -- e.g. "
            "mocker.patch('vision.mss.mss', return_value=<fake sct>) -- or drive "
            "catcher._run() directly. Without this, SlideCatcher photographs "
            "the developer's actual desktop and writes the frames into the "
            "lecture output directory."
        )

    monkeypatch.setattr(vision.SlideCatcher, "start", _refuse)

    mss_module = getattr(vision, "mss", None)
    if mss_module is not None:
        monkeypatch.setattr(mss_module, "mss", _refuse)