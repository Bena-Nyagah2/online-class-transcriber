"""Shared pytest configuration for the headless Phase 1/2 test suites."""

from __future__ import annotations

import sys
import types
import warnings
import importlib.util


def _stub_whisperx_if_absent() -> None:
    """Register a placeholder ``whisperx`` module when it is not installed.

    ``transcriber.py`` imports ``whisperx`` at module scope, and importing it
    for real pulls in torch, pyannote, and the CTranslate2 runtime — hundreds
    of megabytes that the suite is explicitly designed not to touch. Every
    test mocks ``transcriber.whisperx`` anyway, so an empty module is enough
    to satisfy the import.

    The real package is used whenever it is present, so a developer with a
    working install still exercises the true module object.
    """
    if importlib.util.find_spec("whisperx") is not None:
        return
    stub = types.ModuleType("whisperx")
    # Attribute access on a bare module raises; tests patch the whole object.
    stub.__getattr__ = lambda name: None  # type: ignore[attr-defined]
    sys.modules.setdefault("whisperx", stub)


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