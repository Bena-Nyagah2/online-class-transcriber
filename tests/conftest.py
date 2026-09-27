"""Shared pytest configuration for the headless Phase 1/2 test suites."""

from __future__ import annotations

import sys
import types
import warnings
import importlib.util


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