"""Shared pytest configuration for the headless Phase 1/2 test suites."""

from __future__ import annotations

import warnings


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