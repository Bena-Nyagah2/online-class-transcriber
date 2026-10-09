"""SQLAlchemy ORM models for the lecture persistence layer.

The schema is intentionally a flat ledger: one row per completed pipeline run.
It answers "what did we capture, for which domain, and did it reach Drive?"
without trying to model the transcript, the slides, or the notes themselves --
those live on disk, and duplicating them into SQLite would create a second
source of truth that can silently diverge from the files.
"""

from __future__ import annotations

from datetime import datetime, timezone

from sqlalchemy import Column, DateTime, Integer, String

from database import Base


def _utcnow() -> datetime:
    """Timezone-aware UTC now.

    ``datetime.utcnow`` is deprecated and returns a naive value; storing an
    aware timestamp keeps "when did this run happen?" unambiguous across the
    machines this file gets copied between.
    """
    return datetime.now(timezone.utc)


class Lecture(Base):
    """One recorded-and-synthesised lecture session.

    Attributes:
        id: Surrogate primary key.
        course_code: Sanitised course unit code, e.g. ``COURSE_101``.
        domain: Synthesis profile used, e.g. ``criminology`` / ``software``.
        session_date: ``YYYY-MM-DD`` the session belongs to.
        audio_path: Path to the captured ``raw_audio.wav``.
        md_path: Path to the generated study-notes Markdown.
        sync_status: Google Drive upload state: ``PENDING``, ``SUCCESS`` or
            ``FAILED``. The row is written ``PENDING`` *before* the upload so a
            crash mid-sync leaves a visible unfinished record.
        created_at: UTC timestamp the row was written.
    """

    __tablename__ = "lectures"

    id = Column(Integer, primary_key=True, index=True)
    course_code = Column(String, nullable=False, index=True)
    domain = Column(String, nullable=False, default="general")
    session_date = Column(String, nullable=False)
    audio_path = Column(String, nullable=True)
    md_path = Column(String, nullable=True)
    sync_status = Column(String, nullable=False, default="PENDING")
    created_at = Column(DateTime, nullable=False, default=_utcnow)
