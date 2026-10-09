"""Headless unit tests for the SQLite persistence layer (``database``/``models``).

Offline and isolated: every test builds its own in-memory engine, so the suite
never touches the real ``lectures.db`` and never depends on the order the tests
run in. The pipeline's *integration* with this layer is tested in
``test_main.py``, where the pipeline fixtures live.
"""

from __future__ import annotations

from datetime import datetime, timezone
from unittest import mock

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from sqlalchemy.pool import StaticPool

import database
import models


@pytest.fixture
def db_session():
    """A session over a private in-memory database with the schema applied.

    ``StaticPool`` is required, not incidental: SQLite's ``:memory:`` database
    lives per-connection, so the default pool would hand the session a fresh,
    empty database on every checkout and the tables would vanish between
    ``commit`` and ``refresh``.
    """
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    models.Base.metadata.create_all(bind=engine)
    factory = sessionmaker(autocommit=False, autoflush=False, bind=engine)
    session = factory()
    try:
        yield session
    finally:
        session.close()


def test_the_schema_exposes_every_required_column() -> None:
    """The ledger's column set is the contract; a dropped path column would
    only surface when someone tried to find the audio for an old row."""
    columns = set(models.Lecture.__table__.columns.keys())
    assert {
        "id",
        "course_code",
        "domain",
        "session_date",
        "audio_path",
        "md_path",
        "sync_status",
        "created_at",
    } <= columns


def test_a_new_lecture_defaults_to_pending(db_session) -> None:
    """A row is written PENDING before the upload, never left status-less."""
    lecture = models.Lecture(
        course_code="COURSE_101",
        domain="software",
        session_date="2026-10-09",
        audio_path="output/COURSE_101_2026-10-09/raw_audio.wav",
        md_path="output/COURSE_101_2026-10-09/COURSE_101_2026-10-09.md",
    )
    db_session.add(lecture)
    db_session.commit()
    db_session.refresh(lecture)

    assert lecture.id is not None
    assert lecture.sync_status == "PENDING"
    assert lecture.created_at is not None


def test_created_at_records_utc_now(db_session) -> None:
    """The timestamp must land inside the window the row was written in.

    SQLite has no native timezone type, so it round-trips the value without
    ``tzinfo``; the assertion normalises that rather than pretending the
    database stores awareness it cannot.
    """
    before = datetime.now(timezone.utc)
    lecture = models.Lecture(course_code="COURSE_101", session_date="2026-10-09")
    db_session.add(lecture)
    db_session.commit()
    db_session.refresh(lecture)
    after = datetime.now(timezone.utc)

    created = lecture.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=timezone.utc)

    assert before <= created <= after


def test_sync_status_can_be_updated_after_a_sync(db_session) -> None:
    """The happy path the pipeline relies on: PENDING in, SUCCESS out."""
    lecture = models.Lecture(course_code="COURSE_101", session_date="2026-10-09")
    db_session.add(lecture)
    db_session.commit()
    db_session.refresh(lecture)

    lecture.sync_status = "SUCCESS"
    db_session.commit()
    db_session.refresh(lecture)

    assert lecture.sync_status == "SUCCESS"


def test_get_db_closes_the_session_even_when_the_body_raises(monkeypatch) -> None:
    """The dependency's ``finally`` is what stops the pool starving.

    ``SessionLocal`` is replaced with a sentinel because the point under test
    is the generator's control flow, not SQLAlchemy's: the session must be
    closed on every exit path, including the one where the route raised.
    """
    sentinel = mock.MagicMock()
    monkeypatch.setattr(database, "SessionLocal", lambda: sentinel)

    gen = database.get_db()
    assert next(gen) is sentinel

    # Simulate a route raising into the generator.
    with pytest.raises(RuntimeError):
        gen.throw(RuntimeError("route exploded"))

    sentinel.close.assert_called_once()


def test_get_db_closes_the_session_on_normal_exit(monkeypatch) -> None:
    sentinel = mock.MagicMock()
    monkeypatch.setattr(database, "SessionLocal", lambda: sentinel)

    gen = database.get_db()
    assert next(gen) is sentinel
    gen.close()

    sentinel.close.assert_called_once()
