"""SQLite persistence layer (SQLAlchemy) for recorded lecture sessions.

A local, file-backed database that records every pipeline run: which course it
was, which domain profile shaped its notes, where the artifacts landed, and
whether the Google Drive upload succeeded. It is deliberately small -- a single
``lectures.db`` file beside the application, no server to run, no migration
tooling to remember.

Why SQLite: the workload is one writer (the pipeline) and a handful of readers
(the dashboard, an operator with a SQL prompt), all on the same machine. An
embedded database gives durable, queryable history with none of the operational
cost of a network database, and the whole file can be deleted without losing
anything that is not reproducible from the on-disk artifacts.

The engine uses ``check_same_thread=False`` because ``SessionLocal`` is handed
to FastAPI's threadpool (a synchronous ``def`` route runs on a worker thread
that is not the one that created the connection) and to the pipeline's own
background thread. SQLAlchemy's default pooling still serialises access on the
underlying connection, so this relaxes SQLite's per-thread guard without
introducing concurrent writes.
"""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine
from sqlalchemy.orm import Session, declarative_base, sessionmaker

# Relative to the process working directory, matching how the pipeline already
# treats ``./output``. The file is gitignored; it is local state, not source.
SQLALCHEMY_DATABASE_URL = "sqlite:///lectures.db"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    # A connection may be created on the import thread and then used by the
    # threadpool; SQLite would otherwise refuse it. Concurrent writes are still
    # serialised by the connection pool, so this is a threading guard, not a
    # concurrency free-for-all.
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()


def get_db() -> Iterator[Session]:
    """FastAPI dependency: yield a session, always closing it afterwards.

    Yields:
        A request-scoped :class:`~sqlalchemy.orm.Session`. The ``finally``
        guarantees the connection is returned to the pool even when the route
        raises, which is the difference between a transient 500 and a pool that
        slowly starves.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
