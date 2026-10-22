import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .schemas import JobSummary, PageResult


def now() -> str:
    return datetime.now(UTC).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connect(self):
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys=ON")
        connection.execute("PRAGMA busy_timeout=30000")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    filename TEXT NOT NULL,
                    sha256 TEXT NOT NULL,
                    status TEXT NOT NULL,
                    page_count INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    error TEXT,
                    engine_json TEXT NOT NULL DEFAULT '{}',
                    config_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pages (
                    job_id TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,
                    number INTEGER NOT NULL,
                    status TEXT NOT NULL DEFAULT 'pending',
                    attempts INTEGER NOT NULL DEFAULT 0,
                    result_json TEXT,
                    PRIMARY KEY(job_id, number)
                );
                CREATE INDEX IF NOT EXISTS jobs_queue ON jobs(status, created_at);
                CREATE TABLE IF NOT EXISTS worker (
                    id INTEGER PRIMARY KEY CHECK(id=1),
                    state TEXT NOT NULL,
                    heartbeat TEXT NOT NULL,
                    job_id TEXT,
                    error TEXT
                );
            """)

    def create_job(self, job_id: str, filename: str, digest: str, pages: int, config: dict) -> None:
        timestamp = now()
        with self.connect() as db:
            db.execute(
                "INSERT INTO jobs "
                "(id,filename,sha256,status,page_count,created_at,updated_at,config_json) "
                "VALUES (?,?,?,'queued',?,?,?,?)",
                (job_id, filename, digest, pages, timestamp, timestamp, json.dumps(config)),
            )
            db.executemany(
                "INSERT INTO pages(job_id,number) VALUES (?,?)",
                [(job_id, number) for number in range(1, pages + 1)],
            )

    def get_job(self, job_id: str) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            return dict(row) if row else None

    def summary(self, job_id: str) -> JobSummary:
        with self.connect() as db:
            row = db.execute(
                """SELECT j.*, SUM(p.status IN ('completed','failed')) AS processed_pages,
                    SUM(p.status='completed') AS successful_pages,
                    SUM(p.status='failed') AS failed_pages,
                    SUM(CASE WHEN p.result_json IS NOT NULL THEN
                        json_extract(p.result_json,'$.needs_review') ELSE 0 END) AS review_pages
                    FROM jobs j JOIN pages p ON p.job_id=j.id WHERE j.id=? GROUP BY j.id""",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            return JobSummary.model_validate(dict(row))

    def recent(self, limit: int = 30) -> list[JobSummary]:
        with self.connect() as db:
            ids = db.execute(
                "SELECT id FROM jobs ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [self.summary(row["id"]) for row in ids]

    def pages(self, job_id: str) -> list[PageResult]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT * FROM pages WHERE job_id=? ORDER BY number", (job_id,)
            ).fetchall()
        return [
            PageResult.model_validate_json(row["result_json"])
            if row["result_json"]
            else PageResult(number=row["number"], status=row["status"], attempts=row["attempts"])
            for row in rows
        ]

    def claim_job(self) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' ORDER BY created_at LIMIT 1"
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET status='processing',updated_at=? WHERE id=?", (now(), row["id"])
            )
            return dict(row)

    def start_page(self, job_id: str, number: int) -> int:
        with self.connect() as db:
            db.execute(
                "UPDATE pages SET status='processing',attempts=attempts+1 "
                "WHERE job_id=? AND number=?",
                (job_id, number),
            )
            row = db.execute(
                "SELECT attempts FROM pages WHERE job_id=? AND number=?", (job_id, number)
            ).fetchone()
            return row["attempts"]

    def save_page(self, job_id: str, page: PageResult) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE pages SET status=?,result_json=? WHERE job_id=? AND number=?",
                (page.status, page.model_dump_json(), job_id, page.number),
            )
            db.execute("UPDATE jobs SET updated_at=? WHERE id=?", (now(), job_id))

    def set_engine(self, job_id: str, metadata: dict) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET engine_json=? WHERE id=?", (json.dumps(metadata), job_id))

    def finish_job(self, job_id: str) -> str:
        summary = self.summary(job_id)
        if summary.processed_pages != summary.page_count:
            raise ValueError("Cannot finish a document with unfinished pages")
        status = (
            "completed"
            if summary.failed_pages == 0
            else "partial"
            if summary.successful_pages
            else "failed"
        )
        error = "Не удалось обработать ни одной страницы" if status == "failed" else None
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET status=?,error=?,updated_at=? WHERE id=?",
                (status, error, now(), job_id),
            )
        return status

    def recover(self, max_attempts: int) -> None:
        # Only the process holding the exclusive worker lock may call this.
        with self.connect() as db:
            rows = db.execute("SELECT * FROM pages WHERE status='processing'").fetchall()
            for row in rows:
                if row["attempts"] >= max_attempts:
                    page = PageResult(
                        number=row["number"],
                        status="failed",
                        attempts=row["attempts"],
                        error="Worker завершился во время обработки страницы",
                        needs_review=True,
                    )
                    db.execute(
                        "UPDATE pages SET status='failed',result_json=? "
                        "WHERE job_id=? AND number=?",
                        (page.model_dump_json(), row["job_id"], row["number"]),
                    )
                else:
                    db.execute(
                        "UPDATE pages SET status='pending' WHERE job_id=? AND number=?",
                        (row["job_id"], row["number"]),
                    )
            db.execute(
                "UPDATE jobs SET status='queued',updated_at=? WHERE status='processing'", (now(),)
            )

    def heartbeat(self, state: str, job_id: str | None = None, error: str | None = None) -> None:
        with self.connect() as db:
            db.execute(
                "INSERT INTO worker(id,state,heartbeat,job_id,error) VALUES(1,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET state=excluded.state,heartbeat=excluded.heartbeat,"
                "job_id=excluded.job_id,error=excluded.error",
                (state, now(), job_id, error),
            )

    def worker(self) -> dict | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM worker WHERE id=1").fetchone()
            return dict(row) if row else None
