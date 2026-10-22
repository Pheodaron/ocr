import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path

from .backends import configuration_details
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
                CREATE TABLE IF NOT EXISTS workers (
                    backend TEXT PRIMARY KEY,
                    state TEXT NOT NULL,
                    heartbeat TEXT NOT NULL,
                    job_id TEXT,
                    error TEXT
                );
            """)
            db.execute("BEGIN IMMEDIATE")
            columns = {row["name"] for row in db.execute("PRAGMA table_info(jobs)")}
            for name, definition in {
                "backend": "TEXT NOT NULL DEFAULT 'paddle'",
                "comparison_id": "TEXT",
                "processing_seconds": "REAL NOT NULL DEFAULT 0",
                "model_load_seconds": "REAL NOT NULL DEFAULT 0",
                "run_attempts": "INTEGER NOT NULL DEFAULT 0",
                "delete_requested": "INTEGER NOT NULL DEFAULT 0",
            }.items():
                if name not in columns:
                    db.execute(f"ALTER TABLE jobs ADD COLUMN {name} {definition}")
            db.execute("CREATE INDEX IF NOT EXISTS jobs_comparison ON jobs(comparison_id)")

    def create_job(
        self,
        job_id: str,
        filename: str,
        digest: str,
        pages: int,
        config: dict,
        backend: str = "paddle",
        comparison_id: str | None = None,
    ) -> None:
        self.create_jobs(
            [
                {
                    "id": job_id,
                    "filename": filename,
                    "digest": digest,
                    "pages": pages,
                    "config": config,
                    "backend": backend,
                    "comparison_id": comparison_id,
                }
            ]
        )

    def create_jobs(self, jobs: list[dict]) -> None:
        # Publish a complete comparison atomically; the worker cannot claim half a batch.
        with self.connect() as db:
            for job in jobs:
                timestamp = now()
                db.execute(
                    "INSERT INTO jobs "
                    "(id,filename,sha256,status,page_count,created_at,updated_at,config_json,"
                    "backend,comparison_id) VALUES (?,?,?,'queued',?,?,?,?,?,?)",
                    (
                        job["id"],
                        job["filename"],
                        job["digest"],
                        job["pages"],
                        timestamp,
                        timestamp,
                        json.dumps(job["config"]),
                        job["backend"],
                        job["comparison_id"],
                    ),
                )
                db.executemany(
                    "INSERT INTO pages(job_id,number) VALUES (?,?)",
                    [(job["id"], number) for number in range(1, job["pages"] + 1)],
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
                    SUM(p.status='cancelled') AS cancelled_pages,
                    SUM(CASE WHEN p.result_json IS NOT NULL THEN
                        json_extract(p.result_json,'$.needs_review') ELSE 0 END) AS review_pages
                    FROM jobs j JOIN pages p ON p.job_id=j.id WHERE j.id=? GROUP BY j.id""",
                (job_id,),
            ).fetchone()
            if row is None:
                raise KeyError(job_id)
            record = dict(row)
            record["configuration"] = configuration_details(
                record["backend"], json.loads(record["config_json"])
            )
            return JobSummary.model_validate(record)

    def recent(self, limit: int = 30) -> list[JobSummary]:
        with self.connect() as db:
            ids = db.execute(
                "SELECT id FROM jobs WHERE delete_requested=0 ORDER BY created_at DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return self._existing_summaries(ids)

    def comparison_jobs(self, comparison_id: str) -> list[JobSummary]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT id FROM jobs WHERE comparison_id=? AND delete_requested=0 "
                "ORDER BY created_at",
                (comparison_id,),
            ).fetchall()
        return self._existing_summaries(rows)

    def _existing_summaries(self, rows) -> list[JobSummary]:
        summaries = []
        for row in rows:
            try:
                summaries.append(self.summary(row["id"]))
            except KeyError:
                # A concurrent DELETE can remove an ID after the list query.
                continue
        return summaries

    def delete_job(self, job_id: str) -> None:
        with self.connect() as db:
            db.execute("DELETE FROM jobs WHERE id=?", (job_id,))

    def request_cancel(self, job_id: str, *, delete: bool = False) -> None:
        # Serialize with claim/finish: a queued job must never start after this request.
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            status = row["status"]
            if status in {"queued", "processing"}:
                status = "cancelling"
            db.execute(
                "UPDATE jobs SET status=?,delete_requested=MAX(delete_requested,?),updated_at=? "
                "WHERE id=?",
                (status, int(delete), now(), job_id),
            )

    def cancellation_requested(self, job_id: str) -> bool:
        job = self.get_job(job_id)
        return (
            job is None
            or job["status"] in {"cancelling", "cancelled"}
            or bool(job["delete_requested"])
        )

    def pending_controls(self, backend: str) -> list[dict]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT * FROM jobs WHERE backend=? "
                    "AND (status='cancelling' OR delete_requested=1)",
                    (backend,),
                )
            ]

    def finalize_cancel(self, job_id: str) -> None:
        # Called only after the inference process has stopped (or before it was claimed).
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            job = db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone()
            if job is None or job["status"] != "cancelling":
                return
            rows = db.execute(
                "SELECT * FROM pages WHERE job_id=? "
                "AND status NOT IN ('completed','failed','cancelled')",
                (job_id,),
            ).fetchall()
            for row in rows:
                page = PageResult(
                    number=row["number"],
                    status="cancelled",
                    attempts=row["attempts"],
                    error="Распознавание остановлено пользователем",
                )
                db.execute(
                    "UPDATE pages SET status='cancelled',result_json=? WHERE job_id=? AND number=?",
                    (page.model_dump_json(), job_id, row["number"]),
                )
            db.execute(
                "UPDATE jobs SET status='cancelled',error=?,updated_at=? WHERE id=?",
                ("Распознавание остановлено пользователем", now(), job_id),
            )

    def finalize_unclaimed_cancel(self, job_id: str) -> None:
        # A queued job has no inference child, so no live worker is needed to cancel it.
        job = self.get_job(job_id)
        if job and job["status"] == "cancelling" and job["run_attempts"] == 0:
            self.finalize_cancel(job_id)

    def record_timings(self, job_id: str, processing: float = 0, loading: float = 0) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET processing_seconds=processing_seconds+?,"
                "model_load_seconds=model_load_seconds+? WHERE id=?",
                (round(processing, 3), round(loading, 3), job_id),
            )

    def fail_job(self, job_id: str, error: str) -> None:
        if self.cancellation_requested(job_id):
            return
        for page in self.pages(job_id):
            if page.status not in {"completed", "failed"}:
                page.status = "failed"
                page.error = error[:2000]
                page.needs_review = True
                self.save_page(job_id, page)
        self.finish_job(job_id)
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET error=? WHERE id=? "
                "AND status NOT IN ('cancelling','cancelled') AND delete_requested=0",
                (error[:2000], job_id),
            )

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

    def claim_job(self, backends: list[str] | None = None) -> dict | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            condition = ""
            parameters = []
            if backends is not None:
                if not backends:
                    return None
                condition = " AND backend IN (" + ",".join("?" for _ in backends) + ")"
                parameters = backends
            row = db.execute(
                "SELECT * FROM jobs WHERE status='queued' AND delete_requested=0"
                + condition
                + " ORDER BY created_at LIMIT 1",
                parameters,
            ).fetchone()
            if row is None:
                return None
            db.execute(
                "UPDATE jobs SET status='processing',run_attempts=run_attempts+1,updated_at=? "
                "WHERE id=?",
                (now(), row["id"]),
            )
            return {**dict(row), "run_attempts": row["run_attempts"] + 1}

    def start_page(self, job_id: str, number: int) -> int | None:
        with self.connect() as db:
            updated = db.execute(
                "UPDATE pages SET status='processing',attempts=attempts+1 "
                "WHERE job_id=? AND number=? AND EXISTS (SELECT 1 FROM jobs WHERE id=? "
                "AND status='processing' AND delete_requested=0)",
                (job_id, number, job_id),
            )
            if updated.rowcount == 0:
                return None
            row = db.execute(
                "SELECT attempts FROM pages WHERE job_id=? AND number=?", (job_id, number)
            ).fetchone()
            return row["attempts"]

    def save_page(self, job_id: str, page: PageResult) -> None:
        with self.connect() as db:
            updated = db.execute(
                "UPDATE pages SET status=?,result_json=? WHERE job_id=? AND number=? "
                "AND EXISTS (SELECT 1 FROM jobs WHERE id=? "
                "AND status NOT IN ('cancelling','cancelled') AND delete_requested=0)",
                (page.status, page.model_dump_json(), job_id, page.number, job_id),
            )
            if updated.rowcount:
                db.execute("UPDATE jobs SET updated_at=? WHERE id=?", (now(), job_id))

    def set_engine(self, job_id: str, metadata: dict) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET engine_json=? WHERE id=?", (json.dumps(metadata), job_id))

    def finish_job(self, job_id: str) -> str:
        job = self.get_job(job_id)
        if job and (job["status"] in {"cancelling", "cancelled"} or job["delete_requested"]):
            return job["status"]
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
                "UPDATE jobs SET status=?,error=?,updated_at=? WHERE id=? "
                "AND status NOT IN ('cancelling','cancelled') AND delete_requested=0",
                (status, error, now(), job_id),
            )
        return status

    def recover(self, max_attempts: int, backends: list[str] | None = None) -> None:
        # Hold the backend-specific lock. Never reset a different service's active job.
        condition = ""
        parameters = []
        if backends is not None:
            if not backends:
                return
            condition = " AND backend IN (" + ",".join("?" for _ in backends) + ")"
            parameters = backends
        with self.connect() as db:
            rows = db.execute(
                "SELECT pages.* FROM pages JOIN jobs ON jobs.id=pages.job_id "
                "WHERE pages.status='processing' AND jobs.status='processing' "
                "AND jobs.delete_requested=0" + condition,
                parameters,
            ).fetchall()
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
                "UPDATE jobs SET status='queued',updated_at=? WHERE status='processing'"
                + condition,
                [now(), *parameters],
            )

    def heartbeat(
        self,
        state: str,
        job_id: str | None = None,
        error: str | None = None,
        backend: str | None = None,
    ) -> None:
        with self.connect() as db:
            if backend is not None:
                db.execute("DELETE FROM worker WHERE id=1")
                db.execute(
                    "INSERT INTO workers(backend,state,heartbeat,job_id,error) VALUES(?,?,?,?,?) "
                    "ON CONFLICT(backend) DO UPDATE SET state=excluded.state,"
                    "heartbeat=excluded.heartbeat,job_id=excluded.job_id,error=excluded.error",
                    (backend, state, now(), job_id, error),
                )
                return
            db.execute(
                "INSERT INTO worker(id,state,heartbeat,job_id,error) VALUES(1,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET state=excluded.state,heartbeat=excluded.heartbeat,"
                "job_id=excluded.job_id,error=excluded.error",
                (state, now(), job_id, error),
            )

    def worker(self, backend: str | None = None) -> dict | None:
        with self.connect() as db:
            if backend:
                row = db.execute("SELECT * FROM workers WHERE backend=?", (backend,)).fetchone()
            else:
                row = db.execute("SELECT * FROM worker WHERE id=1").fetchone()
            return dict(row) if row else None
