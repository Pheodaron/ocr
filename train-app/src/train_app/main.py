import logging
from contextlib import asynccontextmanager

from alembic import command
from alembic.config import Config
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from .api import router
from .config import Settings
from .db import create_database
from .queue import Dispatcher
from .storage import Storage

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or Settings()
    storage = Storage(settings.data_dir)
    database = create_database(settings.database_url)

    @asynccontextmanager
    async def lifespan(app):
        migration = Config("alembic.ini")
        migration.set_main_option("sqlalchemy.url", settings.database_url)
        command.upgrade(migration, "head")
        dispatcher = Dispatcher(database, storage, settings)
        if settings.dispatcher:
            await dispatcher.start()
        yield
        await dispatcher.stop()
        database.kw["bind"].dispose()

    app = FastAPI(title="Мастерская обучения OCR", lifespan=lifespan)
    app.state.settings, app.state.storage, app.state.database = settings, storage, database

    @app.middleware("http")
    async def local_requests(request: Request, call_next):
        origin = request.headers.get("origin")
        allowed = settings.allowed_origins.split(",")
        if origin and request.method not in {"GET", "HEAD", "OPTIONS"} and origin not in allowed:
            return JSONResponse({"detail": "Недопустимый источник запроса"}, status_code=403)
        response = await call_next(request)
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = (
            "default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; "
            "script-src 'self'; connect-src 'self'; frame-src 'none'; object-src 'none'; base-uri 'none'"
        )
        return response

    app.include_router(router)

    @app.get("/health")
    def health():
        return {"status": "ok"}

    if settings.frontend_dir.exists():
        app.mount("/", StaticFiles(directory=settings.frontend_dir, html=True), name="interface")
    return app


app = create_app()
