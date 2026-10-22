# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.23 AS uv
FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    OCR_DATA_DIR=/data \
    OCR_MODELS_DIR=/models \
    PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True \
    HF_HUB_OFFLINE=1 \
    PATH="/app/.venv/bin:$PATH"

RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates libgl1 libglib2.0-0 libgomp1 fonts-dejavu-core \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --gid 10001 ocr \
    && useradd --uid 10001 --gid 10001 --create-home ocr \
    && mkdir -p /data /models \
    && chown 10001:10001 /data /models

COPY --from=uv /uv /usr/local/bin/uv
WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
COPY src ./src
COPY THIRD_PARTY_NOTICES.md LICENSES ./licenses/
# Optional combined CA bundle for a build behind an organization/session proxy.
# No session certificate is stored in the resulting image.
RUN --mount=type=secret,id=proxy_ca --mount=type=cache,target=/root/.cache/uv \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    uv sync --frozen --extra ocr --no-dev --no-editable

USER 10001:10001
EXPOSE 8000
CMD ["ocr-service", "api", "--host", "0.0.0.0"]
