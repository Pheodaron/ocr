# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:0.12.23 AS uv
FROM debian:bookworm-slim AS llama-build
ARG LLAMA_COMMIT=47a39665e7081dc482feec169961acc09750a5c4
RUN apt-get update && apt-get install -y --no-install-recommends \
      ca-certificates git cmake build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=secret,id=proxy_ca \
    if [ -f /run/secrets/proxy_ca ]; then export GIT_SSL_CAINFO=/run/secrets/proxy_ca; fi; \
    git init /src && git -C /src fetch --depth=1 https://github.com/ggml-org/llama.cpp.git "$LLAMA_COMMIT" \
    && git -C /src checkout FETCH_HEAD \
    && cmake -S /src -B /build -DCMAKE_BUILD_TYPE=Release -DBUILD_SHARED_LIBS=OFF \
      -DGGML_NATIVE=OFF -DGGML_AVX2=ON -DGGML_AVX512=OFF -DGGML_CUDA=OFF \
      -DLLAMA_CURL=OFF -DLLAMA_OPENSSL=OFF -DLLAMA_BUILD_TESTS=OFF \
    && cmake --build /build --target llama-server -j 2

FROM python:3.12-slim-bookworm AS base

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    UV_SYSTEM_CERTS=true \
    PYTHONPATH=/app/src \
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
COPY pyproject.toml uv.lock ./
# Optional combined CA bundle for a build behind an organization/session proxy.
# No session certificate is stored in the resulting image.
RUN --mount=type=secret,id=proxy_ca --mount=type=cache,target=/root/.cache/uv \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    uv sync --frozen --no-dev --no-install-project

# Run the checked-in application directly: no isolated Hatchling build/download.
# Keep source changes after dependency installation so its cache remains reusable.
COPY src ./src
COPY THIRD_PARTY_NOTICES.md LICENSES ./licenses/
COPY --chmod=755 container/ocr-service /usr/local/bin/ocr-service
# Source files may be checked out with a restrictive host umask.
RUN chmod -R a+rX /app/src

USER 10001:10001
EXPOSE 8000
CMD ["ocr-service", "api", "--host", "0.0.0.0"]

FROM base AS api

FROM base AS vlm
COPY --from=llama-build /build/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-build /src/LICENSE /usr/local/share/licenses/llama.cpp-MIT.txt
CMD ["ocr-service", "worker"]

FROM base AS paddle
USER root
RUN --mount=type=secret,id=proxy_ca --mount=type=cache,target=/root/.cache/uv \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    uv sync --frozen --extra ocr --no-dev --no-install-project
USER 10001:10001
CMD ["ocr-service", "worker"]

FROM base AS kdl
USER root
RUN --mount=type=secret,id=proxy_ca --mount=type=cache,target=/root/.cache/uv \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    uv sync --frozen --extra kdl --no-dev --no-install-project
USER 10001:10001
CMD ["ocr-service", "worker"]

FROM base AS falcon
USER root
RUN --mount=type=secret,id=proxy_ca --mount=type=cache,target=/root/.cache/uv \
    if [ -f /run/secrets/proxy_ca ]; then export SSL_CERT_FILE=/run/secrets/proxy_ca; fi; \
    uv sync --frozen --extra falcon --no-dev --no-install-project
USER 10001:10001
CMD ["ocr-service", "worker"]
