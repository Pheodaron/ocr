#!/bin/sh
# Model preparation uses only base dependencies, not Paddle or llama.cpp.
set -eu
cd "$(dirname "$0")/.."
# Workers run as uid 10001 and need read access to host-downloaded weights.
umask 022
exec uv run --frozen --no-dev ocr-service prepare-models --download "$@"
