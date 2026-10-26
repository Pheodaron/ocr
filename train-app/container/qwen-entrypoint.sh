#!/bin/sh
set -eu
test -s /models/qwen/Qwen3.8-27B-UD-Q3_K_XL.gguf || {
    echo 'Нет модели Qwen: выполните подготовку моделей по README.' >&2
    exit 1
}
test -s /models/qwen/mmproj-F16.gguf
echo 'Ожидание исключительного доступа к GPU; при необходимости остановите lighton4b или trainer.'
exec flock --exclusive --no-fork /data/gpu.lock /opt/llama/bin/llama-server "$@"
