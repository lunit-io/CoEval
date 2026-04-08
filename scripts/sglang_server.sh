#!/usr/bin/env bash
set -euo pipefail

# Defaults
GPU=0
PORT=9006
MODEL="learning-unit/placeholder"
TP=1
EXTRA_ARGS=()

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)         GPU="$2";   shift 2 ;;
        --port)        PORT="$2";  shift 2 ;;
        --model-path)  MODEL="$2"; shift 2 ;;
        --tp)          TP="$2";    shift 2 ;;
        *)             EXTRA_ARGS+=("$1"); shift ;;
    esac
done

echo "GPU=$GPU  PORT=$PORT  MODEL=$MODEL  TP=$TP"

CUDA_VISIBLE_DEVICES="$GPU" \
exec uv run python -m sglang.launch_server \
  --model-path "$MODEL" \
  --port "$PORT" --host 0.0.0.0 \
  --tp "$TP" --dtype bfloat16 --trust-remote-code \
  --attention-backend triton \
  --moe-runner-backend triton \
  "${EXTRA_ARGS[@]}"
