#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${PROJECT_DIR:h}/llama.cpp}"
MODEL="${AION_GGUF:-/tmp/aion-q40.gguf}"
PROMPT="${1:-Hello}"
MAX_TOKENS="${AION_MAX_TOKENS:-128}"
TEMP="${AION_TEMP:-0.2}"
TOP_P="${AION_TOP_P:-0.95}"
CTX="${AION_CTX:-2048}"
THREADS="${AION_THREADS:-8}"

if [[ ! -f "$MODEL" ]]; then
  echo "Missing model: $MODEL" >&2
  echo "Set AION_GGUF=/path/to/model.gguf or generate /tmp/aion-q40.gguf first." >&2
  exit 1
fi

if [[ ! -x "$LLAMA_CPP_DIR/build/bin/llama-completion" ]]; then
  echo "Missing llama.cpp runner: $LLAMA_CPP_DIR/build/bin/llama-completion" >&2
  echo "Set LLAMA_CPP_DIR=/path/to/llama.cpp or build llama.cpp first." >&2
  exit 1
fi

exec "$LLAMA_CPP_DIR/build/bin/llama-completion" \
  -m "$MODEL" \
  -e -sp -no-cnv \
  -fa on \
  -c "$CTX" \
  -t "$THREADS" \
  -ub 512 \
  -p "<|user|>${PROMPT}<|end|><|assistant|>" \
  -n "$MAX_TOKENS" \
  --temp "$TEMP" \
  --top-p "$TOP_P" \
  --no-display-prompt \
  --no-warmup
