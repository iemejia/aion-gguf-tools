#!/usr/bin/env zsh
set -euo pipefail

SCRIPT_DIR="${0:A:h}"
PROJECT_DIR="${SCRIPT_DIR:h}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-${PROJECT_DIR:h}/llama.cpp}"
MODEL="${AION_GGUF:-/tmp/aion-q40.gguf}"

if [[ ! -f "$MODEL" ]]; then
  echo "missing model: $MODEL" >&2
  exit 1
fi

if [[ ! -x "$LLAMA_CPP_DIR/build/bin/llama-completion" ]]; then
  echo "missing llama.cpp runner: $LLAMA_CPP_DIR/build/bin/llama-completion" >&2
  exit 1
fi

prompts=(
  "Give two short bullets about private on-device generation."
  "Describe a coding assistant in one short paragraph, no code."
  "Give two short bullets about low-latency local inference."
)

printf '\033[1;36mAion-1.0-Instruct GGUF on llama.cpp / Metal\033[0m\n'
printf 'model: %s\n' "$MODEL"
printf 'runtime: flash-attn=on, ubatch=512, ctx=2048, threads=8\n\n'

for idx in {1..${#prompts[@]}}; do
  prompt="${prompts[$idx]}"
  printf '\033[1;33m[%d/%d] %s\033[0m\n' "$idx" "${#prompts[@]}" "$prompt"

  out="$(mktemp -t aion-twitter-workload.out.XXXXXX)"
  log="$(mktemp -t aion-twitter-workload.log.XXXXXX)"
  "$LLAMA_CPP_DIR/build/bin/llama-completion" \
    -m "$MODEL" \
    -e -sp -no-cnv \
    --log-file "$log" \
    --perf \
    -fa on \
    -c 2048 \
    -t 8 \
    --poll 50 \
    -ub 512 \
    -p "<|user|>${prompt}<|end|><|assistant|>" \
    -n 72 \
    --temp 0.2 \
    --top-p 0.95 \
    --no-display-prompt \
    --no-warmup > >(tee "$out") 2>/dev/null

  awk '
    /^common_perf_print:/ {
      if ($0 ~ /eval time/ || $0 ~ /prompt eval time/ || $0 ~ /total time/) {
        print
      }
    }
  ' "$log"
  rm -f "$out" "$log"
  printf '\n'
done
