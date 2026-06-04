# Aion GGUF demo notes

## Models

- `/tmp/aion-q4_k_m.gguf` - recommended demo model, 1.6 GB, validated, fastest candidate.
- `/tmp/aion-q8_0.gguf` - quality-safe fallback, 2.7 GB, validated.
- `/tmp/aion-f16.gguf` - correctness oracle, 5.0 GB, validated but slower.

## Quick demo

```sh
scripts/aion_demo.sh "Write one sentence about Apple Silicon."
```

Use a fallback model:

```sh
AION_GGUF=/tmp/aion-q8_0.gguf scripts/aion_demo.sh "Hello"
AION_GGUF=/tmp/aion-f16.gguf scripts/aion_demo.sh "Hello"
```

Tune output:

```sh
AION_TEMP=0.2 AION_TOP_P=0.95 AION_MAX_TOKENS=128 scripts/aion_demo.sh "Explain local AI in one paragraph."
```

Tune context/threading:

```sh
AION_CTX=2048 AION_THREADS=8 scripts/aion_demo.sh "Hello"
AION_CTX=8192 scripts/aion_demo.sh "Use the full training context limit if needed."
```

## Validation

```sh
python scripts/aion_validate_gguf.py --model-dir /path/to/aion-onnx-bundle --gguf /tmp/aion-q4_k_m.gguf --allow-quantized --bench-prompt-tokens 128 --bench-gen-tokens 64 --bench-repetitions 3
python scripts/aion_validate_gguf.py --model-dir /path/to/aion-onnx-bundle --gguf /tmp/aion-q8_0.gguf --allow-quantized --bench-prompt-tokens 128 --bench-gen-tokens 64 --bench-repetitions 3
python scripts/aion_validate_gguf.py --model-dir /path/to/aion-onnx-bundle --gguf /tmp/aion-f16.gguf --bench-prompt-tokens 128 --bench-gen-tokens 64 --bench-repetitions 3
```

Validated results on the current M4 Max:

- F16: `tg64 ~89.6 t/s`, size 5.0 GB.
- Q8_0: `tg64 ~144.6 t/s`, size 2.7 GB.
- Q4_K_M: `tg64 ~194.6 t/s`, size 1.6 GB.
- Q4_K_M with safe demo knobs (`-fa on -ub 512 -t 8`): `tg64 ~214.1 t/s`.

The demo wrapper uses those safe knobs by default plus `AION_CTX=2048`, which reduces KV memory from about 896 MiB at 8192 context to about 224 MiB. Raise `AION_CTX` if a demo prompt needs more context.

## Prompt format

The official tokenizer has `rstrip=true` on Aion special tokens, so the newline-form prompt and compact prompt tokenize to the same IDs in the official tokenizer:

```text
<|user|>\nHello<|end|>\n<|assistant|>\n
<|user|>Hello<|end|><|assistant|>
```

llama.cpp does not model special-token `rstrip`, so the demo uses the compact form:

```text
<|user|>{prompt}<|end|><|assistant|>
```

## Notes

No custom kernel was needed. The performance jump from F16 to Q4/Q8 comes from using existing llama.cpp quantized GGUF formats instead of streaming the 5 GB dense F16 baseline on every decode step.
