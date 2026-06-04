#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import onnx
import onnxruntime as ort
from tokenizers import Tokenizer


OFFICIAL_AION_CHAT_TEMPLATE = """{% for message in messages %}<|{{ message['role'] }}|>\n{{ message['content'] }}<|end|>\n{% endfor %}{% if add_generation_prompt %}<|assistant|>\n{% endif %}"""
LLAMA_CPP_AION_CHAT_TEMPLATE = """{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"""

LOGITS_PROBE_CPP = r'''
#include "llama.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <utility>
#include <vector>

static std::vector<llama_token> parse_ids(const char * text) {
    std::vector<llama_token> ids;
    const char * cur = text;
    while (*cur) {
        char * end = nullptr;
        long value = std::strtol(cur, &end, 10);
        if (end == cur) {
            break;
        }
        ids.push_back((llama_token) value);
        cur = end;
        while (*cur == ',' || *cur == ' ') {
            ++cur;
        }
    }
    return ids;
}

int main(int argc, char ** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s model.gguf comma-separated-token-ids [top_k]\n", argv[0]);
        return 2;
    }

    const int top_k = argc >= 4 ? std::atoi(argv[3]) : 20;
    std::vector<llama_token> ids = parse_ids(argv[2]);
    if (ids.empty()) {
        std::fprintf(stderr, "no token ids parsed\n");
        return 2;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = 999;
    llama_model * model = llama_model_load_from_file(argv[1], mparams);
    if (!model) {
        std::fprintf(stderr, "failed to load model\n");
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = 128;
    cparams.n_batch = 128;
    cparams.n_ubatch = 128;
    cparams.n_threads = 8;
    cparams.n_threads_batch = 8;
    cparams.no_perf = true;
    llama_context * ctx = llama_init_from_model(model, cparams);
    if (!ctx) {
        std::fprintf(stderr, "failed to create context\n");
        llama_model_free(model);
        return 1;
    }

    llama_batch batch = llama_batch_init((int32_t) ids.size(), 0, 1);
    for (int32_t i = 0; i < (int32_t) ids.size(); ++i) {
        batch.token[i] = ids[i];
        batch.pos[i] = i;
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = 0;
        batch.logits[i] = (i == (int32_t) ids.size() - 1) ? 1 : 0;
    }
    batch.n_tokens = (int32_t) ids.size();

    int ret = llama_decode(ctx, batch);
    if (ret != 0) {
        std::fprintf(stderr, "llama_decode failed: %d\n", ret);
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    const int32_t n_vocab = llama_vocab_n_tokens(vocab);
    const float * logits = llama_get_logits(ctx);
    std::vector<std::pair<float, int>> top;
    top.reserve(n_vocab);
    for (int i = 0; i < n_vocab; ++i) {
        top.push_back({logits[i], i});
    }
    std::partial_sort(top.begin(), top.begin() + std::min(top_k, n_vocab), top.end(), [](auto a, auto b) {
        return a.first > b.first;
    });

    char piece[512];
    for (int i = 0; i < top_k && i < n_vocab; ++i) {
        int id = top[i].second;
        int n = llama_token_to_piece(vocab, id, piece, sizeof(piece) - 1, 0, true);
        if (n < 0 || n >= (int) sizeof(piece)) {
            std::snprintf(piece, sizeof(piece), "<piece-error>");
        } else {
            piece[n] = 0;
        }
        std::printf("%d\t%.8f\t%s\n", id, top[i].first, piece);
    }

    llama_batch_free(batch);
    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
'''


def run(cmd: list[str], cwd: Path | None = None, timeout: int = 600) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=cwd, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True, timeout=timeout)


def render_aion_template(messages: list[dict[str, str]], add_generation_prompt: bool, compact: bool) -> str:
    parts: list[str] = []
    if compact:
        for message in messages:
            parts.append(f"<|{message['role']}|>{message['content']}<|end|>")
        if add_generation_prompt:
            parts.append("<|assistant|>")
    else:
        for message in messages:
            parts.append(f"<|{message['role']}|>\n{message['content']}<|end|>\n")
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
    return "".join(parts)


def read_gguf_fields(llama_cpp_dir: Path, gguf_path: Path) -> tuple[dict[str, Any], dict[str, tuple[list[int], int]]]:
    gguf_py = llama_cpp_dir / "gguf-py"
    if not gguf_py.exists():
        raise AssertionError(f"missing llama.cpp gguf-py directory: {gguf_py}")
    sys.path.insert(0, str(gguf_py))
    from gguf import GGUFReader  # type: ignore

    reader = GGUFReader(str(gguf_path))
    fields: dict[str, Any] = {}
    for key, field in reader.fields.items():
        value = field.parts[-1] if field.parts else None
        if hasattr(value, "tolist"):
            value = value.tolist()
        if key in {"tokenizer.ggml.add_bos_token", "tokenizer.ggml.add_eos_token"} and isinstance(value, list):
            value = bool(value[0])
        if isinstance(value, list) and all(isinstance(item, int) for item in value):
            try:
                value = bytes(value).decode("utf-8")
            except ValueError:
                pass
        fields[key] = value
    tensors = {tensor.name: (list(tensor.shape), int(tensor.tensor_type)) for tensor in reader.tensors}
    return fields, tensors


def verify_metadata(llama_cpp_dir: Path, gguf_path: Path, require_f32_norms: bool) -> None:
    fields, tensors = read_gguf_fields(llama_cpp_dir, gguf_path)
    expected = {
        "general.architecture": "qwen3",
        "tokenizer.ggml.add_bos_token": False,
        "tokenizer.ggml.add_eos_token": False,
        "tokenizer.chat_template": LLAMA_CPP_AION_CHAT_TEMPLATE,
    }
    for key, expected_value in expected.items():
        actual = fields.get(key)
        if actual != expected_value:
            raise AssertionError(f"metadata {key}: expected {expected_value!r}, got {actual!r}")

    if require_f32_norms:
        for name in ["output_norm.weight", "blk.0.attn_norm.weight", "blk.0.attn_q_norm.weight", "blk.0.attn_k_norm.weight", "blk.0.ffn_norm.weight"]:
            shape, tensor_type = tensors[name]
            if tensor_type != 0:
                raise AssertionError(f"{name} must be F32 tensor type 0, got type {tensor_type} shape {shape}")


def verify_tokenizer(model_dir: Path) -> list[int]:
    tokenizer_json = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))
    added = {item["content"]: item for item in tokenizer_json.get("added_tokens", [])}
    for token in ["<|system|>", "<|user|>", "<|assistant|>", "<|end|>"]:
        item = added.get(token)
        if item is None:
            raise AssertionError(f"missing special token {token}")
        if not item.get("rstrip", False):
            raise AssertionError(f"{token} must have rstrip=true in official tokenizer.json")

    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    messages = [{"role": "user", "content": "Hello"}]
    official_prompt = render_aion_template(messages, add_generation_prompt=True, compact=False)
    compact_prompt = render_aion_template(messages, add_generation_prompt=True, compact=True)
    official_ids = tokenizer.encode(official_prompt, add_special_tokens=False).ids
    compact_ids = tokenizer.encode(compact_prompt, add_special_tokens=False).ids
    expected_ids = [200021, 13225, 200020, 200019]
    if official_ids != expected_ids:
        raise AssertionError(f"official prompt IDs: expected {expected_ids}, got {official_ids}")
    if compact_ids != official_ids:
        raise AssertionError(f"compact prompt IDs differ from official IDs: {compact_ids} vs {official_ids}")
    return expected_ids


def verify_chat_template_engine(llama_cpp_dir: Path) -> None:
    binary = llama_cpp_dir / "build/bin/test-chat-template"
    if not binary.exists():
        raise AssertionError("missing build/bin/test-chat-template; build it with: cmake --build build --target test-chat-template")

    with tempfile.TemporaryDirectory(prefix="aion-template-") as tmp_name:
        tmp = Path(tmp_name)
        template_path = tmp / "aion.jinja"
        input_path = tmp / "input.json"
        output_path = tmp / "out.txt"
        template_path.write_text(LLAMA_CPP_AION_CHAT_TEMPLATE, encoding="utf-8")
        input_path.write_text(json.dumps({
            "messages": [{"role": "user", "content": "Hello"}],
            "add_generation_prompt": True,
        }), encoding="utf-8")
        run([str(binary), str(template_path), "--json", str(input_path), "--output", str(output_path)], cwd=llama_cpp_dir, timeout=120)
        rendered = output_path.read_text(encoding="utf-8")

    expected = "<|user|>Hello<|end|><|assistant|>"
    if rendered != expected:
        raise AssertionError(f"chat-template renderer mismatch: expected {expected!r}, got {rendered!r}")
    print("chat-template renderer:", rendered)


def onnx_top_logits(model_dir: Path, token_ids: list[int], top_k: int) -> list[tuple[int, float, str]]:
    tokenizer = Tokenizer.from_file(str(model_dir / "tokenizer.json"))
    session = ort.InferenceSession(str(model_dir / "model.onnx"), providers=["CPUExecutionProvider"])
    inputs: dict[str, np.ndarray[Any, Any]] = {
        "input_ids": np.array([token_ids], dtype=np.int64),
        "attention_mask": np.ones((1, len(token_ids)), dtype=np.int64),
    }
    for layer in range(28):
        inputs[f"past_key_values.{layer}.key"] = np.zeros((1, 8, 0, 128), dtype=np.float16)
        inputs[f"past_key_values.{layer}.value"] = np.zeros((1, 8, 0, 128), dtype=np.float16)
    logits = session.run(["logits"], inputs)[0][0, -1].astype(np.float32)
    indices = np.argsort(logits)[-top_k:][::-1]
    return [(int(index), float(logits[index]), tokenizer.id_to_token(int(index))) for index in indices]


def compile_logits_probe(llama_cpp_dir: Path, build_dir: Path) -> Path:
    with tempfile.TemporaryDirectory(prefix="aion-logits-") as tmp_name:
        tmp = Path(tmp_name)
        source = tmp / "aion_logits.cpp"
        binary = build_dir / "aion-logits-probe"
        source.write_text(LOGITS_PROBE_CPP, encoding="utf-8")
        cmd = [
            "c++", "-std=c++17",
            "-Iinclude", "-Iggml/include",
            str(source),
            "-Lbuild/bin", "-lllama", f"-Wl,-rpath,{llama_cpp_dir / 'build/bin'}",
            "-o", str(binary),
        ]
        run(cmd, cwd=llama_cpp_dir, timeout=120)
    return build_dir / "aion-logits-probe"


def llama_top_logits(probe: Path, gguf_path: Path, token_ids: list[int], top_k: int) -> list[tuple[int, float, str]]:
    result = run([str(probe), str(gguf_path), ",".join(str(token_id) for token_id in token_ids), str(top_k)], timeout=600)
    rows: list[tuple[int, float, str]] = []
    for line in result.stdout.splitlines():
        parts = line.split("\t", 2)
        if len(parts) == 3:
            rows.append((int(parts[0]), float(parts[1]), parts[2]))
    if len(rows) < top_k:
        raise AssertionError(f"expected {top_k} llama logits rows, got {len(rows)}\nstderr:\n{result.stderr}")
    return rows


def verify_logits(model_dir: Path, gguf_path: Path, llama_cpp_dir: Path, build_dir: Path, token_ids: list[int], max_top_logit_drift: float, min_top10_overlap: int) -> None:
    onnx_rows = onnx_top_logits(model_dir, token_ids, top_k=10)
    probe = compile_logits_probe(llama_cpp_dir, build_dir)
    llama_rows = llama_top_logits(probe, gguf_path, token_ids, top_k=10)
    if onnx_rows[0][0] != llama_rows[0][0]:
        raise AssertionError(f"top token mismatch: ONNX {onnx_rows[0]} vs llama.cpp {llama_rows[0]}")
    if abs(onnx_rows[0][1] - llama_rows[0][1]) > max_top_logit_drift:
        raise AssertionError(f"top logit drift too large: ONNX {onnx_rows[0][1]:.6f} vs llama.cpp {llama_rows[0][1]:.6f}")

    overlap = {row[0] for row in onnx_rows[:10]} & {row[0] for row in llama_rows[:10]}
    if len(overlap) < min_top10_overlap:
        raise AssertionError(f"top-10 overlap too low: {len(overlap)}\nONNX={onnx_rows}\nllama={llama_rows}")

    print("logits top-1:", "ONNX", onnx_rows[0], "llama.cpp", llama_rows[0])
    print("logits top-10 overlap:", len(overlap), sorted(overlap))


def verify_generation(llama_cpp_dir: Path, gguf_path: Path) -> None:
    cmd = [
        str(llama_cpp_dir / "build/bin/llama-completion"),
        "-m", str(gguf_path),
        "-e", "-sp", "-no-cnv",
        "-p", "<|user|>Hello<|end|><|assistant|>",
        "-n", "24",
        "--no-display-prompt",
        "--temp", "0",
        "--seed", "1",
        "--no-warmup",
    ]
    result = run(cmd, cwd=llama_cpp_dir, timeout=600)
    if "Hello! How can I assist you today?" not in result.stdout:
        raise AssertionError(f"unexpected generation output:\n{result.stdout[-2000:]}\nstderr:\n{result.stderr[-2000:]}")
    print("generation smoke: Hello! How can I assist you today?")


def run_speed(llama_cpp_dir: Path, gguf_path: Path, prompt_tokens: int, gen_tokens: int, repetitions: int) -> None:
    cmd = [
        str(llama_cpp_dir / "build/bin/llama-bench"),
        "-m", str(gguf_path),
        "-ngl", "99",
        "-p", str(prompt_tokens),
        "-n", str(gen_tokens),
        "-r", str(repetitions),
    ]
    result = run(cmd, cwd=llama_cpp_dir, timeout=900)
    print(result.stdout)
    if "t/s" not in result.stdout and "tok/s" not in result.stdout:
        raise AssertionError(f"llama-bench did not report tok/s:\n{result.stdout}\nstderr:\n{result.stderr}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Aion ONNX -> GGUF correctness, chat template equivalence, generation, and speed.")
    parser.add_argument("--model-dir", type=Path, default=Path("/Users/alvarovidela/Code/aion-edge-onnx-bundle-backup-2026.5.30.1"))
    parser.add_argument("--gguf", type=Path, default=Path("/tmp/aion-f16.gguf"))
    default_llama_cpp_dir = Path(os.environ.get("LLAMA_CPP_DIR", Path(__file__).resolve().parents[2] / "llama.cpp"))
    parser.add_argument("--llama-cpp-dir", type=Path, default=default_llama_cpp_dir)
    parser.add_argument("--build-dir", type=Path, default=Path("/tmp"))
    parser.add_argument("--skip-speed", action="store_true")
    parser.add_argument("--allow-quantized", action="store_true", help="Relax tensor dtype and logit-drift thresholds for llama.cpp-quantized GGUFs.")
    parser.add_argument("--bench-prompt-tokens", type=int, default=128)
    parser.add_argument("--bench-gen-tokens", type=int, default=64)
    parser.add_argument("--bench-repetitions", type=int, default=3)
    args = parser.parse_args()

    if not args.gguf.exists():
        raise SystemExit(f"missing GGUF: {args.gguf}")
    if not (args.llama_cpp_dir / "build/bin/llama-completion").exists():
        raise SystemExit("missing build/bin/llama-completion; build it first")
    if not args.skip_speed and not (args.llama_cpp_dir / "build/bin/llama-bench").exists():
        raise SystemExit("missing build/bin/llama-bench; build it first or pass --skip-speed")

    print("checking GGUF metadata and tensor dtypes")
    verify_metadata(args.llama_cpp_dir, args.gguf, require_f32_norms=not args.allow_quantized)

    print("checking official tokenizer/template equivalence")
    token_ids = verify_tokenizer(args.model_dir)
    print("prompt token IDs:", token_ids)

    print("checking llama.cpp chat-template renderer")
    verify_chat_template_engine(args.llama_cpp_dir)

    print("checking ONNX vs llama.cpp logits")
    verify_logits(
        args.model_dir,
        args.gguf,
        args.llama_cpp_dir,
        args.build_dir,
        token_ids,
        max_top_logit_drift=2.0 if args.allow_quantized else 0.5,
        min_top10_overlap=5 if args.allow_quantized else 7,
    )

    print("checking deterministic generation smoke")
    verify_generation(args.llama_cpp_dir, args.gguf)

    if not args.skip_speed:
        print("running speed benchmark")
        run_speed(args.llama_cpp_dir, args.gguf, args.bench_prompt_tokens, args.bench_gen_tokens, args.bench_repetitions)

    print("Aion GGUF validation passed")


if __name__ == "__main__":
    main()
