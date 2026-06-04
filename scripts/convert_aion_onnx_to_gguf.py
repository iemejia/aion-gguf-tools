#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import sys
from typing import Any

import numpy as np
import onnx

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = SCRIPT_DIR.parent
DEFAULT_LLAMA_CPP_DIR = PROJECT_DIR.parent / "llama.cpp"


def add_gguf_to_path() -> None:
    candidates = []
    if os.environ.get("GGUF_PY_PATH"):
        candidates.append(Path(os.environ["GGUF_PY_PATH"]))
    if os.environ.get("LLAMA_CPP_DIR"):
        candidates.append(Path(os.environ["LLAMA_CPP_DIR"]) / "gguf-py")
    candidates.append(DEFAULT_LLAMA_CPP_DIR / "gguf-py")

    for candidate in candidates:
        if (candidate / "gguf").exists():
            sys.path.insert(1, str(candidate))
            return

    searched = "\n".join(f"  - {candidate}" for candidate in candidates)
    raise SystemExit(f"could not find llama.cpp gguf-py; set LLAMA_CPP_DIR or GGUF_PY_PATH\nsearched:\n{searched}")


add_gguf_to_path()
import gguf  # noqa: E402


LOGGER = logging.getLogger("aion-onnx-to-gguf")


# The Edge bundle does not include tokenizer_config.chat_template. Aion's observed
# conversation format is newline-delimited, but its special tokens have
# rstrip=true in tokenizer.json. llama.cpp does not model special-token rstrip,
# so the GGUF template uses the compact rendering that tokenizes to the same IDs.
AION_LLAMA_CPP_CHAT_TEMPLATE = "{% for message in messages %}<|{{ message['role'] }}|>{{ message['content'] }}<|end|>{% endfor %}{% if add_generation_prompt %}<|assistant|>{% endif %}"


TENSOR_NAMES = {
    "token_embd": "token_embd.weight",
    "output_norm": "output_norm.weight",
    "output": "output.weight",
    "attn_norm": "blk.{layer}.attn_norm.weight",
    "attn_q": "blk.{layer}.attn_q.weight",
    "attn_q_norm": "blk.{layer}.attn_q_norm.weight",
    "attn_k": "blk.{layer}.attn_k.weight",
    "attn_k_norm": "blk.{layer}.attn_k_norm.weight",
    "attn_v": "blk.{layer}.attn_v.weight",
    "attn_out": "blk.{layer}.attn_output.weight",
    "ffn_norm": "blk.{layer}.ffn_norm.weight",
    "ffn_gate": "blk.{layer}.ffn_gate.weight",
    "ffn_down": "blk.{layer}.ffn_down.weight",
    "ffn_up": "blk.{layer}.ffn_up.weight",
}


ONNX_DTYPES = {
    1: np.float32,
    2: np.uint8,
    3: np.int8,
    6: np.int32,
    7: np.int64,
    10: np.float16,
}


def tensor_external_entry(tensor: onnx.TensorProto) -> dict[str, str]:
    return {entry.key: entry.value for entry in tensor.external_data}


class AionOnnxBundle:
    def __init__(self, model_dir: Path):
        self.model_dir = model_dir
        self.onnx_path = model_dir / "model.onnx"
        self.config_path = model_dir / "genai_config.json"
        self.tokenizer_path = model_dir / "tokenizer.json"

        self.config = json.loads(self.config_path.read_text(encoding="utf-8"))
        self.model = onnx.load(self.onnx_path, load_external_data=False)
        self.initializers = {tensor.name: tensor for tensor in self.model.graph.initializer}
        self.nodes = list(self.model.graph.node)

    @property
    def decoder_config(self) -> dict[str, Any]:
        model_config = self.config["model"]
        config = dict(model_config)
        config.update(model_config["decoder"])
        if "intermediate_size" not in config:
            node = self.quant_node_for("model.layers.0.mlp.gate_proj")
            config["intermediate_size"] = int(node_attrs(node)["N"])
        return config

    def tensor_array(self, name: str) -> np.ndarray[Any, Any]:
        tensor = self.initializers[name]
        dtype = ONNX_DTYPES.get(tensor.data_type)
        if dtype is None:
            raise ValueError(f"unsupported ONNX tensor dtype {tensor.data_type} for {name}")

        if tensor.external_data:
            external = tensor_external_entry(tensor)
            data_path = self.model_dir / external["location"]
            offset = int(external.get("offset", "0"))
            length = int(external["length"])
            with data_path.open("rb") as fp:
                fp.seek(offset)
                data = fp.read(length)
            array = np.frombuffer(data, dtype=dtype).copy()
        elif tensor.raw_data:
            array = np.frombuffer(tensor.raw_data, dtype=dtype).copy()
        else:
            array = onnx.numpy_helper.to_array(tensor)

        return array.reshape(tuple(tensor.dims))

    def quant_node_for(self, prefix: str, op_type: str = "MatMulNBits") -> onnx.NodeProto:
        matches = [
            node for node in self.nodes
            if node.op_type == op_type and any(prefix in item for item in node.input)
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one {op_type} node for {prefix}, found {len(matches)}")
        return matches[0]


def node_attrs(node: onnx.NodeProto) -> dict[str, Any]:
    return {attr.name: onnx.helper.get_attribute_value(attr) for attr in node.attribute}


def unpack_codes(packed: np.ndarray[Any, Any], bits: int, block_size: int) -> np.ndarray[Any, Any]:
    if bits == 8:
        return packed.astype(np.int16, copy=False)
    if bits == 4:
        low = packed & 0x0F
        high = packed >> 4
        return np.stack((low, high), axis=-1).reshape(*packed.shape[:-1], block_size).astype(np.int16, copy=False)
    if bits == 2:
        shifts = np.array([0, 2, 4, 6], dtype=np.uint8)
        return ((packed[..., None] >> shifts) & 0x03).reshape(*packed.shape[:-1], block_size).astype(np.int16, copy=False)
    raise ValueError(f"unsupported quantization bits: {bits}")


def dequant_matmul_nbits(bundle: AionOnnxBundle, prefix: str) -> np.ndarray[Any, Any]:
    node = bundle.quant_node_for(prefix)
    attrs = node_attrs(node)
    weight = bundle.tensor_array(node.input[1])
    scales = bundle.tensor_array(node.input[2]).reshape(int(attrs["N"]), -1).astype(np.float32)
    bits = int(attrs["bits"])
    block_size = int(attrs["block_size"])
    k = int(attrs["K"])
    n = int(attrs["N"])

    expected_blob = block_size * bits // 8
    expected_shape = (n, (k + block_size - 1) // block_size, expected_blob)
    if tuple(weight.shape) != expected_shape:
        raise ValueError(f"{prefix}: expected packed shape {expected_shape}, got {weight.shape}")

    codes = unpack_codes(weight, bits, block_size)
    zero_point = 2 ** (bits - 1)
    dequant = (codes - zero_point).astype(np.float32) * scales[:, :, None]
    return dequant.reshape(n, -1)[:, :k].astype(np.float16)


def dequant_gather_block_quantized(bundle: AionOnnxBundle, prefix: str) -> np.ndarray[Any, Any]:
    node = bundle.quant_node_for(prefix, op_type="GatherBlockQuantized")
    attrs = node_attrs(node)
    weight = bundle.tensor_array(node.input[0])
    scales = bundle.tensor_array(node.input[2]).astype(np.float32)
    bits = int(attrs["bits"])
    block_size = int(attrs["block_size"])

    if int(attrs["gather_axis"]) != 0 or int(attrs["quantize_axis"]) != 1:
        raise ValueError(f"unsupported GatherBlockQuantized axes for {prefix}: {attrs}")

    codes = unpack_codes(weight, bits, block_size)
    zero_point = 2 ** (bits - 1)
    dequant = (codes - zero_point).astype(np.float32) * scales[:, :, None]
    return dequant.reshape(weight.shape[0], -1).astype(np.float16)


def add_tokenizer(writer: gguf.GGUFWriter, model_dir: Path, model_config: dict[str, Any]) -> None:
    vocab_size = int(model_config["vocab_size"])
    tokenizer = json.loads((model_dir / "tokenizer.json").read_text(encoding="utf-8"))

    vocab = tokenizer["model"]["vocab"]
    reverse_vocab = {idx: token for token, idx in vocab.items()}
    token_types = [int(gguf.TokenType.NORMAL)] * vocab_size
    tokens = [""] * vocab_size

    for idx in range(min(vocab_size, len(reverse_vocab))):
        tokens[idx] = reverse_vocab[idx]

    for item in tokenizer.get("added_tokens", []):
        idx = int(item["id"])
        if idx >= vocab_size:
            continue
        tokens[idx] = item["content"]
        token_types[idx] = int(gguf.TokenType.CONTROL if item.get("special") else gguf.TokenType.USER_DEFINED)

    missing = [idx for idx, token in enumerate(tokens) if token == ""]
    if missing:
        raise ValueError(f"tokenizer has missing token IDs, first missing IDs: {missing[:10]}")

    merges = tokenizer["model"].get("merges", [])
    merges = [" ".join(pair) if isinstance(pair, list) else pair for pair in merges]

    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen2")
    writer.add_token_list(tokens)
    writer.add_token_types(token_types)
    writer.add_token_merges(merges)
    writer.add_bos_token_id(int(model_config.get("bos_token_id", 1)))
    eos = model_config.get("eos_token_id", 200018)
    writer.add_eos_token_id(int(eos[0] if isinstance(eos, list) else eos))
    writer.add_pad_token_id(int(model_config.get("pad_token_id", 199999)))
    writer.add_add_bos_token(False)
    writer.add_add_eos_token(False)
    writer.add_chat_template(AION_LLAMA_CPP_CHAT_TEMPLATE)


def add_metadata(writer: gguf.GGUFWriter, bundle: AionOnnxBundle) -> None:
    config = bundle.decoder_config
    writer.add_name("Aion-1.0-Instruct")
    writer.add_description("Aion Edge ONNX bundle converted directly to F16 GGUF")
    writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_F16))
    writer.add_context_length(int(config["context_length"]))
    writer.add_embedding_length(int(config["hidden_size"]))
    writer.add_feed_forward_length(int(config["intermediate_size"]))
    writer.add_block_count(int(config["num_hidden_layers"]))
    writer.add_head_count(int(config["num_attention_heads"]))
    writer.add_head_count_kv(int(config["num_key_value_heads"]))
    writer.add_key_length(int(config["head_size"]))
    writer.add_value_length(int(config["head_size"]))
    writer.add_rope_dimension_count(int(config["head_size"]))
    writer.add_rope_freq_base(float(config.get("rope_theta", 1_000_000.0)))
    writer.add_layer_norm_rms_eps(float(config.get("rms_norm_eps", 1e-6)))
    add_tokenizer(writer, bundle.model_dir, config)


def add_tensor(writer: gguf.GGUFWriter, name: str, data: np.ndarray[Any, Any], dry_run: bool) -> None:
    LOGGER.info("%-38s %s %s", name, tuple(data.shape), data.dtype)
    if not dry_run:
        writer.add_tensor(name, np.ascontiguousarray(data))


def norm_tensor(bundle: AionOnnxBundle, name: str) -> np.ndarray[Any, Any]:
    return bundle.tensor_array(name).astype(np.float32)


def convert(args: argparse.Namespace) -> None:
    bundle = AionOnnxBundle(args.model_dir)
    config = bundle.decoder_config
    layer_count = int(config["num_hidden_layers"])
    layers_to_write = layer_count if args.max_layers is None else min(args.max_layers, layer_count)

    writer = gguf.GGUFWriter(args.outfile, "qwen3", use_temp_file=True, dry_run=args.dry_run)
    add_metadata(writer, bundle)

    lm_head = dequant_matmul_nbits(bundle, "lm_head")
    add_tensor(writer, TENSOR_NAMES["token_embd"], lm_head, args.dry_run)

    for layer in range(layers_to_write):
        prefix = f"model.layers.{layer}"
        add_tensor(writer, TENSOR_NAMES["attn_norm"].format(layer=layer), norm_tensor(bundle, f"{prefix}.input_layernorm.weight"), args.dry_run)
        add_tensor(writer, TENSOR_NAMES["attn_q"].format(layer=layer), dequant_matmul_nbits(bundle, f"{prefix}.self_attn.q_proj"), args.dry_run)
        add_tensor(writer, TENSOR_NAMES["attn_q_norm"].format(layer=layer), norm_tensor(bundle, f"{prefix}.self_attn.q_norm.weight"), args.dry_run)
        add_tensor(writer, TENSOR_NAMES["attn_k"].format(layer=layer), dequant_matmul_nbits(bundle, f"{prefix}.self_attn.k_proj"), args.dry_run)
        add_tensor(writer, TENSOR_NAMES["attn_k_norm"].format(layer=layer), norm_tensor(bundle, f"{prefix}.self_attn.k_norm.weight"), args.dry_run)
        add_tensor(writer, TENSOR_NAMES["attn_v"].format(layer=layer), dequant_matmul_nbits(bundle, f"{prefix}.self_attn.v_proj"), args.dry_run)
        add_tensor(writer, TENSOR_NAMES["attn_out"].format(layer=layer), dequant_matmul_nbits(bundle, f"{prefix}.self_attn.o_proj"), args.dry_run)
        add_tensor(writer, TENSOR_NAMES["ffn_norm"].format(layer=layer), norm_tensor(bundle, f"{prefix}.post_attention_layernorm.weight"), args.dry_run)
        add_tensor(writer, TENSOR_NAMES["ffn_gate"].format(layer=layer), dequant_matmul_nbits(bundle, f"{prefix}.mlp.gate_proj"), args.dry_run)
        add_tensor(writer, TENSOR_NAMES["ffn_down"].format(layer=layer), dequant_matmul_nbits(bundle, f"{prefix}.mlp.down_proj"), args.dry_run)
        add_tensor(writer, TENSOR_NAMES["ffn_up"].format(layer=layer), dequant_matmul_nbits(bundle, f"{prefix}.mlp.up_proj"), args.dry_run)

    if layers_to_write != layer_count:
        LOGGER.warning("partial conversion requested: wrote %d/%d layers", layers_to_write, layer_count)

    add_tensor(writer, TENSOR_NAMES["output_norm"], norm_tensor(bundle, "model.norm.weight"), args.dry_run)
    add_tensor(writer, TENSOR_NAMES["output"], lm_head, args.dry_run)

    if args.dry_run:
        LOGGER.info("dry run complete; no GGUF file written")
        return

    LOGGER.info("writing GGUF: %s", args.outfile)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Edge Aion ONNX bundle directly to F16 Qwen3 GGUF")
    parser.add_argument("model_dir", type=Path, help="directory containing model.onnx, model.onnx.data, tokenizer.json, genai_config.json")
    parser.add_argument("--outfile", type=Path, default=Path("aion-f16.gguf"), help="output GGUF path")
    parser.add_argument("--dry-run", action="store_true", help="parse/dequantize and log tensor shapes without writing a file")
    parser.add_argument("--max-layers", type=int, default=None, help="convert only the first N layers for smoke testing")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    convert(args)


if __name__ == "__main__":
    main()