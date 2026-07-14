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

# GGUF Q4_0 block parameters
GGML_QK4_0 = 32  # values per Q4_0 block
GGML_Q4_0_BLOCK_BYTES = 2 + GGML_QK4_0 // 2  # 18 bytes: fp16 scale + 16 nibble bytes

# GGUF Q8_0 block parameters
GGML_QK8_0 = 32  # values per Q8_0 block
GGML_Q8_0_BLOCK_BYTES = 2 + GGML_QK8_0  # 34 bytes: fp16 scale + 32 int8 values


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
    """Dequantize ONNX MatMulNBits to FP32 (used as fallback for non-32 block sizes)."""
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
    return dequant.reshape(n, -1)[:, :k]


def repack_onnx_nibbles_to_q4_0(onnx_packed: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Repack ONNX 4-bit nibble order to GGUF Q4_0 nibble order.

    ONNX MatMulNBits packing (per 16-byte group = 32 nibbles):
        byte[i] = value[2i] (low nibble) | value[2i+1] (high nibble)

    GGUF Q4_0 packing (per 16-byte group = 32 nibbles):
        byte[j] = value[j] (low nibble) | value[j+16] (high nibble)
    """
    # onnx_packed shape: (..., 16) where last dim = 16 packed bytes = 32 nibbles
    low = onnx_packed & 0x0F   # values at even positions: 0, 2, 4, ..., 30
    high = onnx_packed >> 4    # values at odd positions:  1, 3, 5, ..., 31

    # Reconstruct all 32 values in sequential order
    values = np.empty((*onnx_packed.shape[:-1], GGML_QK4_0), dtype=np.uint8)
    values[..., 0::2] = low
    values[..., 1::2] = high

    # Repack in Q4_0 order: byte[j] = values[j] | (values[j+16] << 4)
    return (values[..., :16] | (values[..., 16:] << 4)).astype(np.uint8)


def require_block_aligned_width(name: str, k: int, block_size: int, quant_type: str) -> None:
    if k % block_size != 0:
        raise ValueError(f"{name}: {quant_type} requires K divisible by {block_size}, got K={k}")


def quantize_f32_to_q4_0(data: np.ndarray[Any, Any]) -> np.ndarray[Any, Any]:
    """Quantize an FP32 array to raw Q4_0 bytes (fallback for non-32 block sizes).

    Uses the same quantization formula as ggml: d = max / -8.0, nibble = round(x/d) + 8.
    """
    if data.ndim == 1:
        data = data.reshape(1, -1)
    n, k = data.shape
    data = data.astype(np.float32)
    require_block_aligned_width("tensor", k, GGML_QK4_0, "Q4_0")

    num_blocks = k // GGML_QK4_0
    blocks_data = data.reshape(n * num_blocks, GGML_QK4_0)

    # Find value with largest absolute magnitude per block (preserving sign)
    idx = np.argmax(np.abs(blocks_data), axis=-1)
    max_vals = blocks_data[np.arange(len(blocks_data)), idx]

    # Compute scale: d = max / -8.0 (ggml convention)
    d = max_vals / -8.0
    # Avoid division by zero
    id_vals = np.where(d != 0, 1.0 / d, 0.0)

    # Quantize to unsigned nibbles [0, 15]
    quantized = (blocks_data * id_vals[:, None] + 8.5).astype(np.int32)
    quantized = np.clip(quantized, 0, 15).astype(np.uint8)

    # Pack in Q4_0 nibble order: byte[j] = quant[j] | (quant[j+16] << 4)
    q4_packed = quantized[:, :16] | (quantized[:, 16:] << 4)

    # Build blocks: [fp16 scale (2 bytes) | packed nibbles (16 bytes)]
    scales_fp16 = d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    blocks = np.concatenate([scales_fp16, q4_packed], axis=-1)  # shape: (n*num_blocks, 18)
    return blocks.reshape(-1).astype(np.uint8)


def matmul_nbits_to_q4_0(bundle: AionOnnxBundle, prefix: str) -> tuple[np.ndarray[Any, Any], list[int]]:
    """Convert ONNX MatMulNBits weights directly to Q4_0 raw bytes.

    Returns (raw_q4_0_data, logical_shape) where logical_shape is [N, K].
    When the ONNX block_size matches Q4_0 (32), this is a direct byte-repack
    with no dequantization — preserving the original precision exactly.
    """
    node = bundle.quant_node_for(prefix)
    attrs = node_attrs(node)
    weight = bundle.tensor_array(node.input[1])
    scales = bundle.tensor_array(node.input[2])
    bits = int(attrs["bits"])
    block_size = int(attrs["block_size"])
    k = int(attrs["K"])
    n = int(attrs["N"])
    require_block_aligned_width(prefix, k, GGML_QK4_0, "Q4_0")

    if bits != 4 or block_size != GGML_QK4_0:
        # Block size mismatch: dequantize then re-quantize to Q4_0
        LOGGER.info("  %s: block_size=%d bits=%d, using dequant+requant fallback", prefix, block_size, bits)
        dequant = dequant_matmul_nbits(bundle, prefix)
        return quantize_f32_to_q4_0(dequant), [n, k]

    # Direct path: repack ONNX nibbles into Q4_0 block layout
    num_blocks_per_row = (k + block_size - 1) // block_size
    scales_fp16 = scales.reshape(n, num_blocks_per_row).astype(np.float16)
    weight_flat = weight.reshape(n, num_blocks_per_row, 16)

    # Repack nibble order
    q4_nibbles = repack_onnx_nibbles_to_q4_0(weight_flat)  # (n, num_blocks_per_row, 16)

    # ONNX uses positive scale with zero_point=8: dequant = (nibble - 8) * scale
    # GGUF Q4_0 uses: dequant = (nibble - 8) * d, where d = max / -8.0
    # For positive scale: ONNX nibble 0 → -8*scale (most negative)
    # For GGUF with d = -scale: nibble 0 → (0-8)*(-scale) = +8*scale (most positive) — WRONG
    # So we must use d = +scale (same sign) to preserve the mapping.
    # GGUF allows both positive and negative d; with d = scale > 0 the formula is identical.
    scales_bytes = scales_fp16.view(np.uint8).reshape(n, num_blocks_per_row, 2)

    # Build Q4_0 blocks: [fp16_d (2 bytes) | nibbles (16 bytes)] per block
    blocks = np.concatenate([scales_bytes, q4_nibbles], axis=-1)  # (n, num_blocks, 18)
    return blocks.reshape(-1).astype(np.uint8), [n, k]


def matmul_nbits_to_q8_0(bundle: AionOnnxBundle, prefix: str) -> tuple[np.ndarray[Any, Any], list[int]]:
    """Convert ONNX MatMulNBits 8-bit weights directly to Q8_0 raw bytes.

    Returns (raw_q8_0_data, logical_shape) where logical_shape is [N, K].
    ONNX stores unsigned uint8 codes with zero_point=128; GGUF Q8_0 uses signed int8
    with scale d. The conversion is: int8_val = uint8_code - 128, d = scale.
    """
    node = bundle.quant_node_for(prefix)
    attrs = node_attrs(node)
    weight = bundle.tensor_array(node.input[1])
    scales = bundle.tensor_array(node.input[2])
    bits = int(attrs["bits"])
    block_size = int(attrs["block_size"])
    k = int(attrs["K"])
    n = int(attrs["N"])
    require_block_aligned_width(prefix, k, GGML_QK8_0, "Q8_0")

    if bits != 8 or block_size != GGML_QK8_0:
        raise ValueError(f"{prefix}: expected 8-bit block_size=32 for Q8_0, got bits={bits} block_size={block_size}")

    # ONNX stores uint8 codes; Q8_0 uses int8 = uint8 - 128
    num_blocks_per_row = (k + block_size - 1) // block_size
    scales_fp16 = scales.reshape(n, num_blocks_per_row).astype(np.float16)
    weight_flat = weight.reshape(n, num_blocks_per_row, GGML_QK8_0)

    # Convert unsigned [0,255] to signed [-128,127]
    weight_signed = (weight_flat.astype(np.int16) - 128).astype(np.int8)

    # Build Q8_0 blocks: [fp16_d (2 bytes) | int8 values (32 bytes)] per block
    scales_bytes = scales_fp16.view(np.uint8).reshape(n, num_blocks_per_row, 2)
    quant_bytes = weight_signed.view(np.uint8).reshape(n, num_blocks_per_row, GGML_QK8_0)
    blocks = np.concatenate([scales_bytes, quant_bytes], axis=-1)  # (n, num_blocks, 34)
    return blocks.reshape(-1).astype(np.uint8), [n, k]


def matmul_nbits_to_gguf(bundle: AionOnnxBundle, prefix: str) -> tuple[np.ndarray[Any, Any], list[int], str]:
    """Convert ONNX MatMulNBits weights to the best-matching GGUF quantization.

    Returns (raw_data, logical_shape, quant_type) where quant_type is 'q4_0' or 'q8_0'.
    8-bit ONNX tensors map to Q8_0; 4-bit tensors map to Q4_0.
    """
    node = bundle.quant_node_for(prefix)
    attrs = node_attrs(node)
    bits = int(attrs["bits"])

    if bits == 8 and int(attrs["block_size"]) == GGML_QK8_0:
        raw, shape = matmul_nbits_to_q8_0(bundle, prefix)
        return raw, shape, "q8_0"
    else:
        raw, shape = matmul_nbits_to_q4_0(bundle, prefix)
        return raw, shape, "q4_0"


def add_tensor_q4_0(writer: gguf.GGUFWriter, name: str, raw_data: np.ndarray[Any, Any], shape: list[int], dry_run: bool) -> None:
    """Write a pre-quantized Q4_0 tensor to the GGUF file."""
    n, k = shape
    require_block_aligned_width(name, k, GGML_QK4_0, "Q4_0")
    LOGGER.info("%-38s %s Q4_0 (%d bytes)", name, shape, len(raw_data))
    if not dry_run:
        # gguf-py expects the data shaped as (rows, bytes_per_row) where
        # bytes_per_row must be a multiple of the Q4_0 block size (18 bytes).
        num_blocks_per_row = k // GGML_QK4_0
        bytes_per_row = num_blocks_per_row * GGML_Q4_0_BLOCK_BYTES
        data_2d = raw_data.reshape(n, bytes_per_row)
        writer.add_tensor(name, data_2d, raw_dtype=gguf.GGMLQuantizationType.Q4_0)


def add_tensor_q8_0(writer: gguf.GGUFWriter, name: str, raw_data: np.ndarray[Any, Any], shape: list[int], dry_run: bool) -> None:
    """Write a pre-quantized Q8_0 tensor to the GGUF file."""
    n, k = shape
    require_block_aligned_width(name, k, GGML_QK8_0, "Q8_0")
    LOGGER.info("%-38s %s Q8_0 (%d bytes)", name, shape, len(raw_data))
    if not dry_run:
        num_blocks_per_row = k // GGML_QK8_0
        bytes_per_row = num_blocks_per_row * GGML_Q8_0_BLOCK_BYTES
        data_2d = raw_data.reshape(n, bytes_per_row)
        writer.add_tensor(name, data_2d, raw_dtype=gguf.GGMLQuantizationType.Q8_0)


def add_tensor_quant(writer: gguf.GGUFWriter, name: str, raw_data: np.ndarray[Any, Any], shape: list[int], quant_type: str, dry_run: bool) -> None:
    """Write a pre-quantized tensor (Q4_0 or Q8_0) to the GGUF file."""
    if quant_type == "q8_0":
        add_tensor_q8_0(writer, name, raw_data, shape, dry_run)
    else:
        add_tensor_q4_0(writer, name, raw_data, shape, dry_run)


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
    writer.add_description("Aion Edge ONNX bundle converted directly to Q4_0 GGUF")
    writer.add_file_type(int(gguf.LlamaFileType.MOSTLY_Q4_0))
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

    # Pack ONNX quantized weights into matching GGUF format (Q8_0 for 8-bit, Q4_0 for 4-bit)
    raw, shape, qt = matmul_nbits_to_gguf(bundle, "lm_head")
    add_tensor_quant(writer, TENSOR_NAMES["token_embd"], raw, shape, qt, args.dry_run)

    for layer in range(layers_to_write):
        prefix = f"model.layers.{layer}"
        add_tensor(writer, TENSOR_NAMES["attn_norm"].format(layer=layer), norm_tensor(bundle, f"{prefix}.input_layernorm.weight"), args.dry_run)
        raw, shape, qt = matmul_nbits_to_gguf(bundle, f"{prefix}.self_attn.q_proj")
        add_tensor_quant(writer, TENSOR_NAMES["attn_q"].format(layer=layer), raw, shape, qt, args.dry_run)
        add_tensor(writer, TENSOR_NAMES["attn_q_norm"].format(layer=layer), norm_tensor(bundle, f"{prefix}.self_attn.q_norm.weight"), args.dry_run)
        raw, shape, qt = matmul_nbits_to_gguf(bundle, f"{prefix}.self_attn.k_proj")
        add_tensor_quant(writer, TENSOR_NAMES["attn_k"].format(layer=layer), raw, shape, qt, args.dry_run)
        add_tensor(writer, TENSOR_NAMES["attn_k_norm"].format(layer=layer), norm_tensor(bundle, f"{prefix}.self_attn.k_norm.weight"), args.dry_run)
        raw, shape, qt = matmul_nbits_to_gguf(bundle, f"{prefix}.self_attn.v_proj")
        add_tensor_quant(writer, TENSOR_NAMES["attn_v"].format(layer=layer), raw, shape, qt, args.dry_run)
        raw, shape, qt = matmul_nbits_to_gguf(bundle, f"{prefix}.self_attn.o_proj")
        add_tensor_quant(writer, TENSOR_NAMES["attn_out"].format(layer=layer), raw, shape, qt, args.dry_run)
        add_tensor(writer, TENSOR_NAMES["ffn_norm"].format(layer=layer), norm_tensor(bundle, f"{prefix}.post_attention_layernorm.weight"), args.dry_run)
        raw, shape, qt = matmul_nbits_to_gguf(bundle, f"{prefix}.mlp.gate_proj")
        add_tensor_quant(writer, TENSOR_NAMES["ffn_gate"].format(layer=layer), raw, shape, qt, args.dry_run)
        raw, shape, qt = matmul_nbits_to_gguf(bundle, f"{prefix}.mlp.down_proj")
        add_tensor_quant(writer, TENSOR_NAMES["ffn_down"].format(layer=layer), raw, shape, qt, args.dry_run)
        raw, shape, qt = matmul_nbits_to_gguf(bundle, f"{prefix}.mlp.up_proj")
        add_tensor_quant(writer, TENSOR_NAMES["ffn_up"].format(layer=layer), raw, shape, qt, args.dry_run)

    if layers_to_write != layer_count:
        LOGGER.warning("partial conversion requested: wrote %d/%d layers", layers_to_write, layer_count)

    add_tensor(writer, TENSOR_NAMES["output_norm"], norm_tensor(bundle, "model.norm.weight"), args.dry_run)

    if args.dry_run:
        LOGGER.info("dry run complete; no GGUF file written")
        return

    LOGGER.info("writing GGUF: %s", args.outfile)
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file(progress=True)
    writer.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert Edge Aion ONNX bundle to Q4_0 Qwen3 GGUF")
    parser.add_argument("model_dir", type=Path, help="directory containing model.onnx, model.onnx.data, tokenizer.json, genai_config.json")
    parser.add_argument("--outfile", type=Path, default=Path("aion-q40.gguf"), help="output GGUF path")
    parser.add_argument("--dry-run", action="store_true", help="parse and log tensor shapes without writing a file")
    parser.add_argument("--max-layers", type=int, default=None, help="convert only the first N layers for smoke testing")
    parser.add_argument("--verbose", action="store_true", help="enable debug logging")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(levelname)s:%(name)s:%(message)s")
    convert(args)


if __name__ == "__main__":
    main()