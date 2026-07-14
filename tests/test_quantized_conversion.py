from __future__ import annotations

import importlib.util
from pathlib import Path
import unittest

import numpy as np
import onnx


PROJECT_DIR = Path(__file__).resolve().parents[1]
CONVERTER_PATH = PROJECT_DIR / "scripts" / "convert_aion_onnx_to_gguf.py"
SPEC = importlib.util.spec_from_file_location("convert_aion_onnx_to_gguf", CONVERTER_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load converter from {CONVERTER_PATH}")
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)


class FakeBundle:
    def __init__(self, bits: int, k: int, block_size: int, weight: np.ndarray, scales: np.ndarray):
        self.node = onnx.helper.make_node(
            "MatMulNBits",
            ["input", "weight", "scales"],
            ["output"],
            K=k,
            N=1,
            bits=bits,
            block_size=block_size,
        )
        self.arrays = {"weight": weight, "scales": scales}

    def quant_node_for(self, prefix: str) -> onnx.NodeProto:
        return self.node

    def tensor_array(self, name: str) -> np.ndarray:
        return self.arrays[name]


class QuantizedConversionTests(unittest.TestCase):
    def test_q4_direct_repack_preserves_codes_and_scale(self) -> None:
        values = (np.arange(32, dtype=np.uint8) % 16).reshape(1, 1, 32)
        weight = (values[..., 0::2] | (values[..., 1::2] << 4)).astype(np.uint8)
        scales = np.array([[0.25]], dtype=np.float16)
        bundle = FakeBundle(4, 32, 32, weight, scales)

        raw, shape = converter.matmul_nbits_to_q4_0(bundle, "weight")

        expected_codes = values[..., :16] | (values[..., 16:] << 4)
        expected = np.concatenate(
            [scales.view(np.uint8).reshape(1, 1, 2), expected_codes], axis=-1
        ).reshape(-1)
        self.assertEqual(shape, [1, 32])
        np.testing.assert_array_equal(raw, expected)

    def test_q8_direct_repack_recenters_codes_and_preserves_scale(self) -> None:
        weight = np.arange(112, 144, dtype=np.uint8).reshape(1, 1, 32)
        scales = np.array([[0.125]], dtype=np.float16)
        bundle = FakeBundle(8, 32, 32, weight, scales)

        raw, shape = converter.matmul_nbits_to_q8_0(bundle, "weight")

        signed = (weight.astype(np.int16) - 128).astype(np.int8).view(np.uint8)
        expected = np.concatenate(
            [scales.view(np.uint8).reshape(1, 1, 2), signed], axis=-1
        ).reshape(-1)
        self.assertEqual(shape, [1, 32])
        np.testing.assert_array_equal(raw, expected)

    def test_q4_fallback_rejects_unaligned_logical_width(self) -> None:
        weight = np.zeros((1, 3, 8), dtype=np.uint8)
        scales = np.ones((1, 3), dtype=np.float16)
        bundle = FakeBundle(4, 48, 16, weight, scales)

        with self.assertRaisesRegex(ValueError, "Q4_0 requires K divisible by 32, got K=48"):
            converter.matmul_nbits_to_q4_0(bundle, "weight")

    def test_q8_direct_conversion_rejects_unaligned_logical_width(self) -> None:
        weight = np.zeros((1, 2, 32), dtype=np.uint8)
        scales = np.ones((1, 2), dtype=np.float16)
        bundle = FakeBundle(8, 48, 32, weight, scales)

        with self.assertRaisesRegex(ValueError, "Q8_0 requires K divisible by 32, got K=48"):
            converter.matmul_nbits_to_q8_0(bundle, "weight")


if __name__ == "__main__":
    unittest.main()