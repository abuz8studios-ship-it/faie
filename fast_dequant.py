"""
Vectorized GGUF dequantization.

FIXED 2026-07-26: the previous implementation assumed a SEQUENTIAL 2-bit packing
(element e <- byte e//4, bits 2*(e%4)); ggml's real K-quant layout is bit-planed
(element e <- byte (e//128)*32 + e%32, bits 2*((e%128)//32)), and Q3_K
additionally uses offset-32 scales in a different 6-bit packing plus hmask bit
order (byte e%32, bit e//32). Independent check vs gguf.quants.dequantize showed
cos 0.06 (Q2_K) / 0.003 (Q3_K) for the old code — deterministic garbage.

These functions now delegate to gguf.quants.dequantize (the same code the
llama.cpp project maintains next to the C reference), keeping the old
signatures. See quant_repack.py for the matching CUDA-kernel-side repack.
"""

import numpy as np
from gguf.constants import GGMLQuantizationType
from gguf.quants import dequantize

QK_K = 256
Q2_K_BLOCK_SIZE = 84
Q3_K_BLOCK_SIZE = 110
Q6_K_BLOCK_SIZE = 210


def _as_u8(raw_data):
    if isinstance(raw_data, np.ndarray):
        return raw_data.reshape(-1).view(np.uint8)
    return np.frombuffer(raw_data, dtype=np.uint8)


def _dequant(raw_data, n_elements, block_bytes, qtype):
    data = _as_u8(raw_data)
    n_blocks = n_elements // QK_K
    if data.size < n_blocks * block_bytes:
        n_blocks = data.size // block_bytes
    blocks = data[:n_blocks * block_bytes]
    out = dequantize(blocks, qtype).astype(np.float32, copy=False).reshape(-1)
    return out[:n_elements]


def dequant_q2k_fast(raw_data, n_elements):
    """Correct Q2_K dequantization (ggml layout)."""
    return _dequant(raw_data, n_elements, Q2_K_BLOCK_SIZE, GGMLQuantizationType.Q2_K)


def dequant_q3k_fast(raw_data, n_elements):
    """Correct Q3_K dequantization (ggml layout)."""
    return _dequant(raw_data, n_elements, Q3_K_BLOCK_SIZE, GGMLQuantizationType.Q3_K)


def dequant_q6k_fast(raw_data, n_elements):
    """Correct Q6_K dequantization (ggml layout)."""
    return _dequant(raw_data, n_elements, Q6_K_BLOCK_SIZE, GGMLQuantizationType.Q6_K)
