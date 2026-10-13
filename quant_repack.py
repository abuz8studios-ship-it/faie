"""ggml -> FAIE-kernel-layout repack for K-quant blocks.

ROOT CAUSE (found 2026-07-26): every FAIE decode of Q2_K / Q3_K — CUDA kernels
AND fast_dequant.py — assumed a SEQUENTIAL 2-bit packing (element e <- qs byte
e//4, bits 2*(e%4)) plus, for Q3_K, a wrong 6-bit scale packing, a wrong sign
convention (two's complement instead of ggml's offset-32), and a wrong hmask
bit order. ggml's real layout is BIT-PLANED:

  Q2_K/Q3_K qs : element e <- qs byte (e//128)*32 + (e%32), bits 2*((e%128)//32)
  Q3_K hmask   : element e <- hmask byte e%32, bit e//32
  Q3_K scales  : 16 x 6-bit, value = packed - 32 (offset binary), packed as
                 low nibbles in bytes 0..7 (j<8 low nib, j>=8 high nib) and
                 high 2 bits in bytes 8..11 (j%4 selects byte, 2*(j//4) shift)

The kernel and its PyTorch mirror shared the wrong assumption, so every
component check agreed at cos 1.000000 while decoding garbage — only an
INDEPENDENT reference (gguf.quants.dequantize) exposed it: cos 0.06 (Q2_K),
0.003 (Q3_K), vs 1.0 for Q4_0 (the only one that had been independently checked).

Instead of rewriting ~16 tuned CUDA kernels, this module PERMUTES the raw block
bytes at upload time so the kernels' self-consistent decode emits the correct
values. Scale association is position-based and identical in both layouts for
Q2_K (bytes 0..15 untouched); Q3_K scales are re-encoded; d/dmin stay put.

Validation (run this file): for real tensors from the GGUF,
  faie_old_decode(repack(raw)) must equal gguf.quants.dequantize(raw) exactly.
"""
import numpy as np

QK_K = 256
Q2K_BS = 84
Q3K_BS = 110

# element permutation: FAIE position e reads the value that ggml stores for
# position e, so we need ggml_value[e] placed at FAIE's slot for e.
# Both decoders map "value stream index" -> output element the same way
# (sequentially for FAIE, plane-order for ggml), so:
#   vals[e]  = ggml 2-bit value of element e   (extract with plane order)
#   new_qs[b]= vals[4b] | vals[4b+1]<<2 | vals[4b+2]<<4 | vals[4b+3]<<6


def _extract_2bit_ggml(qs):
    """qs: (n, 64) uint8 -> (n, 256) uint8 element values in ggml element order."""
    n = qs.shape[0]
    q = qs.reshape(n, 2, 1, 32)                      # halves of 32 bytes
    shifts = np.array([0, 2, 4, 6], dtype=np.uint8).reshape(1, 1, 4, 1)
    v = (q >> shifts) & np.uint8(3)                  # (n, 2, 4, 32) = (half, shift, byte)
    return v.reshape(n, QK_K)


def _pack_2bit_faie(vals):
    """vals: (n, 256) uint8 -> (n, 64) uint8 in FAIE sequential layout."""
    n = vals.shape[0]
    v = vals.reshape(n, 64, 4).astype(np.uint8)
    return (v[:, :, 0] | (v[:, :, 1] << np.uint8(2)) |
            (v[:, :, 2] << np.uint8(4)) | (v[:, :, 3] << np.uint8(6)))


def repack_q2k(raw):
    """raw: bytes or uint8 ndarray of whole Q2_K blocks -> repacked uint8 array."""
    data = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) \
        else raw.reshape(-1).view(np.uint8)
    nb = data.size // Q2K_BS
    assert nb * Q2K_BS == data.size, f"Q2_K repack: {data.size} not multiple of {Q2K_BS}"
    blocks = data[:nb * Q2K_BS].reshape(nb, Q2K_BS).copy()
    vals = _extract_2bit_ggml(blocks[:, 16:80])
    blocks[:, 16:80] = _pack_2bit_faie(vals)
    # scales (0:16) and d/dmin (80:84) are read identically by both layouts
    return blocks.reshape(-1)


def repack_q3k(raw):
    """raw: bytes or uint8 ndarray of whole Q3_K blocks -> repacked uint8 array."""
    data = np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) \
        else raw.reshape(-1).view(np.uint8)
    nb = data.size // Q3K_BS
    assert nb * Q3K_BS == data.size, f"Q3_K repack: {data.size} not multiple of {Q3K_BS}"
    blocks = data[:nb * Q3K_BS].reshape(nb, Q3K_BS).copy()

    hmask = blocks[:, 0:32]
    qs = blocks[:, 32:96]
    sc = blocks[:, 96:108].astype(np.uint16)

    # --- qs: same 2-bit plane permutation as Q2_K
    vals = _extract_2bit_ggml(qs)
    blocks[:, 32:96] = _pack_2bit_faie(vals)

    # --- hmask: ggml bit for element e = (hmask[e%32] >> (e//32)) & 1
    #            FAIE expects       bit e at byte e//8, bit e%8
    n = blocks.shape[0]
    h = hmask.reshape(n, 1, 32) >> np.arange(8, dtype=np.uint8).reshape(1, 8, 1)
    h = (h & np.uint8(1)).reshape(n, QK_K)           # (n, e) with e = bit*32 + byte
    hb = h.reshape(n, 32, 8)
    weights = (np.uint8(1) << np.arange(8, dtype=np.uint8)).reshape(1, 1, 8)
    blocks[:, 0:32] = (hb * weights).sum(axis=2).astype(np.uint8)

    # --- scales: decode ggml (offset-32) -> re-encode FAIE (two's complement, own packing)
    u = np.zeros((n, 16), dtype=np.int16)
    for j in range(4):
        u[:, j]      = (sc[:, j] & 0xF)      | ((sc[:, 8 + j] & 3) << 4)
        u[:, 4 + j]  = (sc[:, 4 + j] & 0xF)  | (((sc[:, 8 + j] >> 2) & 3) << 4)
        u[:, 8 + j]  = ((sc[:, j] >> 4) & 0xF)     | (((sc[:, 8 + j] >> 4) & 3) << 4)
        u[:, 12 + j] = ((sc[:, 4 + j] >> 4) & 0xF) | (((sc[:, 8 + j] >> 6) & 3) << 4)
    v = u - 32                                        # true signed scale value
    f = (v & 0x3F).astype(np.uint16)                  # two's-complement 6-bit for FAIE
    out = np.zeros((n, 12), dtype=np.uint16)
    for i in range(4):
        out[:, i]     = (f[:, i] & 0x3F)     | ((f[:, 8 + i] >> 4) << 6)
        out[:, 4 + i] = (f[:, 4 + i] & 0x3F) | ((f[:, 12 + i] >> 4) << 6)
        out[:, 8 + i] = (f[:, 8 + i] & 0xF)  | ((f[:, 12 + i] & 0xF) << 4)
    blocks[:, 96:108] = out.astype(np.uint8)
    # d (108:110) unchanged
    return blocks.reshape(-1)


def repack_for_kernel(raw, qtype):
    """Dispatch: repack Q2_K/Q3_K raw block bytes for the FAIE CUDA kernels.
    Everything else passes through unchanged."""
    if qtype == "Q2_K":
        return repack_q2k(raw)
    if qtype == "Q3_K":
        return repack_q3k(raw)
    return np.frombuffer(raw, dtype=np.uint8) if not isinstance(raw, np.ndarray) \
        else raw.reshape(-1).view(np.uint8)


# ---------------------------------------------------------------------------
# The OLD (wrong-layout) FAIE decoders, kept verbatim for validation only:
# faie_decode(repack(raw)) must equal gguf.quants.dequantize(raw).
# ---------------------------------------------------------------------------

def _dequant_faie_q2k(raw_data, n_elements):
    data = np.frombuffer(raw_data, dtype=np.uint8) if not isinstance(raw_data, np.ndarray) \
        else raw_data.view(np.uint8)
    n_blocks = n_elements // QK_K
    blocks = data[:n_blocks * Q2K_BS].reshape(n_blocks, Q2K_BS)
    scales_raw = blocks[:, :16]
    qs_raw = blocks[:, 16:80]
    d = blocks[:, 80:82].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    dmin = blocks[:, 82:84].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    dl = (scales_raw & 0x0F).astype(np.float32) * d[:, None]
    ml = (scales_raw >> 4).astype(np.float32) * dmin[:, None]
    quants = np.zeros((n_blocks, QK_K), dtype=np.float32)
    quants[:, 0::4] = (qs_raw & 0x03).astype(np.float32)
    quants[:, 1::4] = ((qs_raw >> 2) & 0x03).astype(np.float32)
    quants[:, 2::4] = ((qs_raw >> 4) & 0x03).astype(np.float32)
    quants[:, 3::4] = ((qs_raw >> 6) & 0x03).astype(np.float32)
    result = np.repeat(dl, 16, axis=1) * quants - np.repeat(ml, 16, axis=1)
    return result.ravel()[:n_elements]


def _dequant_faie_q3k(raw_data, n_elements):
    data = np.frombuffer(raw_data, dtype=np.uint8) if not isinstance(raw_data, np.ndarray) \
        else raw_data.view(np.uint8)
    n_blocks = n_elements // QK_K
    blocks = data[:n_blocks * Q3K_BS].reshape(n_blocks, Q3K_BS)
    hmask_raw = blocks[:, 0:32]
    qs_raw = blocks[:, 32:96]
    scales_raw = blocks[:, 96:108]
    d = blocks[:, 108:110].copy().view(np.float16).astype(np.float32).reshape(n_blocks)
    scales = np.zeros((n_blocks, 16), dtype=np.int32)
    for i in range(8):
        s = scales_raw[:, i].astype(np.int32) & 0x3F
        s[s >= 32] -= 64
        scales[:, i] = s
    for i in range(4):
        lo = scales_raw[:, 8 + i].astype(np.int32) & 0x0F
        hi = (scales_raw[:, 8 + i].astype(np.int32) >> 4) & 0x0F
        s1 = lo | ((scales_raw[:, i].astype(np.int32) >> 6) << 4)
        s2 = hi | ((scales_raw[:, i + 4].astype(np.int32) >> 6) << 4)
        s1[s1 >= 32] -= 64
        s2[s2 >= 32] -= 64
        scales[:, 8 + i] = s1
        scales[:, 12 + i] = s2
    dl = scales.astype(np.float32) * d[:, None]
    quants = np.zeros((n_blocks, QK_K), dtype=np.int32)
    quants[:, 0::4] = (qs_raw & 0x03).astype(np.int32)
    quants[:, 1::4] = ((qs_raw >> 2) & 0x03).astype(np.int32)
    quants[:, 2::4] = ((qs_raw >> 4) & 0x03).astype(np.int32)
    quants[:, 3::4] = ((qs_raw >> 6) & 0x03).astype(np.int32)
    hmask_bits = np.zeros((n_blocks, QK_K), dtype=np.int32)
    for bit_idx in range(QK_K):
        hmask_bits[:, bit_idx] = (hmask_raw[:, bit_idx // 8].astype(np.int32) >> (bit_idx % 8)) & 1
    q_val = quants - 4 + hmask_bits * 4
    result = np.repeat(dl, 16, axis=1) * q_val.astype(np.float32)
    return result.ravel()[:n_elements]


if __name__ == "__main__":
    import sys, io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace", line_buffering=True)
    from gguf import GGUFReader
    from gguf.quants import dequantize

    MODEL = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
    reader = GGUFReader(MODEL, "r")
    tensors = {t.name: t for t in reader.tensors}

    CASES = [
        ("blk.8.attn_q_a.weight", "Q2_K"),
        ("blk.8.attn_q_b.weight", "Q2_K"),
        ("blk.8.attn_kv_a_mqa.weight", "Q2_K"),
        ("blk.8.attn_v_b.weight", "Q2_K"),
        ("blk.0.ffn_gate.weight", "Q2_K"),
        ("blk.8.attn_output.weight", "Q3_K"),
        ("blk.0.ffn_down.weight", "Q3_K"),
        ("blk.40.attn_output.weight", "Q3_K"),
        ("blk.77.attn_q_b.weight", "Q2_K"),
    ]
    all_ok = True
    for name, qt in CASES:
        t = tensors[name]
        ne = [int(s) for s in t.shape]
        n_elem = int(np.prod(ne))
        raw = np.ascontiguousarray(t.data).reshape(-1).view(np.uint8)
        ref = dequantize(t.data, t.tensor_type).astype(np.float32).reshape(-1)[:n_elem]
        rp = repack_for_kernel(raw, qt)
        if qt == "Q2_K":
            got = _dequant_faie_q2k(rp, n_elem)
        else:
            got = _dequant_faie_q3k(rp, n_elem)
        got = np.asarray(got, dtype=np.float32)
        maxd = float(np.abs(ref - got).max())
        ok = maxd == 0.0
        all_ok &= ok
        print(f"{name:32s} {qt}  n={n_elem:>12,}  maxdiff={maxd:.3e}  {'EXACT' if ok else 'FAIL'}")
    print("ALL EXACT" if all_ok else "VALIDATION FAILED")
    sys.exit(0 if all_ok else 1)
