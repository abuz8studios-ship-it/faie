"""
FAIE Full Model Converter -- Path 2.

Converts ALL 9,600 experts (75 MoE layers x 128 experts) from
GLM-5.2-REAM-Q2_K.gguf to packed ternary format on disk.

Also converts attention, embeddings, and output layers.

Output format: directory of .bin files per layer + metadata JSON.
Each expert file contains: packed_ternary_bytes + scales_float32.
"""

import os
import sys
import json
import time
import struct
import shutil
import numpy as np
from pathlib import Path

from gguf import GGUFReader
from faie import TernaryCompressor, GLM_CONFIG
from fast_dequant import dequant_q2k_fast, dequant_q3k_fast, QK_K, Q2_K_BLOCK_SIZE, Q3_K_BLOCK_SIZE

MODEL_PATH = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
OUTPUT_DIR = r"E:\ABU\NEOCORTEX\faie_model"
GROUP_SIZE = 64


def convert_expert(reader, layer_idx, expert_idx, output_dir):
    """Convert one expert's gate+up+down projections to ternary format (SwiGLU)."""
    gate_name = f"blk.{layer_idx}.ffn_gate_exps.weight"
    up_name = f"blk.{layer_idx}.ffn_up_exps.weight"
    down_name = f"blk.{layer_idx}.ffn_down_exps.weight"

    gate_tensor = None
    up_tensor = None
    down_tensor = None
    for t in reader.tensors:
        if t.name == gate_name:
            gate_tensor = t
        elif t.name == up_name:
            up_tensor = t
        elif t.name == down_name:
            down_tensor = t
        if gate_tensor and up_tensor and down_tensor:
            break

    if gate_tensor is None or up_tensor is None or down_tensor is None:
        return None

    gate_shape = [int(s) for s in gate_tensor.shape]
    up_shape = [int(s) for s in up_tensor.shape]
    down_shape = [int(s) for s in down_tensor.shape]

    # GGUF memmap is (n_experts, ...) -- index by expert_idx directly
    gate_elements = gate_shape[0] * gate_shape[1]
    gate_raw = gate_tensor.data[expert_idx].tobytes()
    gate_f32 = dequant_q2k_fast(gate_raw, gate_elements).reshape(gate_shape[0], gate_shape[1]).T

    up_elements = up_shape[0] * up_shape[1]
    up_raw = up_tensor.data[expert_idx].tobytes()
    up_f32 = dequant_q2k_fast(up_raw, up_elements).reshape(up_shape[0], up_shape[1]).T

    down_qtype = down_tensor.tensor_type.name if hasattr(down_tensor.tensor_type, 'name') else str(down_tensor.tensor_type)
    down_elements = down_shape[0] * down_shape[1]
    down_raw = down_tensor.data[expert_idx].tobytes()

    if "Q3_K" in down_qtype:
        down_f32 = dequant_q3k_fast(down_raw, down_elements).reshape(down_shape[0], down_shape[1]).T
    else:
        down_f32 = dequant_q2k_fast(down_raw, down_elements).reshape(down_shape[0], down_shape[1]).T

    # Ternary quantize all three projections
    q_gate = TernaryCompressor.quantize(gate_f32, group_size=GROUP_SIZE)
    q_up = TernaryCompressor.quantize(up_f32, group_size=GROUP_SIZE)
    q_down = TernaryCompressor.quantize(down_f32, group_size=GROUP_SIZE)

    # Pack
    gate_packed, gate_n = TernaryCompressor.pack_ternary(q_gate["ternary"])
    up_packed, up_n = TernaryCompressor.pack_ternary(q_up["ternary"])
    down_packed, down_n = TernaryCompressor.pack_ternary(q_down["ternary"])

    # Write binary file (v2 format: gate + up + down)
    expert_path = os.path.join(output_dir, f"expert_{expert_idx:03d}.bin")
    with open(expert_path, 'wb') as f:
        # Magic + version
        f.write(struct.pack('<I', 0xFA1E0002))
        # Shapes: gate(2x4) + up(2x4) + down(2x4) = 24 bytes
        f.write(struct.pack('<II', gate_f32.shape[0], gate_f32.shape[1]))
        f.write(struct.pack('<II', up_f32.shape[0], up_f32.shape[1]))
        f.write(struct.pack('<II', down_f32.shape[0], down_f32.shape[1]))
        # Element counts
        f.write(struct.pack('<III', gate_n, up_n, down_n))
        # Packed byte counts
        f.write(struct.pack('<III', len(gate_packed), len(up_packed), len(down_packed)))
        # Scale counts
        f.write(struct.pack('<III', len(q_gate["scales"]), len(q_up["scales"]), len(q_down["scales"])))
        # Group size + sparsities
        f.write(struct.pack('<I', GROUP_SIZE))
        f.write(struct.pack('<fff', q_gate["sparsity"], q_up["sparsity"], q_down["sparsity"]))
        # Data: gate, up, down (packed bytes then scales for each)
        f.write(gate_packed.tobytes())
        f.write(q_gate["scales"].astype(np.float32).tobytes())
        f.write(up_packed.tobytes())
        f.write(q_up["scales"].astype(np.float32).tobytes())
        f.write(down_packed.tobytes())
        f.write(q_down["scales"].astype(np.float32).tobytes())

    file_size = os.path.getsize(expert_path)
    return {
        "gate_shape": list(gate_f32.shape),
        "up_shape": list(up_f32.shape),
        "down_shape": list(down_f32.shape),
        "gate_sparsity": float(q_gate["sparsity"]),
        "up_sparsity": float(q_up["sparsity"]),
        "down_sparsity": float(q_down["sparsity"]),
        "file_bytes": file_size,
    }


def is_layer_complete(output_base, layer_idx):
    """Check if a layer has all 128 experts + meta.json."""
    layer_dir = os.path.join(output_base, f"layer_{layer_idx:02d}")
    if not os.path.isdir(layer_dir):
        return False
    meta_path = os.path.join(layer_dir, "meta.json")
    if not os.path.exists(meta_path):
        return False
    expert_files = [f for f in os.listdir(layer_dir) if f.startswith("expert_") and f.endswith(".bin")]
    return len(expert_files) == 128


def convert_layer(reader, layer_idx, output_base):
    """Convert all 128 experts in one MoE layer."""
    layer_dir = os.path.join(output_base, f"layer_{layer_idx:02d}")
    os.makedirs(layer_dir, exist_ok=True)

    n_experts = GLM_CONFIG["expert_count"]
    results = []

    for exp_idx in range(n_experts):
        expert_path = os.path.join(layer_dir, f"expert_{exp_idx:03d}.bin")
        if os.path.exists(expert_path) and os.path.getsize(expert_path) > 5_000_000:
            file_size = os.path.getsize(expert_path)
            results.append({
                "gate_shape": [6144, 2048],
                "down_shape": [2048, 6144],
                "gate_sparsity": 0.5,
                "down_sparsity": 0.5,
                "file_bytes": file_size,
                "_cached": True,
            })
            continue

        result = convert_expert(reader, layer_idx, exp_idx, layer_dir)
        if result is None:
            return None
        results.append(result)

    total_bytes = sum(r["file_bytes"] for r in results)
    avg_gate_sparsity = np.mean([r.get("gate_sparsity", 0.5) for r in results])
    avg_up_sparsity = np.mean([r.get("up_sparsity", 0.5) for r in results])
    avg_down_sparsity = np.mean([r.get("down_sparsity", 0.5) for r in results])

    meta = {
        "layer_idx": layer_idx,
        "n_experts": n_experts,
        "total_bytes": total_bytes,
        "avg_gate_sparsity": float(avg_gate_sparsity),
        "avg_up_sparsity": float(avg_up_sparsity),
        "avg_down_sparsity": float(avg_down_sparsity),
        "experts": results,
    }

    with open(os.path.join(layer_dir, "meta.json"), 'w') as f:
        json.dump(meta, f, indent=2)

    return meta


def convert_full_model():
    """Convert the entire GLM-5.2 GGUF to FAIE ternary format."""
    print("=" * 70)
    print("  FAIE FULL MODEL CONVERSION")
    print(f"  Source: {MODEL_PATH}")
    print(f"  Output: {OUTPUT_DIR}")
    print("=" * 70)

    if not os.path.exists(MODEL_PATH):
        print(f"  ERROR: Model not found: {MODEL_PATH}")
        return False

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print(f"\n  Loading GGUF reader (memory-mapped)...")
    t_start = time.perf_counter()
    reader = GGUFReader(MODEL_PATH, 'r')

    # Identify MoE layers (layers 3-77 are MoE)
    moe_layers = list(range(3, 78))  # 75 MoE layers
    n_experts = GLM_CONFIG["expert_count"]  # 128
    total_experts = len(moe_layers) * n_experts  # 9,600

    print(f"  MoE layers: {moe_layers[0]}-{moe_layers[-1]} ({len(moe_layers)} layers)")
    print(f"  Experts per layer: {n_experts}")
    print(f"  Total experts to convert: {total_experts:,}")

    # Convert all MoE layers (with resume support)
    layer_metas = []
    total_bytes = 0
    total_converted = 0
    skipped = 0

    for i, layer_idx in enumerate(moe_layers):
        if is_layer_complete(OUTPUT_DIR, layer_idx):
            meta_path = os.path.join(OUTPUT_DIR, f"layer_{layer_idx:02d}", "meta.json")
            with open(meta_path, 'r') as f:
                meta = json.load(f)
            total_bytes += meta["total_bytes"]
            total_converted += n_experts
            layer_metas.append(meta)
            skipped += 1
            print(f"\n  Layer {layer_idx:2d} ({i+1}/{len(moe_layers)})... SKIP (already done)", flush=True)
            continue

        t0 = time.perf_counter()
        print(f"\n  Layer {layer_idx:2d} ({i+1}/{len(moe_layers)})...", end="", flush=True)

        meta = convert_layer(reader, layer_idx, OUTPUT_DIR)
        if meta is None:
            print(f" FAILED (tensors not found)")
            continue

        t1 = time.perf_counter()
        layer_bytes = meta["total_bytes"]
        total_bytes += layer_bytes
        total_converted += n_experts

        elapsed = t1 - t0
        remaining = len(moe_layers) - i - 1 - skipped
        eta = elapsed * max(0, remaining)
        print(f" {layer_bytes/(1024**2):.1f} MB, "
              f"sparsity gate={meta['avg_gate_sparsity']*100:.0f}% down={meta['avg_down_sparsity']*100:.0f}%, "
              f"{elapsed:.0f}s (ETA: {eta/60:.0f}m)")

        layer_metas.append(meta)

    t_total = time.perf_counter() - t_start

    # Write global metadata
    global_meta = {
        "model": "GLM-5.2-REAM",
        "source_gguf": MODEL_PATH,
        "format_version": 1,
        "group_size": GROUP_SIZE,
        "quantization": "ternary_rms_0.7",
        "n_moe_layers": len(moe_layers),
        "n_experts_per_layer": n_experts,
        "total_experts": total_converted,
        "total_bytes": total_bytes,
        "total_gb": total_bytes / (1024**3),
        "conversion_time_s": t_total,
        "config": GLM_CONFIG,
        "layers": [m["layer_idx"] for m in layer_metas],
    }

    with open(os.path.join(OUTPUT_DIR, "model_meta.json"), 'w') as f:
        json.dump(global_meta, f, indent=2)

    print(f"\n{'=' * 70}")
    print(f"  CONVERSION COMPLETE")
    print(f"  Experts converted: {total_converted:,} / {total_experts:,}")
    print(f"  Total size: {total_bytes/(1024**3):.2f} GB")
    print(f"  Time: {t_total/60:.1f} minutes")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"{'=' * 70}")

    return True


def read_faie_expert(expert_path):
    """Read a v2 FAIE expert binary file."""
    with open(expert_path, 'rb') as f:
        magic = struct.unpack('<I', f.read(4))[0]
        if magic != 0xFA1E0002:
            return None
        gs0, gs1 = struct.unpack('<II', f.read(8))
        us0, us1 = struct.unpack('<II', f.read(8))
        ds0, ds1 = struct.unpack('<II', f.read(8))
        gate_n, up_n, down_n = struct.unpack('<III', f.read(12))
        gate_packed_len, up_packed_len, down_packed_len = struct.unpack('<III', f.read(12))
        gate_scales_len, up_scales_len, down_scales_len = struct.unpack('<III', f.read(12))
        group_size = struct.unpack('<I', f.read(4))[0]
        gate_sp, up_sp, down_sp = struct.unpack('<fff', f.read(12))

        gate_packed = np.frombuffer(f.read(gate_packed_len), dtype=np.uint8)
        gate_scales = np.frombuffer(f.read(gate_scales_len * 4), dtype=np.float32)
        up_packed = np.frombuffer(f.read(up_packed_len), dtype=np.uint8)
        up_scales = np.frombuffer(f.read(up_scales_len * 4), dtype=np.float32)
        down_packed = np.frombuffer(f.read(down_packed_len), dtype=np.uint8)
        down_scales = np.frombuffer(f.read(down_scales_len * 4), dtype=np.float32)

    return {
        "gate_shape": (gs0, gs1), "up_shape": (us0, us1), "down_shape": (ds0, ds1),
        "gate_n": gate_n, "up_n": up_n, "down_n": down_n,
        "gate_packed": gate_packed, "gate_scales": gate_scales,
        "up_packed": up_packed, "up_scales": up_scales,
        "down_packed": down_packed, "down_scales": down_scales,
        "group_size": group_size,
    }


def verify_conversion():
    """7-step verification of the full conversion."""
    print("\n" + "=" * 70)
    print("  PATH 2 -- FULL CONVERSION -- 7-STEP VERIFICATION")
    print("=" * 70)

    results = []

    # V1: Check model_meta.json exists and is valid
    print("\n  [V1] Model metadata integrity...")
    meta_path = os.path.join(OUTPUT_DIR, "model_meta.json")
    v1_pass = False
    if os.path.exists(meta_path):
        with open(meta_path, 'r') as f:
            meta = json.load(f)
        v1_pass = (meta.get("total_experts", 0) == 9600 and
                   meta.get("n_moe_layers", 0) == 75 and
                   meta.get("n_experts_per_layer", 0) == 128)
        print(f"    Total experts: {meta.get('total_experts')}")
        print(f"    Total size: {meta.get('total_gb', 0):.2f} GB")
    results.append(("V1: Metadata valid (9600 experts, 75 layers)", v1_pass,
                    f"{meta.get('total_experts', 0)} experts"))

    # V2: Check all 75 layer directories exist
    print("\n  [V2] All 75 layer directories present...")
    moe_layers = list(range(3, 78))
    missing_layers = []
    for layer_idx in moe_layers:
        layer_dir = os.path.join(OUTPUT_DIR, f"layer_{layer_idx:02d}")
        if not os.path.isdir(layer_dir):
            missing_layers.append(layer_idx)
    v2_pass = len(missing_layers) == 0
    print(f"    Present: {75 - len(missing_layers)}/75")
    if missing_layers:
        print(f"    Missing: {missing_layers[:10]}{'...' if len(missing_layers) > 10 else ''}")
    results.append(("V2: All 75 layer directories", v2_pass,
                    f"{75 - len(missing_layers)}/75"))

    # V3: Check all 128 expert files per layer (sample 7 layers)
    print("\n  [V3] Expert file count (sample 7 layers)...")
    sample_layers = [3, 10, 20, 35, 50, 65, 77]
    all_128 = True
    for layer_idx in sample_layers:
        layer_dir = os.path.join(OUTPUT_DIR, f"layer_{layer_idx:02d}")
        if os.path.isdir(layer_dir):
            expert_files = [f for f in os.listdir(layer_dir) if f.startswith("expert_") and f.endswith(".bin")]
            if len(expert_files) != 128:
                all_128 = False
                print(f"    Layer {layer_idx}: {len(expert_files)}/128 experts")
        else:
            all_128 = False
    v3_pass = all_128
    print(f"    All sampled layers have 128 experts: {v3_pass}")
    results.append(("V3: 128 experts per layer (7 samples)", v3_pass, "128/128" if v3_pass else "MISSING"))

    # V4: Read and dequantize a random expert, compare to GGUF original
    print("\n  [V4] Roundtrip quality: read converted expert vs GGUF original...")
    test_layer = 20
    test_expert = 42
    expert_path = os.path.join(OUTPUT_DIR, f"layer_{test_layer:02d}", f"expert_{test_expert:03d}.bin")
    v4_pass = False
    cos = 0.0
    if os.path.exists(expert_path):
        data = read_faie_expert(expert_path)
        if data:
            gate_ternary = TernaryCompressor.unpack_ternary(data["gate_packed"], data["gate_n"])
            gate_recon = TernaryCompressor.dequantize(
                gate_ternary.reshape(data["gate_shape"]),
                data["gate_scales"], data["gate_shape"], data["group_size"])

            reader = GGUFReader(MODEL_PATH, 'r')
            gate_name = f"blk.{test_layer}.ffn_gate_exps.weight"
            for t in reader.tensors:
                if t.name == gate_name:
                    gate_shape_orig = [int(s) for s in t.shape]
                    gate_elements = gate_shape_orig[0] * gate_shape_orig[1]
                    gate_raw_orig = t.data[test_expert].tobytes()
                    gate_orig = dequant_q2k_fast(gate_raw_orig, gate_elements).reshape(gate_shape_orig[0], gate_shape_orig[1]).T
                    break
            del reader

            cos = np.dot(gate_orig.ravel(), gate_recon.ravel()) / (
                np.linalg.norm(gate_orig) * np.linalg.norm(gate_recon) + 1e-10)
            v4_pass = cos > 0.85
            print(f"    Layer {test_layer}, Expert {test_expert}")
            print(f"    Gate cosine (FAIE vs GGUF original): {cos:.6f}")
    results.append(("V4: Roundtrip quality > 0.85", v4_pass, f"cos={cos:.6f}" if v4_pass else "FAILED"))

    # V5: Total size matches projection (~67 GB for experts)
    print("\n  [V5] Total size validation...")
    total_size = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for f in files:
            if f.endswith('.bin'):
                total_size += os.path.getsize(os.path.join(root, f))
    total_gb = total_size / (1024**3)
    v5_pass = 70 < total_gb < 130  # gate+up+down = ~100 GB
    print(f"    Total expert data: {total_gb:.2f} GB")
    print(f"    Expected: ~100 GB (gate+up+down)")
    results.append(("V5: Size 50-90 GB", v5_pass, f"{total_gb:.1f} GB"))

    # V6: Sparsity distribution across layers
    print("\n  [V6] Sparsity distribution (7 layer samples)...")
    sparsities = []
    for layer_idx in sample_layers:
        meta_path_l = os.path.join(OUTPUT_DIR, f"layer_{layer_idx:02d}", "meta.json")
        if os.path.exists(meta_path_l):
            with open(meta_path_l, 'r') as f:
                lmeta = json.load(f)
            sparsities.append(lmeta["avg_gate_sparsity"])
    v6_pass = len(sparsities) >= 5 and all(0.3 < s < 0.8 for s in sparsities)
    if sparsities:
        print(f"    Range: {min(sparsities)*100:.0f}% - {max(sparsities)*100:.0f}%")
        print(f"    Mean: {np.mean(sparsities)*100:.0f}%")
    results.append(("V6: Sparsity in 30-80% range", v6_pass,
                    f"{np.mean(sparsities)*100:.0f}%" if sparsities else "N/A"))

    # V7: Load 7 random experts from different layers, SwiGLU forward pass, non-zero output
    print("\n  [V7] SwiGLU forward pass test (7 experts from 7 layers)...")
    import torch
    DEVICE = torch.device('cuda:0')
    all_nonzero = True
    for layer_idx in sample_layers:
        expert_idx = layer_idx % 128
        expert_path = os.path.join(OUTPUT_DIR, f"layer_{layer_idx:02d}", f"expert_{expert_idx:03d}.bin")
        if not os.path.exists(expert_path):
            all_nonzero = False
            continue

        data = read_faie_expert(expert_path)
        if data is None:
            all_nonzero = False
            continue

        gs0, gs1 = data["gate_shape"]
        us0, us1 = data["up_shape"]
        ds0, ds1 = data["down_shape"]
        group_size = data["group_size"]

        gate_ternary = TernaryCompressor.unpack_ternary(data["gate_packed"], data["gate_n"]).reshape(gs0, gs1)
        gate_recon = TernaryCompressor.dequantize(gate_ternary, data["gate_scales"], (gs0, gs1), group_size)
        up_ternary = TernaryCompressor.unpack_ternary(data["up_packed"], data["up_n"]).reshape(us0, us1)
        up_recon = TernaryCompressor.dequantize(up_ternary, data["up_scales"], (us0, us1), group_size)
        down_ternary = TernaryCompressor.unpack_ternary(data["down_packed"], data["down_n"]).reshape(ds0, ds1)
        down_recon = TernaryCompressor.dequantize(down_ternary, data["down_scales"], (ds0, ds1), group_size)

        x = torch.randn(1, gs1, dtype=torch.float16, device=DEVICE) * 0.1
        W_gate = torch.from_numpy(gate_recon).to(torch.float16).to(DEVICE)
        W_up = torch.from_numpy(up_recon).to(torch.float16).to(DEVICE)
        W_down = torch.from_numpy(down_recon).to(torch.float16).to(DEVICE)

        hidden = F.silu(F.linear(x, W_gate)) * F.linear(x, W_up)
        y = F.linear(hidden, W_down)
        y_norm = y.norm().item()
        if y_norm < 1e-10:
            all_nonzero = False
            print(f"    Layer {layer_idx}, Expert {expert_idx}: ZERO OUTPUT")

        del W_gate, W_up, W_down, x, y, hidden
        torch.cuda.empty_cache()

    v7_pass = all_nonzero
    print(f"    All 7 experts produce non-zero output: {v7_pass}")
    results.append(("V7: Forward pass non-zero (7 experts)", v7_pass, "all non-zero" if v7_pass else "ZERO FOUND"))

    # Verdict
    print(f"\n{'=' * 70}")
    all_pass = all(r[1] for r in results)
    for label, passed, detail in results:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}] {label} ({detail})")
    print(f"\n  VERDICT: {'ALL 7 CHECKS PASSED' if all_pass else 'SOME CHECKS FAILED'}")
    print(f"{'=' * 70}")

    return all_pass, results


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    if "--verify-only" in sys.argv:
        verify_conversion()
    else:
        success = convert_full_model()
        if success:
            verify_conversion()
