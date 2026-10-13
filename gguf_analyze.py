"""
GGUF Model Analyzer — Blueprint for the Fused Arabic Inference Engine.

Uses the gguf library to properly parse the model structure.
Maps: architecture, layer count, MoE expert layout, tensor shapes, quantization.
"""

import os
import sys
import re
from pathlib import Path
from collections import defaultdict
from gguf import GGUFReader

GGML_BITS = {
    "F32": 32, "F16": 16, "BF16": 16,
    "Q2_K": 2.5625, "Q3_K": 3.4375, "Q4_K": 4.5, "Q5_K": 5.5, "Q6_K": 6.5625,
    "Q4_0": 4.5, "Q4_1": 5, "Q5_0": 5.5, "Q5_1": 6, "Q8_0": 8.5, "Q8_1": 9,
    "IQ1_S": 1.5625, "IQ1_M": 1.75, "IQ2_XXS": 2.0625, "IQ2_XS": 2.3125,
    "IQ2_S": 2.5, "IQ3_XXS": 3.0625, "IQ3_S": 3.4375, "IQ4_NL": 4.5, "IQ4_XS": 4.25,
    "TQ1_0": 1.6875, "TQ2_0": 2.0625,
    "I8": 8, "I16": 16, "I32": 32, "I64": 64, "F64": 64,
}


def analyze_gguf(model_path):
    file_size = os.path.getsize(model_path)
    print(f"\n  File: {Path(model_path).name}")
    print(f"  Size: {file_size / (1024**3):.2f} GB")

    reader = GGUFReader(model_path, 'r')

    print(f"\n  --- Model Metadata ---")
    arch_fields = {}
    for field in reader.fields.values():
        name = field.name
        if any(k in name.lower() for k in [
            'arch', 'model', 'block', 'layer', 'head', 'embed', 'vocab',
            'expert', 'context', 'rope', 'attention', 'feed', 'norm',
            'hidden', 'intermediate', 'general', 'tensor', 'quant'
        ]):
            try:
                if field.types and len(field.types) > 0:
                    data = field.parts[-1]
                    if hasattr(data, 'tolist'):
                        val = data.tolist()
                        if isinstance(val, list) and len(val) == 1:
                            val = val[0]
                        elif isinstance(val, list) and len(val) > 5:
                            val = f"[{len(val)} items]"
                    elif isinstance(data, bytes):
                        val = data.decode('utf-8', errors='replace')
                    else:
                        val = str(data)
                    arch_fields[name] = val
                    if isinstance(val, str) and len(str(val)) > 100:
                        val = str(val)[:100] + "..."
                    print(f"    {name}: {val}")
            except Exception:
                pass

    print(f"\n  --- Tensor Analysis ---")
    tensors = []
    for tensor in reader.tensors:
        name = tensor.name
        shape = list(tensor.shape)
        qtype = tensor.tensor_type.name if hasattr(tensor.tensor_type, 'name') else str(tensor.tensor_type)
        n_elements = tensor.n_elements
        bpw = GGML_BITS.get(qtype, 4.0)
        size_bytes = int(n_elements * bpw / 8)

        tensors.append({
            "name": name,
            "shape": shape,
            "n_elements": n_elements,
            "qtype": qtype,
            "bpw": bpw,
            "size_bytes": size_bytes,
        })

    total_params = sum(t["n_elements"] for t in tensors)
    total_bytes = sum(t["size_bytes"] for t in tensors)

    max_layer = 0
    expert_tensors = 0
    for t in tensors:
        m = re.search(r'(?:blk|block|layer)[._](\d+)', t["name"])
        if m:
            max_layer = max(max_layer, int(m.group(1)))
        if "expert" in t["name"].lower() or "ffn_gate_exp" in t["name"] or "ffn_up_exp" in t["name"]:
            expert_tensors += 1

    n_layers = max_layer + 1

    print(f"  Total tensors: {len(tensors)}")
    print(f"  Total parameters: {total_params:,} ({total_params/1e9:.2f}B)")
    print(f"  Total weight data: {total_bytes/(1024**3):.2f} GB")
    print(f"  Layers: {n_layers}")
    print(f"  Expert tensors: {expert_tensors}")

    quant_dist = defaultdict(lambda: {"count": 0, "bytes": 0, "params": 0})
    for t in tensors:
        quant_dist[t["qtype"]]["count"] += 1
        quant_dist[t["qtype"]]["bytes"] += t["size_bytes"]
        quant_dist[t["qtype"]]["params"] += t["n_elements"]

    print(f"\n  --- Quantization Distribution ---")
    for qname, stats in sorted(quant_dist.items(), key=lambda x: -x[1]["bytes"]):
        pct = stats["bytes"] / max(total_bytes, 1) * 100
        print(f"    {qname:12s}: {stats['count']:5d} tensors, "
              f"{stats['bytes']/(1024**3):8.3f} GB ({pct:5.1f}%), "
              f"{stats['params']/1e9:.3f}B params")

    categories = {
        "embedding": lambda n: "embed" in n.lower() or ("token" in n.lower() and "norm" not in n.lower()),
        "attention": lambda n: any(k in n.lower() for k in ["attn", "q_proj", "k_proj", "v_proj", "o_proj", "qkv"]),
        "ffn_moe": lambda n: any(k in n.lower() for k in ["ffn", "mlp", "gate_proj", "up_proj", "down_proj", "expert", "w1", "w2", "w3"]),
        "norm": lambda n: "norm" in n.lower(),
        "output": lambda n: "output" in n.lower() and "norm" not in n.lower(),
    }

    component_stats = {}
    categorized = set()
    for cat_name, matcher in categories.items():
        cat_tensors = [t for t in tensors if matcher(t["name"])]
        cat_bytes = sum(t["size_bytes"] for t in cat_tensors)
        cat_params = sum(t["n_elements"] for t in cat_tensors)
        component_stats[cat_name] = {"bytes": cat_bytes, "params": cat_params, "count": len(cat_tensors)}
        for t in cat_tensors:
            categorized.add(t["name"])

    uncategorized = [t for t in tensors if t["name"] not in categorized]
    if uncategorized:
        unc_bytes = sum(t["size_bytes"] for t in uncategorized)
        component_stats["other"] = {"bytes": unc_bytes, "params": sum(t["n_elements"] for t in uncategorized), "count": len(uncategorized)}

    print(f"\n  --- Component Breakdown ---")
    for cat_name, stats in sorted(component_stats.items(), key=lambda x: -x[1]["bytes"]):
        pct = stats["bytes"] / max(total_bytes, 1) * 100
        print(f"    {cat_name:15s}: {stats['bytes']/(1024**3):8.3f} GB ({pct:5.1f}%)  "
              f"{stats['params']/1e9:.3f}B params  [{stats['count']} tensors]")

    print(f"\n  --- Sample Tensors (first + last + largest) ---")
    for t in tensors[:5]:
        print(f"    {t['name']:60s} {str(t['shape']):20s} {t['qtype']:8s} {t['size_bytes']/(1024**2):8.1f} MB")
    print(f"    ...")
    for t in tensors[-3:]:
        print(f"    {t['name']:60s} {str(t['shape']):20s} {t['qtype']:8s} {t['size_bytes']/(1024**2):8.1f} MB")
    print(f"\n    Largest tensors:")
    for t in sorted(tensors, key=lambda x: -x["size_bytes"])[:10]:
        print(f"    {t['name']:60s} {str(t['shape']):20s} {t['qtype']:8s} {t['size_bytes']/(1024**2):8.1f} MB")

    n_experts_per_layer = arch_fields.get("glm-dsa.expert_count", 0)
    experts_used = arch_fields.get("glm-dsa.expert_used_count", 0)
    shared_experts = arch_fields.get("glm-dsa.expert_shared_count", 0)
    dense_layers = arch_fields.get("glm-dsa.leading_dense_block_count", 0)
    if isinstance(n_experts_per_layer, list):
        n_experts_per_layer = n_experts_per_layer[0] if n_experts_per_layer else 0
    if isinstance(experts_used, list):
        experts_used = experts_used[0] if experts_used else 0
    if isinstance(shared_experts, list):
        shared_experts = shared_experts[0] if shared_experts else 0
    if isinstance(dense_layers, list):
        dense_layers = dense_layers[0] if dense_layers else 0
    n_experts_per_layer = int(n_experts_per_layer)
    experts_used = int(experts_used)
    shared_experts = int(shared_experts)
    dense_layers = int(dense_layers)

    moe_layers = set()
    for t in tensors:
        if "_exps" in t["name"] or "expert" in t["name"].lower():
            m = re.search(r'blk\.(\d+)', t["name"])
            if m:
                moe_layers.add(int(m.group(1)))

    moe_bytes = sum(t["size_bytes"] for t in tensors if "_exps" in t["name"] or "shexp" in t["name"])
    print(f"\n  --- MoE Structure ---")
    print(f"    Experts per layer: {n_experts_per_layer}")
    print(f"    Experts used per token: {experts_used}")
    print(f"    Shared experts (always active): {shared_experts}")
    print(f"    Leading dense layers: {dense_layers}")
    print(f"    MoE layers: {len(moe_layers)} (of {n_layers} total)")
    if moe_layers:
        print(f"    MoE layer range: {min(moe_layers)}-{max(moe_layers)}")
    print(f"    Total MoE weight: {moe_bytes/(1024**3):.2f} GB ({moe_bytes/max(total_bytes,1)*100:.1f}%)")
    print(f"    Sparsity ratio: {experts_used}/{n_experts_per_layer} = {experts_used/max(n_experts_per_layer,1)*100:.1f}% active")

    print(f"\n{'=' * 70}")
    print(f"  FUSED ARABIC INFERENCE ENGINE — COMPRESSION MATH")
    print(f"{'=' * 70}")

    ffn_params = component_stats.get("ffn_moe", {}).get("params", 0)
    non_ffn_params = total_params - ffn_params
    ternary_bpw = 1.58
    gpu_bw = 1790
    pcie_bw = 64
    vram_total = 32
    kv_reserve = 2
    vram_budget = vram_total - kv_reserve

    current_bpw = total_bytes * 8 / total_params
    ternary_full = total_params * ternary_bpw / 8

    print(f"\n  Current state:")
    print(f"    Model size: {total_bytes/(1024**3):.2f} GB at {current_bpw:.2f} avg bpw")
    print(f"    Speed: ~2 tok/s (CPU-offloaded MoE)")

    print(f"\n  Step 1: Arabic ternary quantization ({ternary_bpw}-bit)")
    print(f"    Full model: {total_bytes/(1024**3):.2f} GB → {ternary_full/(1024**3):.2f} GB")
    print(f"    Compression: {current_bpw/ternary_bpw:.1f}x from current Q2_K/Q3_K mix")
    print(f"    Fits in 32 GB: {'YES' if ternary_full < vram_budget * 1e9 else 'NO'}")

    if n_experts_per_layer > 0 and experts_used > 0:
        n_moe = len(moe_layers) if moe_layers else (n_layers - dense_layers)
        active_total = experts_used + shared_experts
        active_ffn = int(ffn_params * active_total / n_experts_per_layer)
        runtime_params = non_ffn_params + active_ffn
        runtime_bytes_val = runtime_params * ternary_bpw / 8

        print(f"\n  Step 2: MoE sparse routing ({experts_used}+{shared_experts} of {n_experts_per_layer} experts)")
        print(f"    Active experts per layer: {active_total} ({experts_used} routed + {shared_experts} shared)")
        print(f"    Sparsity: {active_total/n_experts_per_layer*100:.1f}% activation")
        print(f"    Active FFN params: {ffn_params/1e9:.2f}B → {active_ffn/1e9:.2f}B per token")
        print(f"    Active total: {runtime_params/1e9:.2f}B per token")
        print(f"    Per-token read at ternary: {runtime_bytes_val/(1024**3):.2f} GB")

        emb_params = component_stats.get("embedding", {}).get("params", 0)
        attn_params = component_stats.get("attention", {}).get("params", 0)
        out_params = component_stats.get("output", {}).get("params", 0)
        norm_params = component_stats.get("norm", {}).get("params", 0)
        shared_exp_params = 0
        for t in tensors:
            if "shexp" in t["name"]:
                shared_exp_params += t["n_elements"]

        expert_dim = arch_fields.get("glm-dsa.expert_feed_forward_length", 2048)
        if isinstance(expert_dim, list):
            expert_dim = expert_dim[0]
        expert_dim = int(expert_dim)
        embed_dim = arch_fields.get("glm-dsa.embedding_length", 6144)
        if isinstance(embed_dim, list):
            embed_dim = embed_dim[0]
        embed_dim = int(embed_dim)

        gpu_permanent = (attn_params + emb_params + out_params + norm_params + shared_exp_params) * ternary_bpw / 8
        expert_pool = ternary_full - gpu_permanent
        gpu_for_cache = (vram_budget * 1e9) - gpu_permanent

        per_expert_bytes = 3 * expert_dim * embed_dim * ternary_bpw / 8
        experts_cacheable = int(gpu_for_cache / per_expert_bytes) if per_expert_bytes > 0 else 0
        total_experts = n_experts_per_layer * n_moe
        experts_per_token = active_total * n_moe

        print(f"\n  Step 3: FAIE GPU memory plan (RTX 5090, {vram_total} GB VRAM)")
        print(f"    Permanent on GPU:")
        print(f"      Attention:     {attn_params/1e9:.3f}B → {attn_params*ternary_bpw/8/(1024**3):.2f} GB")
        print(f"      Embedding:     {emb_params/1e9:.3f}B → {emb_params*ternary_bpw/8/(1024**3):.2f} GB")
        print(f"      Output head:   {out_params/1e9:.3f}B → {out_params*ternary_bpw/8/(1024**3):.2f} GB")
        print(f"      Shared experts:{shared_exp_params/1e9:.3f}B → {shared_exp_params*ternary_bpw/8/(1024**3):.2f} GB")
        print(f"      Norms:         {norm_params/1e9:.3f}B → {norm_params*ternary_bpw/8/(1024**3):.3f} GB")
        print(f"      Total permanent: {gpu_permanent/(1024**3):.2f} GB")
        print(f"    KV cache reserve: {kv_reserve} GB")
        print(f"    Expert cache budget: {gpu_for_cache/(1024**3):.2f} GB")
        print(f"    Per expert (ternary): {per_expert_bytes/(1024):.1f} KB")
        print(f"    Cacheable experts: {experts_cacheable:,} of {total_experts:,} total")
        print(f"    Experts needed per token: {experts_per_token:,}")

        print(f"\n  Step 4: Speed projections")
        if ternary_full < vram_budget * 1e9:
            tps = gpu_bw / (runtime_bytes_val / 1e9)
            print(f"    ALL in VRAM: {tps:.0f} tok/s (bandwidth-bound at {gpu_bw} GB/s)")
        else:
            print(f"    Full model ({ternary_full/(1024**3):.1f} GB) > VRAM budget ({vram_budget} GB)")

            transfer_per_token = experts_per_token * per_expert_bytes
            gpu_compute_time = runtime_bytes_val / (gpu_bw * 1e9)
            pcie_transfer_time = transfer_per_token / (pcie_bw * 1e9)

            print(f"\n    Strategy A: CPU→GPU expert paging (presynthesizer pre-routes)")
            print(f"      Expert data per token: {transfer_per_token/(1024**2):.1f} MB ({experts_per_token} experts)")
            print(f"      GPU compute: {gpu_compute_time*1000:.2f} ms")
            print(f"      PCIe transfer: {pcie_transfer_time*1000:.2f} ms (at {pcie_bw} GB/s)")
            total_time = max(gpu_compute_time, pcie_transfer_time)
            tps_paged = 1.0 / total_time
            print(f"      Pipelined tok/s: {tps_paged:.0f}")

            cached_pct = min(1.0, experts_cacheable / total_experts)
            miss_rate = 1.0 - cached_pct
            avg_transfer = transfer_per_token * miss_rate
            avg_time = max(gpu_compute_time, avg_transfer / (pcie_bw * 1e9))
            tps_cached = 1.0 / avg_time if avg_time > 0 else float('inf')
            print(f"\n    Strategy B: Hot expert cache ({cached_pct*100:.0f}% cached)")
            print(f"      Cache hit rate: {cached_pct*100:.0f}%")
            print(f"      Avg transfer per token: {avg_transfer/(1024**2):.1f} MB")
            print(f"      Projected tok/s: {tps_cached:.0f}")

            prune_targets = [64, 48, 32, 24, 16]
            print(f"\n    Strategy C: Expert pruning + ternary")
            for keep in prune_targets:
                pruned_ffn = int(ffn_params * keep / n_experts_per_layer)
                pruned_total = non_ffn_params + pruned_ffn
                pruned_bytes = pruned_total * ternary_bpw / 8
                active_pruned = min(experts_used + shared_experts, keep)
                runtime_pruned = non_ffn_params + int(pruned_ffn * active_pruned / keep)
                runtime_pruned_bytes = runtime_pruned * ternary_bpw / 8
                fits = pruned_bytes < vram_budget * 1e9
                if fits:
                    speed = gpu_bw / (runtime_pruned_bytes / 1e9)
                else:
                    overflow = pruned_bytes - vram_budget * 1e9
                    speed = 1.0 / (overflow / (pcie_bw * 1e9) + runtime_pruned_bytes / (gpu_bw * 1e9))
                marker = " ← FITS" if fits else ""
                print(f"      Keep {keep}/{n_experts_per_layer}: {pruned_bytes/(1024**3):.1f} GB, "
                      f"{runtime_pruned_bytes/(1024**3):.1f} GB active → {speed:.0f} tok/s{marker}")

        print(f"\n  {'=' * 66}")
        print(f"  VERDICT")
        print(f"  {'=' * 66}")
        print(f"  Architecture: {n_experts_per_layer} experts × {n_moe} MoE layers + {dense_layers} dense")
        print(f"  The 95.4% MoE structure is the compression leverage point.")
        print(f"  Only {active_total}/{n_experts_per_layer} experts ({active_total/n_experts_per_layer*100:.1f}%) "
              f"activate per token.")
        print(f"  Per-token active data at ternary: {runtime_bytes_val/(1024**3):.2f} GB")
        print(f"  GPU bandwidth headroom: {gpu_bw:.0f} GB/s ÷ {runtime_bytes_val/(1024**3):.2f} GB "
              f"= {gpu_bw / (runtime_bytes_val / 1e9):.0f} tok/s theoretical max")
    else:
        print(f"\n  No MoE detected — dense model")
        print(f"  Ternary size: {ternary_full/(1024**3):.2f} GB")
        if ternary_full < vram_budget * 1e9:
            print(f"  Projected tok/s: {gpu_bw / (ternary_full / 1e9):.0f}")

    return {
        "total_params": total_params,
        "total_bytes": total_bytes,
        "n_layers": n_layers,
        "n_experts": n_experts_per_layer,
        "experts_used": experts_used,
        "shared_experts": shared_experts,
        "dense_layers": dense_layers,
        "moe_layers": len(moe_layers),
        "component_stats": component_stats,
        "metadata": arch_fields,
        "tensors": tensors,
    }


if __name__ == "__main__":
    model_path = r"E:\ABU\MODELS\llm\glm-5.2-ream\GLM-5.2-REAM-Q2_K.gguf"
    if not os.path.exists(model_path):
        print(f"Model not found: {model_path}")
        sys.exit(1)

    print("=" * 70)
    print("  GGUF MODEL ANALYZER — Fused Arabic Inference Engine Blueprint")
    print("=" * 70)

    result = analyze_gguf(model_path)

    print(f"\n{'=' * 70}")
    print(f"  BLUEPRINT COMPLETE")
    print(f"{'=' * 70}")
