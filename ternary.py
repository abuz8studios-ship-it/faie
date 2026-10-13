"""
Heresy 3 — Ternary Quantization of Quranic Root Embeddings.

BitNet proved every weight can be {-1, 0, +1}. 1.58 bits per parameter.
We prove Arabic root semantics SURVIVE ternary compression.

The ternary-Arabic convergence:
  - Ternary weights: inhibit (-1), gate (0), excite (+1)
  - Arabic roots: 3 consonants per semantic nucleus
  - Real neurons: inhibitory, silent, excitatory

This module quantizes 10D Poincaré embeddings to ternary and measures
how much semantic structure is preserved.
"""

import torch
import torch.nn.functional as F
import numpy as np
import json


def load_embeddings(model_path="E:/ABU/NEOCORTEX/models/quranic_poincare_10d.pt"):
    checkpoint = torch.load(model_path, weights_only=False)
    embeddings = checkpoint["state_dict"]["embeddings"].detach()
    nodes = checkpoint["nodes"]
    node_to_idx = checkpoint["node_to_idx"]
    node_depths = checkpoint["node_depths"]
    return embeddings, nodes, node_to_idx, node_depths


def load_graph(graph_path="E:/ABU/NEOCORTEX/data/full_graph.json"):
    with open(graph_path, "r", encoding="utf-8") as f:
        return json.load(f)


def ternary_quantize(embeddings, threshold="mean"):
    """
    Quantize floating-point embeddings to {-1, 0, +1}.

    Strategy: per-dimension, values above +threshold → +1,
    below -threshold → -1, between → 0.
    """
    if threshold == "mean":
        scale = embeddings.abs().mean(dim=0)
    elif threshold == "median":
        scale = embeddings.abs().median(dim=0).values
    else:
        scale = torch.tensor(threshold)

    alpha = 0.7
    ternary = torch.zeros_like(embeddings)
    ternary[embeddings > alpha * scale] = 1.0
    ternary[embeddings < -alpha * scale] = -1.0

    return ternary, scale


def adaptive_ternary_quantize(embeddings, n_bits=2):
    """
    Adaptive quantization: use the sign and top-k dimensions.
    For each embedding, keep the k most important dimensions at +/-1,
    zero out the rest. k = n_bits determines sparsity.
    """
    abs_emb = embeddings.abs()
    _, top_indices = abs_emb.topk(n_bits, dim=-1)

    ternary = torch.zeros_like(embeddings)
    signs = embeddings.sign()
    ternary.scatter_(1, top_indices, signs.gather(1, top_indices))

    return ternary


def measure_preservation(original, quantized, nodes, node_to_idx, graph_data, k=10):
    """
    Measure how much semantic structure is preserved after quantization.

    Metrics:
    1. Nearest-neighbor overlap (Jaccard) for top-k
    2. Rank correlation (Spearman) of pairwise distances
    3. Theological co-occurrence preservation
    """
    roots = [n for n in nodes if n in graph_data["nodes"]
             and graph_data["nodes"][n].get("type") == "root"
             and n in node_to_idx]
    root_indices = [node_to_idx[r] for r in roots]

    orig_root_emb = original[root_indices]
    quant_root_emb = quantized[root_indices]

    orig_dists = torch.cdist(orig_root_emb, orig_root_emb)
    quant_dists = torch.cdist(quant_root_emb, quant_root_emb)

    nn_overlaps = []
    for i in range(len(roots)):
        orig_nn = set(orig_dists[i].topk(k + 1, largest=False).indices[1:].tolist())
        quant_nn = set(quant_dists[i].topk(k + 1, largest=False).indices[1:].tolist())
        overlap = len(orig_nn & quant_nn) / len(orig_nn | quant_nn)
        nn_overlaps.append(overlap)

    mean_overlap = np.mean(nn_overlaps)

    orig_flat = orig_dists[torch.triu(torch.ones(len(roots), len(roots)), diagonal=1).bool()]
    quant_flat = quant_dists[torch.triu(torch.ones(len(roots), len(roots)), diagonal=1).bool()]

    orig_ranks = orig_flat.argsort().argsort().float()
    quant_ranks = quant_flat.argsort().argsort().float()
    rank_corr = torch.corrcoef(torch.stack([orig_ranks, quant_ranks]))[0, 1].item()

    theological_pairs = [
        ("أله", "قول"), ("أله", "علم"), ("أله", "أمن"),
        ("أرض", "سمو"), ("أمن", "كفر"), ("أمن", "عمل"),
    ]
    pair_preserved = 0
    pair_total = 0
    for r1, r2 in theological_pairs:
        if r1 not in node_to_idx or r2 not in node_to_idx:
            continue
        if r1 not in roots or r2 not in roots:
            continue
        i1 = roots.index(r1)
        i2 = roots.index(r2)
        orig_rank = (orig_dists[i1] < orig_dists[i1, i2]).sum().item()
        quant_rank = (quant_dists[i1] < quant_dists[i1, i2]).sum().item()
        pair_total += 1
        if abs(orig_rank - quant_rank) < len(roots) * 0.1:
            pair_preserved += 1

    return {
        "nn_overlap_jaccard": mean_overlap,
        "rank_correlation": rank_corr,
        "theological_pairs_preserved": f"{pair_preserved}/{pair_total}",
        "n_roots_tested": len(roots),
    }


def compute_compression_stats(original, quantized):
    """Compute compression statistics."""
    orig_bits = original.numel() * 32
    ternary_bits = original.numel() * 1.58

    nonzero = (quantized != 0).sum().item()
    sparsity = 1.0 - nonzero / quantized.numel()

    effective_bits = nonzero * 1

    return {
        "original_bits": orig_bits,
        "ternary_bits": ternary_bits,
        "effective_bits_sparse": effective_bits,
        "compression_ratio_raw": orig_bits / ternary_bits,
        "compression_ratio_sparse": orig_bits / max(effective_bits, 1),
        "sparsity": sparsity,
        "nonzero_fraction": 1.0 - sparsity,
    }


if __name__ == "__main__":
    print("Heresy 3 — Ternary Quantization")
    print("=" * 60)

    print("\nLoading embeddings + graph...")
    embeddings, nodes, node_to_idx, node_depths = load_embeddings()
    graph_data = load_graph()
    print(f"  {len(nodes)} nodes, {embeddings.shape[1]}D embeddings")

    print("\n--- Method 1: Mean-threshold ternary ---")
    ternary_mean, scale_mean = ternary_quantize(embeddings, threshold="mean")
    stats_mean = compute_compression_stats(embeddings, ternary_mean)
    pres_mean = measure_preservation(embeddings, ternary_mean, nodes, node_to_idx, graph_data)
    print(f"  Compression: {stats_mean['compression_ratio_raw']:.1f}x "
          f"(32-bit → 1.58-bit)")
    print(f"  Sparsity: {stats_mean['sparsity']:.1%} zeros")
    print(f"  NN overlap (Jaccard@10): {pres_mean['nn_overlap_jaccard']:.4f}")
    print(f"  Rank correlation: {pres_mean['rank_correlation']:.4f}")
    print(f"  Theological pairs: {pres_mean['theological_pairs_preserved']}")

    print("\n--- Method 2: Adaptive top-k ternary ---")
    for k in [3, 5, 7]:
        ternary_adaptive = adaptive_ternary_quantize(embeddings, n_bits=k)
        stats_adap = compute_compression_stats(embeddings, ternary_adaptive)
        pres_adap = measure_preservation(embeddings, ternary_adaptive, nodes, node_to_idx, graph_data)
        print(f"  Top-{k}: overlap={pres_adap['nn_overlap_jaccard']:.4f} "
              f"rank_corr={pres_adap['rank_correlation']:.4f} "
              f"theo_pairs={pres_adap['theological_pairs_preserved']} "
              f"sparsity={stats_adap['sparsity']:.1%}")

    print("\n--- VRAM savings projection ---")
    for model_params in [(2e9, "2B"), (7e9, "7B"), (35e9, "35B")]:
        n, name = model_params
        fp32_gb = n * 4 / 1e9
        fp16_gb = n * 2 / 1e9
        int4_gb = n * 0.5 / 1e9
        ternary_gb = n * 1.58 / 8 / 1e9
        print(f"  {name} model: FP32={fp32_gb:.1f}GB → FP16={fp16_gb:.1f}GB "
              f"→ INT4={int4_gb:.1f}GB → Ternary={ternary_gb:.1f}GB")

    torch.save({
        "ternary_embeddings": ternary_mean,
        "scale": scale_mean,
        "method": "mean_threshold",
        "stats": stats_mean,
        "preservation": pres_mean,
    }, "E:/ABU/NEOCORTEX/models/ternary_embeddings.pt")

    print("\n  Ternary embeddings saved.")
    print("  Heresy 3 COMPLETE.")
