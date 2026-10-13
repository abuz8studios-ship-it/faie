"""
Heresy 4 — The Wave: Frequency-Domain Representation of Root Semantics.

Life is waves. The brain doesn't think in tokens — it thinks in oscillations.
FNet replaced ALL attention with DFT and got 92% of BERT at 7x speed.
FourierQK discovered language has natural frequency structure.

This module:
1. FFT the Poincaré embeddings to get frequency signatures per root
2. Show that semantically similar roots share spectral peaks
3. Build a harmonic field where concepts RESONATE
4. Demonstrate interference: similar ideas amplify, contradictions cancel
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
from collections import defaultdict


def load_data():
    checkpoint = torch.load("E:/ABU/NEOCORTEX/models/quranic_poincare_10d.pt",
                            weights_only=False)
    embeddings = checkpoint["state_dict"]["embeddings"].detach()
    nodes = checkpoint["nodes"]
    node_to_idx = checkpoint["node_to_idx"]
    node_depths = checkpoint["node_depths"]

    with open("E:/ABU/NEOCORTEX/data/full_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)

    return embeddings, nodes, node_to_idx, node_depths, graph


def spectral_embedding(embeddings):
    """Transform embeddings into the frequency domain using DFT."""
    spectral = torch.fft.rfft(embeddings, dim=-1)
    magnitude = spectral.abs()
    phase = spectral.angle()
    return spectral, magnitude, phase


def spectral_similarity(mag1, mag2):
    """Cosine similarity in the frequency domain."""
    return F.cosine_similarity(mag1.unsqueeze(0), mag2.unsqueeze(0)).item()


def harmonic_field(embeddings, nodes, node_to_idx, graph, n_freqs=6):
    """
    Build a harmonic field: each root is a standing wave.
    Nearby roots resonate (constructive interference).
    Distant roots cancel (destructive interference).
    """
    spectral, magnitude, phase = spectral_embedding(embeddings)

    roots = [n for n in nodes if n in graph["nodes"]
             and graph["nodes"][n].get("type") == "root"
             and n in node_to_idx]

    root_spectra = {}
    for root in roots:
        idx = node_to_idx[root]
        root_spectra[root] = {
            "magnitude": magnitude[idx].numpy(),
            "phase": phase[idx].numpy(),
            "dominant_freq": int(magnitude[idx].argmax().item()),
            "energy": float(magnitude[idx].pow(2).sum().item()),
        }

    return root_spectra


def interference_test(embeddings, nodes, node_to_idx, graph):
    """
    Test: do semantically related roots show constructive interference?
    Do contradictory roots show destructive interference?
    """
    spectral, magnitude, phase = spectral_embedding(embeddings)

    constructive_pairs = [
        ("أله", "ربب", "Allah-Lord: same theological axis"),
        ("أمن", "عمل", "Faith-Deeds: Quranic pairing"),
        ("أرض", "سمو", "Earth-Heaven: cosmic pair"),
        ("علم", "حكم", "Knowledge-Wisdom: epistemic pair"),
        ("رحم", "غفر", "Mercy-Forgiveness: divine attributes"),
    ]

    destructive_pairs = [
        ("أمن", "كفر", "Faith-Disbelief: opposition"),
        ("هدي", "ضلل", "Guidance-Misguidance: opposition"),
        ("نور", "ظلم", "Light-Darkness: opposition"),
        ("حيي", "موت", "Life-Death: opposition"),
    ]

    results = {"constructive": [], "destructive": []}

    print("\n  Constructive interference (semantic allies):")
    for r1, r2, desc in constructive_pairs:
        if r1 not in node_to_idx or r2 not in node_to_idx:
            continue
        idx1, idx2 = node_to_idx[r1], node_to_idx[r2]
        combined = spectral[idx1] + spectral[idx2]
        combined_energy = combined.abs().pow(2).sum().item()
        individual_energy = (magnitude[idx1].pow(2).sum() + magnitude[idx2].pow(2).sum()).item()
        ratio = combined_energy / max(individual_energy, 1e-8)

        results["constructive"].append(ratio)
        marker = "+" if ratio > 1.0 else "-"
        print(f"    [{marker}] {r1} + {r2}: energy_ratio={ratio:.4f} ({desc})")

    print("\n  Destructive interference (semantic opposites):")
    for r1, r2, desc in destructive_pairs:
        if r1 not in node_to_idx or r2 not in node_to_idx:
            continue
        idx1, idx2 = node_to_idx[r1], node_to_idx[r2]
        combined = spectral[idx1] + spectral[idx2]
        combined_energy = combined.abs().pow(2).sum().item()
        individual_energy = (magnitude[idx1].pow(2).sum() + magnitude[idx2].pow(2).sum()).item()
        ratio = combined_energy / max(individual_energy, 1e-8)

        results["destructive"].append(ratio)
        marker = "-" if ratio < 1.0 else "+"
        print(f"    [{marker}] {r1} + {r2}: energy_ratio={ratio:.4f} ({desc})")

    return results


def spectral_clustering(embeddings, nodes, node_to_idx, graph, n_clusters=8):
    """Cluster roots by their spectral signatures."""
    _, magnitude, _ = spectral_embedding(embeddings)

    roots = [n for n in nodes if n in graph["nodes"]
             and graph["nodes"][n].get("type") == "root"
             and n in node_to_idx
             and graph["nodes"][n].get("frequency", 0) > 50]

    root_mags = torch.stack([magnitude[node_to_idx[r]] for r in roots])

    root_mags_norm = F.normalize(root_mags, dim=-1)
    sim_matrix = root_mags_norm @ root_mags_norm.T

    clusters = defaultdict(list)
    dominant_freqs = root_mags.argmax(dim=-1)
    for i, root in enumerate(roots):
        freq = dominant_freqs[i].item()
        clusters[freq].append((root, (graph["nodes"][root].get("summary") or "")[:50]))

    return dict(clusters)


def fourier_attention(query_emb, key_embs, embeddings):
    """
    FNet-style attention: replace dot-product with DFT.
    Query in frequency domain attends to keys in frequency domain.
    """
    q_spectral = torch.fft.rfft(query_emb)
    k_spectral = torch.fft.rfft(key_embs, dim=-1)

    attention = (q_spectral.unsqueeze(0).conj() * k_spectral).real.sum(dim=-1)
    attention = F.softmax(attention, dim=0)

    return attention


if __name__ == "__main__":
    print("Heresy 4 — The Wave: Frequency-Domain Semantics")
    print("=" * 60)

    print("\nLoading data...")
    embeddings, nodes, node_to_idx, node_depths, graph = load_data()
    print(f"  {len(nodes)} nodes, {embeddings.shape[1]}D embeddings")

    print("\n--- Spectral analysis of root embeddings ---")
    spectral, magnitude, phase = spectral_embedding(embeddings)
    print(f"  FFT output: {spectral.shape} complex coefficients")
    print(f"  Frequency bins: {magnitude.shape[1]} per embedding")

    print("\n--- Harmonic field ---")
    field = harmonic_field(embeddings, nodes, node_to_idx, graph)
    top_energy = sorted(field.items(), key=lambda x: -x[1]["energy"])[:15]
    print("  Top 15 roots by spectral energy:")
    for root, data in top_energy:
        summary = (graph["nodes"].get(root, {}).get("summary") or "")[:40]
        print(f"    {root:8s}  energy={data['energy']:.6f}  "
              f"dominant_freq={data['dominant_freq']}  {summary}")

    print("\n--- Interference test ---")
    interference = interference_test(embeddings, nodes, node_to_idx, graph)

    if interference["constructive"]:
        avg_constructive = np.mean(interference["constructive"])
        print(f"\n  Avg constructive ratio: {avg_constructive:.4f} (>1.0 = amplification)")
    if interference["destructive"]:
        avg_destructive = np.mean(interference["destructive"])
        print(f"  Avg destructive ratio:  {avg_destructive:.4f} (<1.0 = cancellation)")

    print("\n--- Spectral clustering ---")
    clusters = spectral_clustering(embeddings, nodes, node_to_idx, graph)
    for freq, members in sorted(clusters.items()):
        if len(members) >= 3:
            names = ", ".join(m[0] for m in members[:8])
            print(f"  Freq bin {freq}: {len(members)} roots — {names}")

    print("\n--- Fourier attention test ---")
    test_root = "رحم"
    if test_root in node_to_idx:
        roots = [n for n in nodes if n in graph["nodes"]
                 and graph["nodes"][n].get("type") == "root"
                 and n in node_to_idx
                 and graph["nodes"][n].get("frequency", 0) > 20]
        root_indices = [node_to_idx[r] for r in roots]
        root_embs = embeddings[root_indices]

        query_emb = embeddings[node_to_idx[test_root]]
        attn = fourier_attention(query_emb, root_embs, embeddings)

        top_attn = attn.topk(10)
        print(f"  Fourier attention from '{test_root}':")
        for i, (score, idx) in enumerate(zip(top_attn.values, top_attn.indices)):
            root_name = roots[idx.item()]
            summary = (graph["nodes"].get(root_name, {}).get("summary") or "")[:40]
            print(f"    {score.item():.4f}  {root_name:8s}  {summary}")

    torch.save({
        "spectral_embeddings": spectral,
        "magnitude": magnitude,
        "phase": phase,
        "harmonic_field": {k: {kk: vv.tolist() if hasattr(vv, 'tolist') else vv
                               for kk, vv in v.items()}
                          for k, v in field.items()},
    }, "E:/ABU/NEOCORTEX/models/wave_data.pt")

    print("\n  Wave data saved.")
    print("  Heresy 4 COMPLETE.")
