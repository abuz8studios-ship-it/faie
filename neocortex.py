"""
THE SYNTHETIC NEOCORTEX v2 — All 8 Heresies Fused Into One Brain.

This is the final module. It loads every heresy and wires them into
a single inference pipeline:

  INPUT (Arabic root query)
    → H1: Poincaré embedding lookup (curved space)
    → H3: Ternary quantization (1.58-bit compression)
    → H4: FFT spectral signature (frequency domain)
    → H7: KAN morphological prediction (spline transforms)
    → H2: Liquid routing (input-adaptive path selection)
    → H8: Fractal hive (3-level MoE processing)
    → H6: TTT adaptation (learn from this query)
    → H5: Unified memory (store + retrieve across modalities)
  OUTPUT: enriched semantic representation + full routing trace
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(__file__))

from ternary import ternary_quantize, compute_compression_stats
from wave import spectral_embedding, harmonic_field as build_harmonic
from kan import MorphKAN
from liquid import LiquidBrain
from hive import FractalHive
from ttt import TTTBrain
from memory import UnifiedMemory


class SyntheticNeocortex:
    """
    The fused brain. All 8 heresies in one pipeline.
    """
    def __init__(self, base_dir="E:/ABU/NEOCORTEX"):
        self.base_dir = base_dir
        print("  Loading Heresy 1: Poincaré embeddings (curved space)...")
        self._load_poincare()

        print("  Loading Heresy 3: Ternary quantization...")
        self._load_ternary()

        print("  Loading Heresy 4: Spectral signatures...")
        self._load_wave()

        print("  Loading Heresy 7: KAN morphological functions...")
        self._load_kan()

        print("  Loading Heresy 2: Liquid brain...")
        self._load_liquid()

        print("  Loading Heresy 8: Fractal hive...")
        self._load_hive()

        print("  Loading Heresy 6: TTT brain...")
        self._load_ttt()

        print("  Loading Heresy 5: Unified memory...")
        self._load_memory()

    def _load_poincare(self):
        path = f"{self.base_dir}/models/quranic_poincare_10d.pt"
        checkpoint = torch.load(path, weights_only=False)
        self.embeddings = checkpoint["state_dict"]["embeddings"].detach()
        self.nodes = checkpoint["nodes"]
        self.node_to_idx = checkpoint["node_to_idx"]
        self.node_depths = checkpoint["node_depths"]

        with open(f"{self.base_dir}/data/full_graph.json", "r", encoding="utf-8") as f:
            self.graph = json.load(f)

    def _load_ternary(self):
        path = f"{self.base_dir}/models/ternary_embeddings.pt"
        data = torch.load(path, weights_only=False)
        self.ternary_embeddings = data["ternary_embeddings"]
        self.ternary_scale = data["scale"]
        self.ternary_stats = data["stats"]

    def _load_wave(self):
        path = f"{self.base_dir}/models/wave_data.pt"
        data = torch.load(path, weights_only=False)
        self.spectral_embeddings = data["spectral_embeddings"]
        self.spectral_magnitude = data["magnitude"]
        self.spectral_phase = data["phase"]
        self.harmonic_field = data["harmonic_field"]

    def _load_kan(self):
        path = f"{self.base_dir}/models/morph_kan.pt"
        data = torch.load(path, weights_only=False)
        self.kan = MorphKAN(
            embed_dim=data["embed_dim"],
            n_patterns=data["n_patterns"],
            hidden=48, grid_size=8
        )
        self.kan.load_state_dict(data["state_dict"])
        self.kan.eval()
        self.kan_patterns = data["patterns"]
        self.kan_pattern_to_idx = data["pattern_to_idx"]

    def _load_liquid(self):
        path = f"{self.base_dir}/models/liquid_brain.pt"
        data = torch.load(path, weights_only=False)
        self.liquid = LiquidBrain(embed_dim=data["embed_dim"])
        self.liquid.load_state_dict(data["state_dict"])
        self.liquid.eval()

    def _load_hive(self):
        path = f"{self.base_dir}/models/fractal_hive.pt"
        data = torch.load(path, weights_only=False)
        self.hive = FractalHive(
            embed_dim=data["embed_dim"],
            n_domains=data["n_domains"]
        )
        self.hive.load_state_dict(data["state_dict"])
        self.hive.eval()
        self.hive.domain_names = data["domain_names"]

    def _load_ttt(self):
        path = f"{self.base_dir}/models/ttt_brain.pt"
        data = torch.load(path, weights_only=False)
        inner_lr = data.get("inner_lr", 0.01)
        self.ttt = TTTBrain(embed_dim=data["embed_dim"], inner_lr=inner_lr)
        self.ttt.load_state_dict(data["state_dict"])
        self.ttt.eval()

    def _load_memory(self):
        self.memory = UnifiedMemory(
            model_path=f"{self.base_dir}/models/quranic_poincare_10d.pt",
            graph_path=f"{self.base_dir}/data/full_graph.json",
            db_path=f"{self.base_dir}/models/unified_memory.db"
        )

    def query(self, root_name):
        """
        Full neocortex inference pipeline.
        Returns a rich result combining all 8 heresies.
        """
        if root_name not in self.node_to_idx:
            return {"error": f"Root '{root_name}' not found"}

        result = {"root": root_name, "heresies": {}}
        idx = self.node_to_idx[root_name]

        # H1: Poincaré embedding
        emb = self.embeddings[idx]
        norm = emb.norm().item()
        depth = self.node_depths.get(idx, -1)
        node_data = self.graph["nodes"].get(root_name, {})
        result["heresies"]["H1_poincare"] = {
            "embedding_norm": round(norm, 6),
            "depth": depth,
            "frequency": node_data.get("frequency", 0),
            "summary": (node_data.get("summary") or "")[:80],
        }

        # H3: Ternary representation
        ternary = self.ternary_embeddings[idx]
        nonzero = (ternary != 0).sum().item()
        result["heresies"]["H3_ternary"] = {
            "ternary_vector": ternary.tolist(),
            "nonzero_dims": nonzero,
            "compression": "20.3x (32-bit → 1.58-bit)",
        }

        # H4: Spectral signature
        spec = self.spectral_embeddings[idx]
        mag = self.spectral_magnitude[idx]
        result["heresies"]["H4_wave"] = {
            "dominant_frequency": int(mag.argmax().item()),
            "spectral_energy": round(mag.pow(2).sum().item(), 6),
            "magnitude_profile": [round(m, 4) for m in mag.tolist()],
        }

        # H7: KAN morphological predictions
        kan_preds = {}
        emb_input = emb.unsqueeze(0)
        for pattern in self.kan_patterns[:5]:
            if pattern in self.kan_pattern_to_idx:
                pat_idx = self.kan_pattern_to_idx[pattern]
                pat_oh = F.one_hot(
                    torch.tensor([pat_idx]), len(self.kan_patterns)).float()
                with torch.no_grad():
                    pred = self.kan(emb_input, pat_oh)
                kan_preds[pattern] = round(pred.norm().item(), 4)
        result["heresies"]["H7_kan"] = {
            "n_patterns": len(self.kan_patterns),
            "predicted_norms": kan_preds,
        }

        # H2: Liquid routing
        with torch.no_grad():
            _, routing = self.liquid(emb_input, return_routing=True)
        gate1 = routing["gate1"].squeeze().numpy()
        top_experts = np.argsort(-gate1)[:3]
        result["heresies"]["H2_liquid"] = {
            "active_experts": top_experts.tolist(),
            "expert_weights": [round(gate1[e], 4) for e in top_experts],
            "complexity": round(routing["complexity"].item(), 4),
        }

        # H8: Fractal hive routing
        with torch.no_grad():
            _, hive_routing = self.hive(emb_input, return_routing=True)
        domain_w = hive_routing["domain_weights"].squeeze().numpy()
        top_domain = int(np.argmax(domain_w))
        result["heresies"]["H8_hive"] = {
            "primary_domain": self.hive.domain_names[top_domain],
            "domain_weights": {
                name: round(w, 4)
                for name, w in zip(self.hive.domain_names, domain_w)
            },
        }

        # H6: TTT adaptation trace
        with torch.no_grad():
            pre_ttt = self.ttt(emb_input, update_inner=False)
        post_ttt = self.ttt(emb_input, update_inner=True)
        with torch.no_grad():
            delta = F.mse_loss(pre_ttt, post_ttt).item()
        result["heresies"]["H6_ttt"] = {
            "adaptation_delta": round(delta, 8),
            "adapted": delta > 1e-8,
        }

        # H5: Unified memory retrieval
        unified = self.memory.unified_query(root_name, k_vector=5, graph_depth=1)
        result["heresies"]["H5_memory"] = {
            "unified_results": [
                {"name": r["name"], "score": round(r["score"], 4),
                 "signals": [(s[0], round(s[1], 4)) for s in r["signals"]]}
                for r in unified[:5]
            ],
        }

        return result

    def compare_roots(self, root1, root2):
        """Compare two roots across all heresies."""
        r1 = self.query(root1)
        r2 = self.query(root2)
        if "error" in r1 or "error" in r2:
            return {"error": "One or both roots not found"}

        idx1 = self.node_to_idx[root1]
        idx2 = self.node_to_idx[root2]

        poincare_dist = (self.embeddings[idx1] - self.embeddings[idx2]).norm().item()

        spec1 = self.spectral_magnitude[idx1]
        spec2 = self.spectral_magnitude[idx2]
        spectral_sim = F.cosine_similarity(spec1.unsqueeze(0), spec2.unsqueeze(0)).item()

        t1 = self.ternary_embeddings[idx1]
        t2 = self.ternary_embeddings[idx2]
        ternary_match = (t1 == t2).float().mean().item()

        return {
            "roots": [root1, root2],
            "poincare_distance": round(poincare_dist, 6),
            "spectral_similarity": round(spectral_sim, 6),
            "ternary_overlap": round(ternary_match, 4),
            "r1_domain": r1["heresies"]["H8_hive"]["primary_domain"],
            "r2_domain": r2["heresies"]["H8_hive"]["primary_domain"],
            "same_domain": (r1["heresies"]["H8_hive"]["primary_domain"] ==
                          r2["heresies"]["H8_hive"]["primary_domain"]),
        }

    def stats(self):
        """Full system statistics."""
        mem_stats = self.memory.stats()
        total_params = sum([
            sum(p.numel() for p in self.kan.parameters()),
            sum(p.numel() for p in self.liquid.parameters()),
            sum(p.numel() for p in self.hive.parameters()),
            sum(p.numel() for p in self.ttt.parameters()),
        ])

        return {
            "nodes": len(self.nodes),
            "roots": mem_stats["roots"],
            "edges": mem_stats["edges"],
            "embedding_dim": self.embeddings.shape[1],
            "total_neural_params": total_params,
            "ternary_compression": self.ternary_stats["compression_ratio_raw"],
            "spectral_bins": self.spectral_magnitude.shape[1],
            "kan_patterns": len(self.kan_patterns),
            "liquid_experts": 8,
            "hive_domains": len(self.hive.domain_names),
            "hive_specialists": len(self.hive.domain_names) * 4,
            "memory_db_kb": mem_stats["db_size_kb"],
            "queries_served": mem_stats["queries_served"],
        }


def print_query_result(result):
    """Pretty-print a neocortex query result."""
    root = result["root"]
    h = result["heresies"]

    print(f"\n{'='*60}")
    print(f"  NEOCORTEX QUERY: {root}")
    print(f"{'='*60}")

    p = h["H1_poincare"]
    print(f"\n  H1 Poincaré: norm={p['embedding_norm']:.4f} "
          f"depth={p['depth']} freq={p['frequency']}")
    print(f"     {p['summary']}")

    t = h["H3_ternary"]
    print(f"\n  H3 Ternary: {t['nonzero_dims']}/10 active dims  ({t['compression']})")
    print(f"     vector: {t['ternary_vector']}")

    w = h["H4_wave"]
    print(f"\n  H4 Wave: dominant_freq={w['dominant_frequency']}  "
          f"energy={w['spectral_energy']:.4f}")
    print(f"     magnitude: {w['magnitude_profile']}")

    k = h["H7_kan"]
    print(f"\n  H7 KAN: {k['n_patterns']} patterns learned")
    for pat, norm in k["predicted_norms"].items():
        print(f"     {pat:12s} → norm={norm}")

    l = h["H2_liquid"]
    print(f"\n  H2 Liquid: experts={l['active_experts']} "
          f"weights={l['expert_weights']} complexity={l['complexity']}")

    hv = h["H8_hive"]
    print(f"\n  H8 Hive: domain='{hv['primary_domain']}'")
    for name, w in hv["domain_weights"].items():
        bar = "█" * int(w * 40)
        print(f"     {name:12s} {bar} {w:.2f}")

    tt = h["H6_ttt"]
    print(f"\n  H6 TTT: delta={tt['adaptation_delta']:.8f}  "
          f"adapted={'YES' if tt['adapted'] else 'NO'}")

    m = h["H5_memory"]
    print(f"\n  H5 Memory: top unified results")
    for r in m["unified_results"]:
        signals = " + ".join(f"{s[0]}({s[1]:.2f})" for s in r["signals"])
        print(f"     {r['name']:12s} score={r['score']:.4f}  [{signals}]")


if __name__ == "__main__":
    print("THE SYNTHETIC NEOCORTEX v2")
    print("All 8 Heresies — One Brain")
    print("=" * 60)

    print("\nLoading all heresies...")
    cortex = SyntheticNeocortex()

    print("\n--- System stats ---")
    stats = cortex.stats()
    for k, v in stats.items():
        print(f"  {k:25s}: {v}")

    print("\n--- Full query pipeline ---")
    test_roots = ["رحم", "علم", "أله", "كفر", "خلق"]
    for root in test_roots:
        result = cortex.query(root)
        if "error" not in result:
            print_query_result(result)

    print("\n\n--- Root comparisons ---")
    comparisons = [
        ("أله", "ربب", "theological allies"),
        ("أمن", "كفر", "semantic opposites"),
        ("علم", "حكم", "epistemic pair"),
        ("أرض", "سمو", "cosmic pair"),
    ]
    for r1, r2, desc in comparisons:
        cmp = cortex.compare_roots(r1, r2)
        if "error" not in cmp:
            print(f"\n  {r1} vs {r2} ({desc}):")
            print(f"    poincaré_dist={cmp['poincare_distance']:.6f}  "
                  f"spectral_sim={cmp['spectral_similarity']:.4f}  "
                  f"ternary_overlap={cmp['ternary_overlap']:.2f}")
            print(f"    domains: {cmp['r1_domain']} / {cmp['r2_domain']}  "
                  f"same={cmp['same_domain']}")

    cortex.memory.close()
    print("\n" + "=" * 60)
    print("  SYNTHETIC NEOCORTEX v2 — ALL 8 HERESIES FUSED")
    print("  The Unwalked Path is now a road.")
    print("=" * 60)
