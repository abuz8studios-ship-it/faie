"""
CALIBRATE — Full recalibration of the Synthetic Neocortex v2.

Fixes:
1. Poincaré norms pushed to target hierarchy (0.30 / 0.55 / 0.75)
2. Ternary switched to adaptive top-5 (5/10 dims active)
3. Wave recomputed from rescaled embeddings
4. KAN retrained on new embeddings
5. Liquid retrained with routing diversity loss
6. Hive retrained with domain classification labels
7. TTT retrained with 10x higher inner learning rate
8. Memory rebuilt
9. Full fusion test
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import os
import sys
import copy

sys.path.insert(0, "E:/ABU/NEOCORTEX")
BASE = "E:/ABU/NEOCORTEX"


def phase1_poincare():
    print("\n" + "=" * 60)
    print("  PHASE 1: Poincaré Norm Rescaling")
    print("=" * 60)

    path = f"{BASE}/models/quranic_poincare_10d.pt"
    checkpoint = torch.load(path, weights_only=False)
    emb_raw = checkpoint["state_dict"]["embeddings"]
    embeddings = emb_raw.detach().clone()
    nodes = checkpoint["nodes"]
    node_depths = checkpoint["node_depths"]
    n = len(nodes)

    depths = torch.tensor([node_depths.get(i, 0) for i in range(n)])
    norms = embeddings.norm(dim=-1)

    for d in sorted(set(depths.tolist())):
        mask = depths == d
        print(f"  depth={d}: {mask.sum().item()} nodes, avg_norm={norms[mask].mean():.4f}")

    target_map = {0: 0.30, 1: 0.55}
    default_target = 0.75

    for i in range(n):
        depth = node_depths.get(i, 0)
        target = target_map.get(depth, default_target)
        current_norm = norms[i].item()

        if current_norm < 1e-8:
            embeddings[i] = torch.randn(embeddings.shape[1]) * target * 0.9
            continue

        direction = embeddings[i] / current_norm
        noise = torch.randn_like(direction) * (0.01 + depth * 0.005)
        new_dir = F.normalize(direction + noise, dim=-1)
        embeddings[i] = new_dir * target

    norms = embeddings.norm(dim=-1)
    for d in sorted(set(depths.tolist())):
        mask = depths == d
        print(f"  AFTER depth={d}: avg_norm={norms[mask].mean():.4f}")

    new_checkpoint = {
        "state_dict": {"embeddings": embeddings},
        "nodes": nodes,
        "node_to_idx": checkpoint["node_to_idx"],
        "node_depths": node_depths,
        "dim": checkpoint.get("dim", 10),
    }
    torch.save(new_checkpoint, path)
    print(f"  Saved to {path}")


def phase2_ternary():
    print("\n" + "=" * 60)
    print("  PHASE 2: Adaptive Ternary Quantization (top-5)")
    print("=" * 60)

    from ternary import adaptive_ternary_quantize, compute_compression_stats, \
        load_embeddings, load_graph, measure_preservation

    embeddings, nodes, node_to_idx, node_depths = load_embeddings()
    graph_data = load_graph()

    ternary = adaptive_ternary_quantize(embeddings, n_bits=5)
    stats = compute_compression_stats(embeddings, ternary)
    pres = measure_preservation(embeddings, ternary, nodes, node_to_idx, graph_data)

    active = (ternary != 0).float().mean(dim=-1)
    print(f"  Avg active dims: {active.mean() * embeddings.shape[1]:.1f}/10")
    print(f"  Sparsity: {stats['sparsity']:.1%}")
    print(f"  NN overlap: {pres['nn_overlap_jaccard']:.4f}")
    print(f"  Rank correlation: {pres['rank_correlation']:.4f}")
    print(f"  Theological pairs: {pres['theological_pairs_preserved']}")

    test_roots = ["رحم", "أله", "علم", "كفر", "شمس"]
    for root in test_roots:
        if root in node_to_idx:
            idx = node_to_idx[root]
            vec = ternary[idx].tolist()
            active_n = (ternary[idx] != 0).sum().item()
            print(f"    {root}: {active_n}/10 active → {vec}")

    scale = embeddings.abs().mean(dim=0)
    torch.save({
        "ternary_embeddings": ternary,
        "scale": scale,
        "method": "adaptive_top5",
        "stats": stats,
        "preservation": pres,
    }, f"{BASE}/models/ternary_embeddings.pt")
    print("  Saved.")


def phase3_wave():
    print("\n" + "=" * 60)
    print("  PHASE 3: Wave Recomputation")
    print("=" * 60)

    from wave import spectral_embedding, harmonic_field as build_harmonic, load_data

    embeddings, nodes, node_to_idx, node_depths, graph = load_data()
    spectral, magnitude, phase = spectral_embedding(embeddings)
    print(f"  FFT: {spectral.shape}, {magnitude.shape[1]} freq bins")

    field = build_harmonic(embeddings, nodes, node_to_idx, graph)

    roots = [n for n in nodes if n in graph["nodes"]
             and graph["nodes"][n].get("type") == "root"
             and n in node_to_idx]
    root_mags = torch.stack([magnitude[node_to_idx[r]] for r in roots])
    dom = root_mags.argmax(dim=-1)
    dist = torch.bincount(dom, minlength=magnitude.shape[1])
    print(f"  Dominant freq dist: {dist.tolist()}")

    torch.save({
        "spectral_embeddings": spectral,
        "magnitude": magnitude,
        "phase": phase,
        "harmonic_field": {k: {kk: vv.tolist() if hasattr(vv, 'tolist') else vv
                               for kk, vv in v.items()}
                          for k, v in field.items()},
    }, f"{BASE}/models/wave_data.pt")
    print("  Saved.")


def phase4_kan():
    print("\n" + "=" * 60)
    print("  PHASE 4: KAN Retrain")
    print("=" * 60)

    from kan import MorphKANTrainer
    trainer = MorphKANTrainer()
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"  KAN: {n_params:,} params, {trainer.n_patterns} patterns")
    trainer.train(epochs=800, lr=0.002, print_every=200)
    trainer.save()


def phase5_liquid():
    print("\n" + "=" * 60)
    print("  PHASE 5: Liquid Retrain (diversity loss)")
    print("=" * 60)

    from liquid import LiquidBrain

    checkpoint = torch.load(f"{BASE}/models/quranic_poincare_10d.pt", weights_only=False)
    embeddings = checkpoint["state_dict"]["embeddings"].detach()
    nodes = checkpoint["nodes"]
    node_to_idx = checkpoint["node_to_idx"]

    with open(f"{BASE}/data/full_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)

    embed_dim = embeddings.shape[1]
    model = LiquidBrain(embed_dim=embed_dim, hidden_dim=64, n_experts=8, n_ltc_steps=3)

    pairs = []
    for edge in graph["edges"]:
        if edge.get("relation") == "semantic":
            src, tgt = edge["source"], edge["target"]
            if src in node_to_idx and tgt in node_to_idx:
                pairs.append((node_to_idx[src], node_to_idx[tgt]))

    pair_tensor = torch.tensor(pairs[:5000])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(600):
        model.train()
        optimizer.zero_grad()

        src_emb = embeddings[pair_tensor[:, 0]]
        tgt_emb = embeddings[pair_tensor[:, 1]]

        pred, routing = model(src_emb, return_routing=True)
        recon_loss = F.mse_loss(pred, tgt_emb)

        gate1 = routing["gate1"]
        usage = gate1.mean(dim=0)
        usage_entropy = -(usage * (usage + 1e-8).log()).sum()
        sample_entropy = -(gate1 * (gate1 + 1e-8).log()).sum(dim=-1).mean()

        total_loss = recon_loss - 0.1 * usage_entropy + 0.05 * sample_entropy
        total_loss.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            print(f"  epoch {epoch + 1:4d} | loss {total_loss.item():.6f} "
                  f"recon={recon_loss.item():.6f} "
                  f"sample_ent={sample_entropy.item():.3f}")

    model.eval()
    print("\n  Routing check:")
    cats = {
        "high": ["أله", "قول", "كون", "ربب"],
        "rare": ["ميد", "تجر", "بخل", "لون"],
    }
    for cat, roots in cats.items():
        valid = [r for r in roots if r in node_to_idx]
        if not valid:
            continue
        idxs = [node_to_idx[r] for r in valid]
        embs = embeddings[idxs]
        with torch.no_grad():
            _, rt = model(embs, return_routing=True)
        for i, root in enumerate(valid):
            g = rt["gate1"][i]
            top3 = g.topk(3)
            print(f"    {cat:5s} {root}: e={top3.indices.tolist()} "
                  f"w=[{', '.join(f'{w:.3f}' for w in top3.values.tolist())}]")

    torch.save({
        "state_dict": model.state_dict(),
        "embed_dim": embed_dim,
    }, f"{BASE}/models/liquid_brain.pt")
    print("  Saved.")


def phase6_hive():
    print("\n" + "=" * 60)
    print("  PHASE 6: Hive Retrain (domain labels)")
    print("=" * 60)

    from hive import FractalHive

    checkpoint = torch.load(f"{BASE}/models/quranic_poincare_10d.pt", weights_only=False)
    embeddings = checkpoint["state_dict"]["embeddings"].detach()
    nodes = checkpoint["nodes"]
    node_to_idx = checkpoint["node_to_idx"]

    with open(f"{BASE}/data/full_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)

    embed_dim = embeddings.shape[1]

    seeds = {
        0: ["أله", "ربب", "عبد", "رحم", "غفر", "صلو", "دعو"],
        1: ["أرض", "سمو", "شمس", "قمر", "نجم", "بحر", "ماء"],
        2: ["حكم", "عدل", "حرم", "حلل", "أمر", "نهي", "شهد"],
        3: ["قصص", "نبأ", "ذهب", "خرج", "أخذ", "جعل", "أتي"],
    }

    centroids = {}
    for did, slist in seeds.items():
        valid = [s for s in slist if s in node_to_idx]
        if valid:
            centroids[did] = embeddings[[node_to_idx[s] for s in valid]].mean(dim=0)

    roots = [n for n in nodes if n in graph["nodes"]
             and graph["nodes"][n].get("type") == "root"
             and n in node_to_idx]

    root_idxs, root_labels = [], []
    for root in roots:
        idx = node_to_idx[root]
        emb = embeddings[idx]
        dists = {d: (emb - c).norm().item() for d, c in centroids.items()}
        root_idxs.append(idx)
        root_labels.append(min(dists, key=dists.get))

    idx_t = torch.tensor(root_idxs)
    lbl_t = torch.tensor(root_labels)

    names = ["theology", "nature", "law", "narrative"]
    for d in range(4):
        print(f"  {names[d]}: {(lbl_t == d).sum().item()} roots")

    model = FractalHive(embed_dim=embed_dim, hidden_dim=64, n_domains=4,
                        specialists_per_domain=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.002)

    sem_pairs = []
    for edge in graph["edges"]:
        if edge.get("relation") == "semantic":
            src, tgt = edge["source"], edge["target"]
            if src in node_to_idx and tgt in node_to_idx:
                sem_pairs.append((node_to_idx[src], node_to_idx[tgt]))
    sem_t = torch.tensor(sem_pairs[:5000]) if sem_pairs else None

    for epoch in range(500):
        model.train()
        optimizer.zero_grad()

        root_embs = embeddings[idx_t]
        _, routing = model(root_embs, return_routing=True)
        dw = routing["domain_weights"]
        class_loss = F.cross_entropy(dw, lbl_t)

        recon_loss = torch.tensor(0.0)
        if sem_t is not None and len(sem_t) > 0:
            pred = model(embeddings[sem_t[:, 0]])
            recon_loss = F.mse_loss(pred, embeddings[sem_t[:, 1]])

        total = class_loss + 0.5 * recon_loss
        total.backward()
        optimizer.step()

        if (epoch + 1) % 100 == 0:
            acc = (dw.argmax(dim=-1) == lbl_t).float().mean()
            print(f"  epoch {epoch + 1:4d} | loss {total.item():.4f} "
                  f"class={class_loss.item():.4f} acc={acc.item():.1%}")

    model.eval()
    print("\n  Routing check:")
    cats = {
        "divine": ["أله", "ربب", "رحم", "غفر"],
        "nature": ["أرض", "شمس", "قمر", "بحر"],
        "law": ["حكم", "عدل", "حرم", "أمر"],
        "story": ["قصص", "نبأ", "ذهب", "خرج"],
    }
    for cat, cat_roots in cats.items():
        valid = [r for r in cat_roots if r in node_to_idx]
        if not valid:
            continue
        embs = embeddings[[node_to_idx[r] for r in valid]]
        with torch.no_grad():
            _, rt = model(embs, return_routing=True)
        dw = rt["domain_weights"].mean(dim=0)
        top = dw.argmax().item()
        print(f"    {cat:8s} → {names[top]} [{', '.join(f'{w:.2f}' for w in dw.tolist())}]")

    model.domain_names = names
    torch.save({
        "state_dict": model.state_dict(),
        "embed_dim": embed_dim,
        "n_domains": 4,
        "domain_names": names,
    }, f"{BASE}/models/fractal_hive.pt")
    print("  Saved.")


def phase7_ttt():
    print("\n" + "=" * 60)
    print("  PHASE 7: TTT Retrain (inner_lr=0.1)")
    print("=" * 60)

    from ttt import TTTBrain

    checkpoint = torch.load(f"{BASE}/models/quranic_poincare_10d.pt", weights_only=False)
    embeddings = checkpoint["state_dict"]["embeddings"].detach()
    nodes = checkpoint["nodes"]
    node_to_idx = checkpoint["node_to_idx"]

    with open(f"{BASE}/data/full_graph.json", "r", encoding="utf-8") as f:
        graph = json.load(f)

    embed_dim = embeddings.shape[1]
    model = TTTBrain(embed_dim=embed_dim, hidden_dim=64, n_ttt_layers=2, inner_lr=0.1)

    pairs = []
    for edge in graph["edges"]:
        if edge.get("relation") == "semantic":
            src, tgt = edge["source"], edge["target"]
            if src in node_to_idx and tgt in node_to_idx:
                pairs.append((node_to_idx[src], node_to_idx[tgt]))

    pair_t = torch.tensor(pairs[:3000])
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(300):
        model.train()
        optimizer.zero_grad()
        pred = model(embeddings[pair_t[:, 0]], update_inner=False)
        loss = F.mse_loss(pred, embeddings[pair_t[:, 1]])
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 60 == 0:
            print(f"  epoch {epoch + 1:4d} | loss {loss.item():.6f}")

    model.eval()
    print("\n  Adaptation test:")
    seq = ["أله", "ربب", "رحم", "غفر", "عبد"]
    valid = [r for r in seq if r in node_to_idx]

    model_ttt = copy.deepcopy(model)
    model_ttt.eval()

    for i in range(len(valid) - 1):
        src = embeddings[node_to_idx[valid[i]]].unsqueeze(0)
        tgt = embeddings[node_to_idx[valid[i + 1]]]

        with torch.no_grad():
            pred_no = model(src, update_inner=False)
            err_no = F.mse_loss(pred_no.squeeze(), tgt).item()

        pred_yes = model_ttt(src, update_inner=True)
        with torch.no_grad():
            err_yes = F.mse_loss(pred_yes.squeeze(), tgt).item()

        impr = ((err_no - err_yes) / max(err_no, 1e-8)) * 100
        print(f"    {valid[i]}→{valid[i + 1]}: no_ttt={err_no:.6f} "
              f"ttt={err_yes:.6f} Δ={impr:+.1f}%")

    torch.save({
        "state_dict": model.state_dict(),
        "embed_dim": embed_dim,
        "inner_lr": 0.1,
    }, f"{BASE}/models/ttt_brain.pt")
    print("  Saved.")


def phase8_memory():
    print("\n" + "=" * 60)
    print("  PHASE 8: Memory Rebuild")
    print("=" * 60)

    db_path = f"{BASE}/models/unified_memory.db"
    if os.path.exists(db_path):
        os.remove(db_path)

    from memory import UnifiedMemory
    mem = UnifiedMemory()
    s = mem.stats()
    print(f"  Nodes: {s['nodes']}  Edges: {s['edges']}  DB: {s['db_size_kb']}KB")

    for root in ["رحم", "أله", "شمس"]:
        r = mem.unified_query(root, k_vector=5, graph_depth=1)
        print(f"  '{root}' → {', '.join(x['name'] for x in r)}")

    mem.close()
    print("  Saved.")


def phase9_fusion():
    print("\n" + "=" * 60)
    print("  PHASE 9: Calibrated Fusion Test")
    print("=" * 60)

    from neocortex import SyntheticNeocortex, print_query_result

    cortex = SyntheticNeocortex()
    stats = cortex.stats()
    print(f"\n  Total params: {stats['total_neural_params']:,}")

    for root in ["رحم", "أله", "شمس", "حكم", "قصص"]:
        result = cortex.query(root)
        if "error" not in result:
            print_query_result(result)

    print("\n--- Comparisons ---")
    for r1, r2, desc in [("أله", "ربب", "theological"),
                          ("أمن", "كفر", "opposites"),
                          ("شمس", "قمر", "celestial"),
                          ("حكم", "عدل", "legal")]:
        cmp = cortex.compare_roots(r1, r2)
        if "error" not in cmp:
            print(f"  {r1}↔{r2} ({desc}): dist={cmp['poincare_distance']:.4f} "
                  f"spec={cmp['spectral_similarity']:.4f} "
                  f"tern={cmp['ternary_overlap']:.2f} "
                  f"dom={cmp['r1_domain']}/{cmp['r2_domain']}")

    cortex.memory.close()

    print("\n" + "=" * 60)
    print("  CALIBRATION COMPLETE — ALL 9 PHASES DONE")
    print("=" * 60)


if __name__ == "__main__":
    print("SYNTHETIC NEOCORTEX v2 — FULL CALIBRATION")
    print("=" * 60)

    phase1_poincare()
    phase2_ternary()
    phase3_wave()
    phase4_kan()
    phase5_liquid()
    phase6_hive()
    phase7_ttt()
    phase8_memory()
    phase9_fusion()
