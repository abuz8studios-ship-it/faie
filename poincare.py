"""
Poincaré Embedding Trainer for the Quranic Root Graph.

Maps the Arabic root derivational DAG into the Poincaré ball:
  - Roots near the center (general concepts)
  - Derived words near the boundary (specific forms)
  - Distance = semantic dissimilarity
  - Depth from origin = specificity

Uses geoopt's PoincaréBall manifold with Riemannian SGD.
Based on Nickel & Kiela (NeurIPS 2017).
"""

import torch
import torch.nn as nn
import geoopt
import numpy as np
from collections import defaultdict


class PoincareEmbedding(nn.Module):
    def __init__(self, n_nodes, dim=10, curvature=1.0):
        super().__init__()
        self.ball = geoopt.PoincareBall(c=curvature)
        init = torch.randn(n_nodes, dim) * 1e-3
        self.embeddings = geoopt.ManifoldParameter(
            self.ball.expmap0(init), manifold=self.ball
        )
        self.dim = dim

    def poincare_distance(self, u, v):
        return self.ball.dist(u, v)


class QuranicPoincareTrainer:
    def __init__(self, graph, dim=10, curvature=1.0, lr=0.01):
        self.graph = graph
        self.dim = dim
        self.lr = lr

        self.nodes = list(graph.nodes())
        self.node_to_idx = {n: i for i, n in enumerate(self.nodes)}
        self.idx_to_node = {i: n for n, i in self.node_to_idx.items()}
        self.n_nodes = len(self.nodes)

        depth_list = []
        for node in self.nodes:
            depth_list.append(graph.nodes[node].get("depth", 0))
        self.node_depths = {i: d for i, d in enumerate(depth_list)}
        self.depth_tensor = torch.tensor(depth_list, dtype=torch.float32)

        target_norms = torch.where(
            self.depth_tensor == 0, torch.tensor(0.25),
            torch.where(self.depth_tensor == 1, torch.tensor(0.55), torch.tensor(0.80))
        )
        self.target_norms = target_norms

        self.model = PoincareEmbedding(self.n_nodes, dim, curvature)

        self.pos_pairs = []
        self.edge_weights = []
        for u, v, data in graph.edges(data=True):
            if u not in self.node_to_idx or v not in self.node_to_idx:
                continue
            ui, vi = self.node_to_idx[u], self.node_to_idx[v]
            rel = data.get("relation", "derivation")
            if rel == "derivation":
                weight = 3.0
            elif rel == "semantic":
                co_occ = data.get("co_occurrence", 50)
                weight = 1.0 + min(co_occ / 100.0, 2.0)
            else:
                weight = 1.0
            self.pos_pairs.append((ui, vi))
            self.edge_weights.append(weight)

        self.pos_pairs = torch.tensor(self.pos_pairs, dtype=torch.long)
        self.edge_weights = torch.tensor(self.edge_weights, dtype=torch.float32)

    def _hierarchy_regularization_vectorized(self):
        norms = self.model.embeddings.data.norm(dim=-1)
        return ((norms - self.target_norms) ** 2).mean()

    def train(self, epochs=500, batch_size=8192, print_every=50):
        optimizer = geoopt.optim.RiemannianSGD(
            self.model.parameters(), lr=self.lr
        )

        n_edges = len(self.pos_pairs)
        losses = []

        for epoch in range(epochs):
            if n_edges <= batch_size:
                batch_idx = torch.arange(n_edges)
            else:
                batch_idx = torch.randperm(n_edges)[:batch_size]

            batch_pos = self.pos_pairs[batch_idx]
            batch_weights = self.edge_weights[batch_idx]

            optimizer.zero_grad()

            pos_u = self.model.embeddings[batch_pos[:, 0]]
            pos_v = self.model.embeddings[batch_pos[:, 1]]
            pos_dist = self.model.poincare_distance(pos_u, pos_v)
            pos_loss = (pos_dist * batch_weights).mean()

            neg_pairs = torch.randint(0, self.n_nodes, (len(batch_idx), 2), dtype=torch.long)
            neg_u = self.model.embeddings[neg_pairs[:, 0]]
            neg_v = self.model.embeddings[neg_pairs[:, 1]]
            neg_dist = self.model.poincare_distance(neg_u, neg_v)
            neg_loss = torch.clamp(12.0 - neg_dist, min=0).mean()

            hier_loss = self._hierarchy_regularization_vectorized()

            loss = pos_loss + neg_loss + 5.0 * hier_loss
            loss.backward()
            optimizer.step()

            losses.append(loss.item())

            if (epoch + 1) % print_every == 0:
                with torch.no_grad():
                    norms = self.model.embeddings.data.norm(dim=-1)
                    root_mask = self.depth_tensor == 0
                    derived_mask = self.depth_tensor > 0

                    avg_root_norm = norms[root_mask].mean().item() if root_mask.any() else 0
                    avg_derived_norm = norms[derived_mask].mean().item() if derived_mask.any() else 0

                print(f"  epoch {epoch+1:4d} | loss {loss.item():.4f} | "
                      f"pos_d {pos_dist.mean().item():.3f} neg_d {neg_dist.mean().item():.3f} | "
                      f"root_norm {avg_root_norm:.3f} derived_norm {avg_derived_norm:.3f}")

        return losses

    def get_embedding(self, node_name):
        idx = self.node_to_idx[node_name]
        return self.model.embeddings[idx].detach().cpu().numpy()

    def nearest_neighbors(self, query, k=10):
        if query not in self.node_to_idx:
            return []
        q_idx = self.node_to_idx[query]
        q_emb = self.model.embeddings[q_idx].unsqueeze(0)

        with torch.no_grad():
            all_emb = self.model.embeddings.data
            q_expanded = q_emb.expand(self.n_nodes, -1)
            dists = self.model.ball.dist(q_expanded, all_emb)
            dists[q_idx] = float("inf")
            topk_dists, topk_idx = torch.topk(dists, k, largest=False)

        result = []
        for d, i in zip(topk_dists.tolist(), topk_idx.tolist()):
            result.append((self.idx_to_node[i], d))
        return result

    def get_all_embeddings(self):
        result = {}
        emb_data = self.model.embeddings.detach().cpu().numpy()
        for node, idx in self.node_to_idx.items():
            e = emb_data[idx]
            depth = self.node_depths.get(idx, 0)
            data = self.graph.nodes[node]
            result[node] = {
                "embedding": e.tolist(),
                "norm": float(np.linalg.norm(e)),
                "depth": depth,
                "type": data.get("type", "unknown"),
                "meaning": data.get("summary", data.get("meaning", data.get("semantic_field", ""))),
                "frequency": data.get("frequency", 0),
            }
        return result

    def save(self, path):
        torch.save({
            "state_dict": self.model.state_dict(),
            "nodes": self.nodes,
            "node_to_idx": self.node_to_idx,
            "node_depths": self.node_depths,
            "dim": self.dim,
        }, path)

    @classmethod
    def load(cls, path, graph):
        checkpoint = torch.load(path, weights_only=False)
        trainer = cls(graph, dim=checkpoint["dim"])
        trainer.model.load_state_dict(checkpoint["state_dict"])
        return trainer


if __name__ == "__main__":
    from data.load_full_graph import load_graph, graph_stats

    print("Loading full Quranic root graph...")
    G = load_graph()
    stats = graph_stats(G)
    print(f"  {stats['total_nodes']} nodes ({stats['root_nodes']} roots + {stats['derived_nodes']} derived)")
    print(f"  {stats['total_edges']} edges ({stats['derivation_edges']} derivation + {stats['semantic_edges']} semantic)\n")

    print("Training Poincaré embeddings (10D, full graph)...")
    trainer = QuranicPoincareTrainer(G, dim=10, lr=1.0)
    losses = trainer.train(epochs=8000, batch_size=16384, print_every=1000)

    print("\n--- Nearest neighbors test ---")
    test_queries = ["رحم", "علم", "كتب", "أله", "نزل", "قلب"]
    for q in test_queries:
        if q in trainer.node_to_idx:
            print(f"\n  {q}:")
            neighbors = trainer.nearest_neighbors(q, k=8)
            for name, dist in neighbors:
                node_data = G.nodes[name]
                meaning = node_data.get("summary", "")[:50]
                ntype = node_data.get("type", "")
                print(f"    {dist:.3f}  {name:20s}  [{ntype}]  {meaning}")

    print("\n--- Hierarchy verification ---")
    embs = trainer.get_all_embeddings()
    root_norms = [e["norm"] for e in embs.values() if e["depth"] == 0]
    d1_norms = [e["norm"] for e in embs.values() if e["depth"] == 1]
    d2_norms = [e["norm"] for e in embs.values() if e["depth"] >= 2]

    print(f"  Root avg norm:    {np.mean(root_norms):.4f} (should be smallest — near center)")
    if d1_norms:
        print(f"  Depth-1 avg norm: {np.mean(d1_norms):.4f} (verbal forms — middle)")
    if d2_norms:
        print(f"  Depth-2 avg norm: {np.mean(d2_norms):.4f} (derived words — near boundary)")

    model_path = "E:/ABU/NEOCORTEX/models/quranic_poincare_10d.pt"
    trainer.save(model_path)
    print(f"\n  Model saved to {model_path}")
