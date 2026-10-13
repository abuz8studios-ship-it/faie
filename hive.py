"""
Heresy 8 — Fractal Hive: 3-Level Mixture of Experts.

Standard MoE has one router → N experts. Flat.
The brain has cortical columns → areas → hemispheres. Fractal.

This module builds a 3-level routing hierarchy:
  Level 1: Super-master routes to domain hives (theology, nature, law, etc.)
  Level 2: Domain hives route to specialist swarms
  Level 3: Specialists process the actual input

The key insight: routing IS computation. The path through
the hive carries as much information as the final output.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


class Specialist(nn.Module):
    """Level 3: A single specialist expert."""
    def __init__(self, dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, x):
        return self.net(x)


class DomainHive(nn.Module):
    """Level 2: A domain hive containing specialist swarms."""
    def __init__(self, dim, n_specialists=4):
        super().__init__()
        self.dim = dim
        self.n_specialists = n_specialists

        self.router = nn.Sequential(
            nn.Linear(dim, dim),
            nn.SiLU(),
            nn.Linear(dim, n_specialists),
        )

        self.specialists = nn.ModuleList([
            Specialist(dim) for _ in range(n_specialists)
        ])

        self.norm = nn.LayerNorm(dim)

    def forward(self, x, return_routing=False):
        logits = self.router(x)
        weights = F.softmax(logits, dim=-1)

        top1_weight, top1_idx = weights.topk(1, dim=-1)

        output = torch.zeros_like(x)
        for sid in range(self.n_specialists):
            mask = (top1_idx.squeeze(-1) == sid)
            if mask.any():
                spec_out = self.specialists[sid](x[mask])
                output[mask] = spec_out

        result = self.norm(x + output)

        if return_routing:
            return result, weights
        return result


class FractalHive(nn.Module):
    """
    Level 1: Super-master routing to domain hives.
    The full 3-level fractal MoE.
    """
    def __init__(self, embed_dim=10, hidden_dim=64,
                 n_domains=4, specialists_per_domain=4):
        super().__init__()
        self.n_domains = n_domains
        self.hidden_dim = hidden_dim

        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        self.super_router = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, n_domains),
        )

        self.domain_hives = nn.ModuleList([
            DomainHive(hidden_dim, n_specialists=specialists_per_domain)
            for _ in range(n_domains)
        ])

        self.output_proj = nn.Linear(hidden_dim, embed_dim)

        self.domain_names = [
            "theology", "nature", "law", "narrative"
        ][:n_domains]

    def forward(self, x, return_routing=False):
        h = self.input_proj(x)

        domain_logits = self.super_router(h)
        domain_weights = F.softmax(domain_logits, dim=-1)

        top2_weights, top2_indices = domain_weights.topk(
            min(2, self.n_domains), dim=-1)
        top2_weights = top2_weights / top2_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(h)
        specialist_routing = []

        for i in range(min(2, self.n_domains)):
            domain_idx = top2_indices[:, i]
            weight = top2_weights[:, i:i+1]

            for did in range(self.n_domains):
                mask = (domain_idx == did)
                if mask.any():
                    if return_routing:
                        hive_out, spec_weights = self.domain_hives[did](
                            h[mask], return_routing=True)
                        specialist_routing.append((did, spec_weights))
                    else:
                        hive_out = self.domain_hives[did](h[mask])
                    output[mask] += weight[mask] * hive_out

        result = self.output_proj(output)

        if return_routing:
            return result, {
                "domain_weights": domain_weights,
                "specialist_routing": specialist_routing,
                "top2_domains": top2_indices,
                "domain_names": self.domain_names,
            }
        return result


class HiveTrainer:
    def __init__(self, model_path="E:/ABU/NEOCORTEX/models/quranic_poincare_10d.pt",
                 graph_path="E:/ABU/NEOCORTEX/data/full_graph.json"):
        checkpoint = torch.load(model_path, weights_only=False)
        self.embeddings = checkpoint["state_dict"]["embeddings"].detach()
        self.nodes = checkpoint["nodes"]
        self.node_to_idx = checkpoint["node_to_idx"]

        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph = json.load(f)

        self.embed_dim = self.embeddings.shape[1]
        self.model = FractalHive(
            embed_dim=self.embed_dim,
            hidden_dim=64,
            n_domains=4,
            specialists_per_domain=4,
        )

        self.pairs = []
        for edge in self.graph["edges"]:
            if edge.get("relation") == "semantic":
                src, tgt = edge["source"], edge["target"]
                if src in self.node_to_idx and tgt in self.node_to_idx:
                    self.pairs.append((self.node_to_idx[src], self.node_to_idx[tgt]))

    def train(self, epochs=300, lr=0.001, print_every=50):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        pair_tensor = torch.tensor(self.pairs[:5000])

        load_balance_weight = 0.01

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            src_emb = self.embeddings[pair_tensor[:, 0]]
            tgt_emb = self.embeddings[pair_tensor[:, 1]]

            pred, routing = self.model(src_emb, return_routing=True)
            loss = F.mse_loss(pred, tgt_emb)

            domain_weights = routing["domain_weights"]
            mean_usage = domain_weights.mean(dim=0)
            balance_loss = (mean_usage * torch.log(mean_usage + 1e-8)).sum()
            total_loss = loss + load_balance_weight * balance_loss

            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % print_every == 0:
                print(f"  epoch {epoch+1:4d} | loss {total_loss.item():.6f} "
                      f"domain_balance=[{', '.join(f'{w:.2f}' for w in mean_usage.detach().tolist())}]")

    def analyze_routing(self):
        """Which domains do different root categories activate?"""
        print("\n--- Hive routing analysis ---")

        categories = {
            "divine_attrs": ["أله", "ربب", "رحم", "غفر", "عفو"],
            "natural_world": ["أرض", "سمو", "شمس", "قمر", "نجم", "بحر"],
            "ethics_law": ["حكم", "عدل", "حرم", "حلل", "أمر"],
            "knowledge": ["علم", "كتب", "قرأ", "فقه", "فهم"],
            "narrative": ["قصص", "نبأ", "أخذ", "ذهب", "خرج"],
        }

        self.model.eval()

        for cat_name, roots in categories.items():
            valid = [r for r in roots if r in self.node_to_idx]
            if not valid:
                continue

            indices = [self.node_to_idx[r] for r in valid]
            embs = self.embeddings[indices]

            with torch.no_grad():
                _, routing = self.model(embs, return_routing=True)

            domain_weights = routing["domain_weights"].mean(dim=0)
            top_domain = domain_weights.argmax().item()
            domain_name = self.model.domain_names[top_domain]

            print(f"\n  {cat_name}: → domain '{domain_name}' "
                  f"(weights: [{', '.join(f'{w:.2f}' for w in domain_weights.tolist())}])")

            for i, root in enumerate(valid):
                w = routing["domain_weights"][i]
                top2 = w.topk(2)
                print(f"    {root:6s}: {self.model.domain_names[top2.indices[0]]}({top2.values[0]:.2f})"
                      f" + {self.model.domain_names[top2.indices[1]]}({top2.values[1]:.2f})")

    def measure_specialization(self):
        """How specialized are the domain hives?"""
        print("\n--- Domain specialization ---")

        root_names = [n for n in self.nodes
                      if n in self.graph["nodes"]
                      and self.graph["nodes"][n].get("type") == "root"
                      and n in self.node_to_idx
                      and self.graph["nodes"][n].get("frequency", 0) > 10]

        if not root_names:
            print("  No qualifying roots found")
            return

        root_indices = [self.node_to_idx[r] for r in root_names]
        root_embs = self.embeddings[root_indices]

        self.model.eval()
        with torch.no_grad():
            _, routing = self.model(root_embs, return_routing=True)

        domain_weights = routing["domain_weights"]

        for did in range(self.model.n_domains):
            scores = domain_weights[:, did]
            top_roots_idx = scores.topk(min(8, len(root_names))).indices
            top_roots = [root_names[i] for i in top_roots_idx]

            print(f"\n  Domain '{self.model.domain_names[did]}' top roots: "
                  f"{', '.join(top_roots)}")

        entropy = -(domain_weights * (domain_weights + 1e-8).log()).sum(dim=-1)
        print(f"\n  Mean routing entropy: {entropy.mean().item():.4f} "
              f"(lower = more specialized)")
        print(f"  Max entropy (uniform): {np.log(self.model.n_domains):.4f}")

    def save(self, path="E:/ABU/NEOCORTEX/models/fractal_hive.pt"):
        torch.save({
            "state_dict": self.model.state_dict(),
            "embed_dim": self.embed_dim,
            "n_domains": self.model.n_domains,
            "domain_names": self.model.domain_names,
        }, path)
        print(f"\n  Fractal hive saved to {path}")


if __name__ == "__main__":
    print("Heresy 8 — Fractal Hive: 3-Level MoE")
    print("=" * 60)

    print("\nLoading data...")
    trainer = HiveTrainer()
    n_params = sum(p.numel() for p in trainer.model.parameters())
    n_specialists = trainer.model.n_domains * 4
    print(f"  Fractal hive: {n_params:,} parameters, "
          f"{trainer.model.n_domains} domains × 4 specialists = {n_specialists} total experts")

    print("\nTraining...")
    trainer.train(epochs=300, lr=0.001, print_every=50)

    trainer.analyze_routing()
    trainer.measure_specialization()
    trainer.save()

    print("\n  Heresy 8 COMPLETE.")
