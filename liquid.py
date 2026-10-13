"""
Heresy 2 — Liquid Architecture: Input-Adaptive Dynamic Computation.

Transformers are stone — fixed weights, fixed computation paths.
Liquid brains are water — different inputs activate different circuits.

This module builds input-adaptive gating for the Quranic root graph:
  - Each root query activates a DIFFERENT computational path
  - High-frequency roots (أله, قول) get deep processing
  - Rare roots get broader, exploratory paths
  - The network learns WHICH path to take from the input itself
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json


class LiquidGate(nn.Module):
    """
    Input-adaptive gate that routes computation dynamically.
    Inspired by LFM2's gated short convolutions.
    """
    def __init__(self, dim, n_experts=8):
        super().__init__()
        self.dim = dim
        self.n_experts = n_experts

        self.router = nn.Sequential(
            nn.Linear(dim, dim * 2),
            nn.SiLU(),
            nn.Linear(dim * 2, n_experts),
        )

        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.SiLU(),
                nn.Linear(dim * 2, dim),
            ) for _ in range(n_experts)
        ])

        self.gate_norm = nn.LayerNorm(dim)

    def forward(self, x):
        gate_logits = self.router(x)
        gate_weights = F.softmax(gate_logits, dim=-1)

        top2_weights, top2_indices = gate_weights.topk(2, dim=-1)
        top2_weights = top2_weights / top2_weights.sum(dim=-1, keepdim=True)

        output = torch.zeros_like(x)
        for i in range(2):
            expert_idx = top2_indices[:, i]
            weight = top2_weights[:, i:i+1]
            for eid in range(self.n_experts):
                mask = (expert_idx == eid)
                if mask.any():
                    expert_out = self.experts[eid](x[mask])
                    output[mask] += weight[mask] * expert_out

        return self.gate_norm(x + output), gate_weights


class LiquidTimeConstant(nn.Module):
    """
    Simplified Liquid Time-Constant (LTC) cell.
    The time constant adapts to the input — different inputs
    cause different temporal dynamics.
    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        self.tau_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Sigmoid(),
        )
        self.f_net = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.Tanh(),
        )

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = torch.zeros_like(x)

        combined = torch.cat([x, hidden], dim=-1)
        tau = self.tau_net(combined) + 0.1
        f = self.f_net(combined)

        new_hidden = hidden + (1.0 / tau) * (f - hidden)
        return new_hidden


class LiquidBrain(nn.Module):
    """
    Full liquid architecture for root graph processing.
    Combines adaptive gating + liquid time constants + dynamic routing.
    """
    def __init__(self, embed_dim=10, hidden_dim=64, n_experts=8, n_ltc_steps=3):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_ltc_steps = n_ltc_steps

        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        self.gate1 = LiquidGate(hidden_dim, n_experts)
        self.ltc = LiquidTimeConstant(hidden_dim)
        self.gate2 = LiquidGate(hidden_dim, n_experts)

        self.output_proj = nn.Linear(hidden_dim, embed_dim)
        self.complexity_head = nn.Linear(hidden_dim, 1)

    def forward(self, x, return_routing=False):
        h = self.input_proj(x)

        h, gate1_weights = self.gate1(h)

        state = h
        for _ in range(self.n_ltc_steps):
            state = self.ltc(h, state)

        h_final, gate2_weights = self.gate2(state)

        output = self.output_proj(h_final)
        complexity = torch.sigmoid(self.complexity_head(h_final))

        if return_routing:
            return output, {
                "gate1": gate1_weights,
                "gate2": gate2_weights,
                "complexity": complexity,
                "n_ltc_steps": self.n_ltc_steps,
            }
        return output


class LiquidTrainer:
    def __init__(self, model_path="E:/ABU/NEOCORTEX/models/quranic_poincare_10d.pt",
                 graph_path="E:/ABU/NEOCORTEX/data/full_graph.json"):
        checkpoint = torch.load(model_path, weights_only=False)
        self.embeddings = checkpoint["state_dict"]["embeddings"].detach()
        self.nodes = checkpoint["nodes"]
        self.node_to_idx = checkpoint["node_to_idx"]

        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph = json.load(f)

        self.embed_dim = self.embeddings.shape[1]
        self.model = LiquidBrain(
            embed_dim=self.embed_dim,
            hidden_dim=64,
            n_experts=8,
            n_ltc_steps=3
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

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            src_emb = self.embeddings[pair_tensor[:, 0]]
            tgt_emb = self.embeddings[pair_tensor[:, 1]]

            pred = self.model(src_emb)
            loss = F.mse_loss(pred, tgt_emb)

            neg_idx = torch.randint(0, len(self.nodes), (len(pair_tensor),))
            neg_emb = self.embeddings[neg_idx]
            neg_pred = self.model(src_emb)
            neg_loss = -torch.clamp(F.mse_loss(neg_pred, neg_emb) - 0.5, max=0).mean()

            total_loss = loss + 0.1 * neg_loss
            total_loss.backward()
            optimizer.step()

            if (epoch + 1) % print_every == 0:
                print(f"  epoch {epoch+1:4d} | loss {total_loss.item():.6f}")

    def analyze_routing(self):
        """Analyze which experts fire for which roots."""
        print("\n--- Routing analysis ---")

        test_roots = {
            "high_freq": ["أله", "قول", "كون", "ربب", "أمن"],
            "medium_freq": ["خلق", "رحم", "نزل", "سمع", "بصر"],
            "rare": ["ميد", "تجر", "بخل", "لون", "نطف"],
        }

        for category, roots in test_roots.items():
            print(f"\n  {category}:")
            for root in roots:
                if root not in self.node_to_idx:
                    continue
                idx = self.node_to_idx[root]
                emb = self.embeddings[idx].unsqueeze(0)

                self.model.eval()
                with torch.no_grad():
                    _, routing = self.model(emb, return_routing=True)

                gate1 = routing["gate1"].squeeze().numpy()
                top_experts = np.argsort(-gate1)[:3]
                complexity = routing["complexity"].item()

                freq = self.graph["nodes"].get(root, {}).get("frequency", 0)
                print(f"    {root:8s} (freq={freq:4d}): "
                      f"experts=[{','.join(str(e) for e in top_experts)}] "
                      f"weights=[{','.join(f'{gate1[e]:.2f}' for e in top_experts)}] "
                      f"complexity={complexity:.3f}")

    def save(self, path="E:/ABU/NEOCORTEX/models/liquid_brain.pt"):
        torch.save({
            "state_dict": self.model.state_dict(),
            "embed_dim": self.embed_dim,
        }, path)
        print(f"\n  Liquid brain saved to {path}")


if __name__ == "__main__":
    print("Heresy 2 — Liquid Architecture")
    print("=" * 60)

    print("\nLoading data...")
    trainer = LiquidTrainer()
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"  Liquid brain: {n_params:,} parameters, 8 experts, 3 LTC steps")

    print("\nTraining...")
    trainer.train(epochs=300, lr=0.001, print_every=50)

    trainer.analyze_routing()
    trainer.save()

    print("\n  Heresy 2 COMPLETE.")
