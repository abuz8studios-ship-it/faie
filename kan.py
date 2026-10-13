"""
Heresy 7 — Kolmogorov-Arnold Network for Arabic Morphological Functions.

Standard neural networks have fixed activations on nodes, learnable weights on edges.
KANs flip this: LEARNABLE SPLINE FUNCTIONS on edges, summation on nodes.

Here, each wazn (morphological pattern) IS a learnable spline transform:
  Root embedding (10D) + Pattern ID → Derived form embedding (10D)

The KAN doesn't memorize root→form mappings. It learns the FUNCTION that
generates derived forms from roots — the same function Arabic has used
for 1,400 years.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
from pathlib import Path


class BSplineBasis(nn.Module):
    """B-spline basis functions for KAN edges."""
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order

        h = 2.0 / grid_size
        grid = torch.linspace(-1 - h * spline_order, 1 + h * spline_order,
                              grid_size + 2 * spline_order + 1)
        self.register_buffer('grid', grid)

        self.spline_weight = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * 0.1
        )
        self.base_weight = nn.Parameter(
            torch.randn(out_features, in_features) * (1.0 / np.sqrt(in_features))
        )

    def b_spline_basis(self, x, grid, k):
        """Compute B-spline basis values using Cox-de Boor recursion."""
        x = x.unsqueeze(-1)
        if k == 0:
            return ((x >= grid[:-1]) & (x < grid[1:])).float()
        else:
            b_prev = self.b_spline_basis(x.squeeze(-1), grid, k - 1)
            left_num = x.squeeze(-1) - grid[:-(k + 1)]
            left_den = grid[k:-1] - grid[:-(k + 1)]
            right_num = grid[k + 1:] - x.squeeze(-1)
            right_den = grid[k + 1:] - grid[1:-k]

            left = torch.where(left_den.abs() > 1e-8,
                             left_num / left_den * b_prev[..., :-1],
                             torch.zeros_like(b_prev[..., :-1]))
            right = torch.where(right_den.abs() > 1e-8,
                              right_num / right_den * b_prev[..., 1:],
                              torch.zeros_like(b_prev[..., 1:]))
            return left + right

    def forward(self, x):
        batch_size = x.shape[0]
        x_clamped = torch.clamp(x, -1.0, 1.0)

        bases = self.b_spline_basis(x_clamped, self.grid, self.spline_order)

        spline_out = torch.einsum('bi,oig,big->bo', x_clamped,
                                   self.spline_weight,
                                   bases.unsqueeze(1).expand(-1, self.out_features, -1, -1)
                                   .reshape(batch_size, self.out_features, -1)
                                   [..., :self.spline_weight.shape[-1]])

        base_out = F.linear(F.silu(x), self.base_weight)

        return spline_out + base_out


class KANLayer(nn.Module):
    """Single KAN layer — learnable functions on edges, sum on nodes."""
    def __init__(self, in_dim, out_dim, grid_size=5):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.edge_fns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1, grid_size),
                nn.SiLU(),
                nn.Linear(grid_size, 1)
            ) for _ in range(in_dim * out_dim)
        ])

        self.scale = nn.Parameter(torch.ones(out_dim, in_dim) * 0.1)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x):
        batch = x.shape[0]
        outputs = torch.zeros(batch, self.out_dim, device=x.device)

        for j in range(self.out_dim):
            for i in range(self.in_dim):
                xi = x[:, i:i+1]
                fn_idx = j * self.in_dim + i
                phi = self.edge_fns[fn_idx](xi).squeeze(-1)
                outputs[:, j] += self.scale[j, i] * phi

        return outputs + self.bias


class MorphKAN(nn.Module):
    """
    KAN that learns Arabic morphological functions.

    Input: root_embedding (10D) + pattern_encoding (one-hot, ~20 patterns)
    Output: predicted derived_form_embedding (10D)

    Each KAN layer learns a FUNCTION, not a weight. The learned functions
    ARE the awzan — the morphological patterns that transform roots into words.
    """
    def __init__(self, embed_dim=10, n_patterns=20, hidden=32, grid_size=8):
        super().__init__()
        input_dim = embed_dim + n_patterns

        self.layer1 = nn.Linear(input_dim, hidden)
        self.kan1_edges = nn.ModuleList([
            nn.Sequential(nn.Linear(1, grid_size), nn.Tanh(), nn.Linear(grid_size, 1))
            for _ in range(hidden)
        ])
        self.kan1_scale = nn.Parameter(torch.ones(hidden) * 0.5)

        self.layer2 = nn.Linear(hidden, hidden)
        self.kan2_edges = nn.ModuleList([
            nn.Sequential(nn.Linear(1, grid_size), nn.Tanh(), nn.Linear(grid_size, 1))
            for _ in range(hidden)
        ])
        self.kan2_scale = nn.Parameter(torch.ones(hidden) * 0.5)

        self.output = nn.Linear(hidden, embed_dim)
        self.norm1 = nn.LayerNorm(hidden)
        self.norm2 = nn.LayerNorm(hidden)

    def forward(self, root_emb, pattern_onehot):
        x = torch.cat([root_emb, pattern_onehot], dim=-1)

        h = self.layer1(x)
        kan_out = torch.zeros_like(h)
        for i, fn in enumerate(self.kan1_edges):
            kan_out[:, i] = fn(h[:, i:i+1]).squeeze(-1) * self.kan1_scale[i]
        h = self.norm1(h + kan_out)

        h2 = self.layer2(h)
        kan_out2 = torch.zeros_like(h2)
        for i, fn in enumerate(self.kan2_edges):
            kan_out2[:, i] = fn(h2[:, i:i+1]).squeeze(-1) * self.kan2_scale[i]
        h2 = self.norm2(h2 + kan_out2)

        return self.output(h2)


class MorphKANTrainer:
    """Train the KAN on root→derived form mappings from the Quranic graph."""

    def __init__(self, graph_path="E:/ABU/NEOCORTEX/data/full_graph.json",
                 model_path="E:/ABU/NEOCORTEX/models/quranic_poincare_10d.pt"):
        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph_data = json.load(f)

        checkpoint = torch.load(model_path, weights_only=False)
        self.nodes = checkpoint["nodes"]
        self.node_to_idx = checkpoint["node_to_idx"]
        emb_data = checkpoint["state_dict"]["embeddings"]
        self.embeddings = emb_data.detach()

        self.patterns = sorted(set(
            self.graph_data["nodes"][n].get("pattern", "")
            for n in self.graph_data["nodes"]
            if self.graph_data["nodes"][n].get("type") == "derived"
            and self.graph_data["nodes"][n].get("pattern", "")
        ))
        self.pattern_to_idx = {p: i for i, p in enumerate(self.patterns)}
        self.n_patterns = len(self.patterns)
        print(f"  {self.n_patterns} unique morphological patterns found")

        self.triples = []
        for edge in self.graph_data["edges"]:
            if edge.get("relation") != "derivation":
                continue
            root = edge["source"]
            derived = edge["target"]
            if root not in self.node_to_idx or derived not in self.node_to_idx:
                continue
            pattern = self.graph_data["nodes"][derived].get("pattern", "")
            if pattern not in self.pattern_to_idx:
                continue

            root_idx = self.node_to_idx[root]
            derived_idx = self.node_to_idx[derived]
            pattern_idx = self.pattern_to_idx[pattern]

            self.triples.append((root_idx, pattern_idx, derived_idx))

        print(f"  {len(self.triples)} (root, pattern, derived) training triples")

        self.embed_dim = self.embeddings.shape[1]
        self.model = MorphKAN(
            embed_dim=self.embed_dim,
            n_patterns=self.n_patterns,
            hidden=48,
            grid_size=8
        )

    def train(self, epochs=500, lr=0.001, print_every=50):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs)

        root_indices = torch.tensor([t[0] for t in self.triples])
        pattern_indices = torch.tensor([t[1] for t in self.triples])
        derived_indices = torch.tensor([t[2] for t in self.triples])

        root_embs = self.embeddings[root_indices]
        derived_embs = self.embeddings[derived_indices]
        pattern_onehot = F.one_hot(pattern_indices, self.n_patterns).float()

        n = len(self.triples)
        split = int(0.9 * n)
        perm = torch.randperm(n)
        train_idx = perm[:split]
        val_idx = perm[split:]

        best_val_loss = float('inf')
        losses = []

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            pred = self.model(root_embs[train_idx], pattern_onehot[train_idx])
            target = derived_embs[train_idx]

            loss = F.mse_loss(pred, target)
            cos_sim = F.cosine_similarity(pred, target, dim=-1).mean()

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            losses.append(loss.item())

            if (epoch + 1) % print_every == 0:
                self.model.eval()
                with torch.no_grad():
                    val_pred = self.model(root_embs[val_idx], pattern_onehot[val_idx])
                    val_loss = F.mse_loss(val_pred, derived_embs[val_idx])
                    val_cos = F.cosine_similarity(val_pred, derived_embs[val_idx], dim=-1).mean()

                if val_loss.item() < best_val_loss:
                    best_val_loss = val_loss.item()

                print(f"  epoch {epoch+1:4d} | train_loss {loss.item():.6f} "
                      f"cos_sim {cos_sim.item():.4f} | "
                      f"val_loss {val_loss.item():.6f} val_cos {val_cos.item():.4f}")

        return losses

    def predict_derivation(self, root_name, pattern_name):
        """Given a root and pattern, predict the derived form embedding."""
        if root_name not in self.node_to_idx:
            return None
        if pattern_name not in self.pattern_to_idx:
            return None

        root_idx = self.node_to_idx[root_name]
        root_emb = self.embeddings[root_idx].unsqueeze(0)
        pattern_oh = F.one_hot(
            torch.tensor([self.pattern_to_idx[pattern_name]]),
            self.n_patterns
        ).float()

        self.model.eval()
        with torch.no_grad():
            pred = self.model(root_emb, pattern_oh)
        return pred.squeeze(0).numpy()

    def evaluate_pattern_consistency(self):
        """Test: does the same pattern produce similar TRANSFORMATIONS across roots?"""
        print("\n--- Pattern consistency test ---")
        print("  Same wazn applied to different roots should produce parallel shifts\n")

        for pattern in ["form_I", "form_II", "form_IV", "corpus"]:
            if pattern not in self.pattern_to_idx:
                continue

            roots_with_pattern = []
            for t in self.triples:
                if t[1] == self.pattern_to_idx[pattern]:
                    root_name = self.nodes[t[0]]
                    roots_with_pattern.append(t)
            if len(roots_with_pattern) < 5:
                continue

            deltas = []
            for root_idx, pat_idx, derived_idx in roots_with_pattern[:20]:
                root_emb = self.embeddings[root_idx]
                derived_emb = self.embeddings[derived_idx]
                delta = derived_emb - root_emb
                deltas.append(delta)

            deltas = torch.stack(deltas)
            mean_delta = deltas.mean(dim=0)
            cosines = F.cosine_similarity(deltas, mean_delta.unsqueeze(0), dim=-1)

            print(f"  Pattern '{pattern}' ({len(roots_with_pattern)} instances):")
            print(f"    Mean delta norm: {mean_delta.norm().item():.4f}")
            print(f"    Cosine to mean:  {cosines.mean().item():.4f} "
                  f"(1.0 = perfectly consistent transform)")
            print(f"    Std of cosines:  {cosines.std().item():.4f}")

    def save(self, path="E:/ABU/NEOCORTEX/models/morph_kan.pt"):
        torch.save({
            "state_dict": self.model.state_dict(),
            "patterns": self.patterns,
            "pattern_to_idx": self.pattern_to_idx,
            "embed_dim": self.embed_dim,
            "n_patterns": self.n_patterns,
        }, path)
        print(f"\n  KAN model saved to {path}")


if __name__ == "__main__":
    print("Heresy 7 — KAN Morphological Function Learner")
    print("=" * 60)

    print("\nLoading root graph + Poincaré embeddings...")
    trainer = MorphKANTrainer()

    print(f"\nTraining KAN ({sum(p.numel() for p in trainer.model.parameters())} params)...")
    losses = trainer.train(epochs=800, lr=0.002, print_every=100)

    trainer.evaluate_pattern_consistency()

    trainer.save()

    print("\n--- Derivation prediction test ---")
    test_roots = ["رحم", "علم", "كتب", "أله", "خلق"]
    test_patterns = ["form_I", "form_II", "corpus"]
    for root in test_roots:
        if root not in trainer.node_to_idx:
            continue
        print(f"\n  Root: {root}")
        for pat in test_patterns:
            pred = trainer.predict_derivation(root, pat)
            if pred is not None:
                print(f"    {pat:12s} → predicted embedding norm: {np.linalg.norm(pred):.4f}")

    print("\n  Heresy 7 COMPLETE.")
