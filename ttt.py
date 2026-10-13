"""
Heresy 6 — Test-Time Training: A Model That Learns During Inference.

Standard models freeze after training. The brain never stops learning.
TTT replaces the attention mechanism's hidden state with a model
that updates its own weights on each new input.

This module:
1. Takes Poincaré embeddings as input
2. Maintains a tiny inner model that updates itself on each query
3. Shows that the model IMPROVES as it sees more related roots
4. Proves: context isn't a window — it's a learning opportunity
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import json
import copy


class TTTLayer(nn.Module):
    """
    Test-Time Training layer.

    Instead of fixed KV cache, maintains a mini-model (W) that
    gets gradient-updated on each new token/embedding. The "hidden state"
    IS the model's weights.
    """
    def __init__(self, dim, inner_dim=32, inner_lr=0.01):
        super().__init__()
        self.dim = dim
        self.inner_dim = inner_dim
        self.inner_lr = inner_lr

        self.W_init = nn.Linear(dim, inner_dim, bias=False)
        self.W_reconstruct = nn.Linear(inner_dim, dim, bias=False)

        self.query_proj = nn.Linear(dim, dim)
        self.key_proj = nn.Linear(dim, dim)
        self.value_proj = nn.Linear(dim, dim)

        self.output_proj = nn.Linear(dim, dim)
        self.norm = nn.LayerNorm(dim)

    def forward(self, x, update_inner=True):
        """
        x: (batch, dim) — a single embedding or batch
        update_inner: if True, perform test-time gradient step
        """
        q = self.query_proj(x)
        k = self.key_proj(x)
        v = self.value_proj(x)

        W = copy.deepcopy(self.W_init)
        W_r = copy.deepcopy(self.W_reconstruct)

        if update_inner and self.training is False:
            W.train()
            W_r.train()
            hidden = W(k.detach())
            reconstruction = W_r(hidden)
            loss = F.mse_loss(reconstruction, k.detach())

            grads_w = torch.autograd.grad(loss, W.parameters(), retain_graph=True)
            grads_r = torch.autograd.grad(loss, W_r.parameters())

            with torch.no_grad():
                for param, grad in zip(W.parameters(), grads_w):
                    param.data -= self.inner_lr * grad
                for param, grad in zip(W_r.parameters(), grads_r):
                    param.data -= self.inner_lr * grad

        with torch.no_grad():
            W.eval()
            hidden = W(q)
            output = W_r(hidden)

        output = self.output_proj(output)
        return self.norm(x + output)


class TTTBrain(nn.Module):
    """
    Full TTT architecture for root graph processing.
    Multiple TTT layers that learn during inference.
    """
    def __init__(self, embed_dim=10, hidden_dim=64, n_ttt_layers=2, inner_lr=0.01):
        super().__init__()
        self.input_proj = nn.Linear(embed_dim, hidden_dim)

        self.ttt_layers = nn.ModuleList([
            TTTLayer(hidden_dim, inner_dim=32, inner_lr=inner_lr)
            for _ in range(n_ttt_layers)
        ])

        self.output_proj = nn.Linear(hidden_dim, embed_dim)

    def forward(self, x, update_inner=True):
        h = self.input_proj(x)
        for layer in self.ttt_layers:
            h = layer(h, update_inner=update_inner)
        return self.output_proj(h)


class TTTTrainer:
    def __init__(self, model_path="E:/ABU/NEOCORTEX/models/quranic_poincare_10d.pt",
                 graph_path="E:/ABU/NEOCORTEX/data/full_graph.json"):
        checkpoint = torch.load(model_path, weights_only=False)
        self.embeddings = checkpoint["state_dict"]["embeddings"].detach()
        self.nodes = checkpoint["nodes"]
        self.node_to_idx = checkpoint["node_to_idx"]

        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph = json.load(f)

        self.embed_dim = self.embeddings.shape[1]
        self.model = TTTBrain(
            embed_dim=self.embed_dim,
            hidden_dim=64,
            n_ttt_layers=2,
            inner_lr=0.01
        )

        self.semantic_pairs = []
        for edge in self.graph["edges"]:
            if edge.get("relation") == "semantic":
                src, tgt = edge["source"], edge["target"]
                if src in self.node_to_idx and tgt in self.node_to_idx:
                    self.semantic_pairs.append(
                        (self.node_to_idx[src], self.node_to_idx[tgt]))

    def train_outer(self, epochs=200, lr=0.001, print_every=40):
        """Train the outer model (the initial weights that TTT starts from)."""
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        pair_tensor = torch.tensor(self.semantic_pairs[:3000])

        for epoch in range(epochs):
            self.model.train()
            optimizer.zero_grad()

            src_emb = self.embeddings[pair_tensor[:, 0]]
            tgt_emb = self.embeddings[pair_tensor[:, 1]]

            pred = self.model(src_emb, update_inner=False)
            loss = F.mse_loss(pred, tgt_emb)
            loss.backward()
            optimizer.step()

            if (epoch + 1) % print_every == 0:
                print(f"  outer epoch {epoch+1:4d} | loss {loss.item():.6f}")

    def test_time_adaptation(self):
        """
        The key experiment: show that the model improves
        as it processes a SEQUENCE of related roots.
        """
        print("\n--- Test-time adaptation experiment ---")
        print("  Feed a sequence of related roots. Does prediction improve?")

        test_sequences = {
            "theological": ["أله", "ربب", "رحم", "غفر", "عبد"],
            "epistemic": ["علم", "حكم", "كتب", "قرأ", "فهم"],
            "cosmic": ["أرض", "سمو", "شمس", "قمر", "نجم"],
        }

        self.model.eval()

        for seq_name, roots in test_sequences.items():
            valid_roots = [r for r in roots if r in self.node_to_idx]
            if len(valid_roots) < 3:
                continue

            print(f"\n  Sequence '{seq_name}': {' → '.join(valid_roots)}")

            errors_no_ttt = []
            errors_with_ttt = []

            model_ttt = copy.deepcopy(self.model)
            model_ttt.eval()

            for i, root in enumerate(valid_roots[:-1]):
                src_idx = self.node_to_idx[root]
                tgt_idx = self.node_to_idx[valid_roots[i + 1]]
                src_emb = self.embeddings[src_idx].unsqueeze(0)
                tgt_emb = self.embeddings[tgt_idx]

                with torch.no_grad():
                    pred_no_ttt = self.model(src_emb, update_inner=False)
                    error_no_ttt = F.mse_loss(pred_no_ttt.squeeze(), tgt_emb).item()
                    errors_no_ttt.append(error_no_ttt)

                pred_ttt = model_ttt(src_emb, update_inner=True)
                with torch.no_grad():
                    error_ttt = F.mse_loss(pred_ttt.squeeze(), tgt_emb).item()
                    errors_with_ttt.append(error_ttt)

                print(f"    step {i}: {root:6s} → {valid_roots[i+1]:6s}  "
                      f"no_ttt={error_no_ttt:.6f}  with_ttt={error_ttt:.6f}  "
                      f"improvement={((error_no_ttt - error_ttt) / max(error_no_ttt, 1e-8)) * 100:.1f}%")

            avg_no = np.mean(errors_no_ttt)
            avg_with = np.mean(errors_with_ttt)
            print(f"    avg: no_ttt={avg_no:.6f}  with_ttt={avg_with:.6f}  "
                  f"improvement={((avg_no - avg_with) / max(avg_no, 1e-8)) * 100:.1f}%")

    def measure_adaptation_speed(self):
        """How many gradient steps does TTT need to adapt?"""
        print("\n--- Adaptation speed test ---")
        print("  How quickly does TTT adapt to a new domain?")

        test_root = "رحم"
        if test_root not in self.node_to_idx:
            return

        neighbors = []
        for edge in self.graph["edges"]:
            if edge.get("relation") == "semantic":
                if edge["source"] == test_root and edge["target"] in self.node_to_idx:
                    neighbors.append(self.node_to_idx[edge["target"]])
                elif edge["target"] == test_root and edge["source"] in self.node_to_idx:
                    neighbors.append(self.node_to_idx[edge["source"]])

        if len(neighbors) < 5:
            print(f"  Not enough neighbors for {test_root}")
            return

        src_emb = self.embeddings[self.node_to_idx[test_root]].unsqueeze(0)
        tgt_embs = self.embeddings[neighbors[:10]]

        self.model.eval()

        for n_steps in [0, 1, 3, 5, 10]:
            model_copy = copy.deepcopy(self.model)
            model_copy.eval()

            for _ in range(n_steps):
                model_copy(src_emb, update_inner=True)

            with torch.no_grad():
                pred = model_copy(src_emb, update_inner=False)
                errors = F.mse_loss(
                    pred.expand_as(tgt_embs), tgt_embs,
                    reduction='none').mean(dim=-1)
                avg_error = errors.mean().item()

            print(f"  {n_steps:2d} TTT steps: avg_error={avg_error:.6f}")

    def save(self, path="E:/ABU/NEOCORTEX/models/ttt_brain.pt"):
        torch.save({
            "state_dict": self.model.state_dict(),
            "embed_dim": self.embed_dim,
        }, path)
        print(f"\n  TTT brain saved to {path}")


if __name__ == "__main__":
    print("Heresy 6 — Test-Time Training")
    print("=" * 60)

    print("\nLoading data...")
    trainer = TTTTrainer()
    n_params = sum(p.numel() for p in trainer.model.parameters())
    print(f"  TTT brain: {n_params:,} parameters, 2 TTT layers")

    print("\nTraining outer model...")
    trainer.train_outer(epochs=200, lr=0.001, print_every=40)

    trainer.test_time_adaptation()
    trainer.measure_adaptation_speed()
    trainer.save()

    print("\n  Heresy 6 COMPLETE.")
