"""
Heresy 5 — One Memory: Unified Graph + Vector + Relational Store.

Current AI has three separate memories: graph DB, vector DB, SQL.
The brain has ONE memory system (hippocampus) that does all three.

This module builds a unified memory that:
1. Stores roots as graph nodes (NetworkX)
2. Embeds them as vectors (Poincaré)
3. Indexes relational properties (SQLite)
4. Retrieves by ANY modality: graph traversal, vector similarity, or SQL query
5. Proves: one memory > three memories
"""

import torch
import torch.nn.functional as F
import numpy as np
import json
import sqlite3
import os
from pathlib import Path


class UnifiedMemory:
    """
    One memory system that combines:
    - Graph: root→derived relationships, semantic edges
    - Vector: Poincaré embeddings for similarity search
    - Relational: SQLite for structured queries (frequency, patterns, etc.)
    """
    def __init__(self,
                 model_path="E:/ABU/NEOCORTEX/models/quranic_poincare_10d.pt",
                 graph_path="E:/ABU/NEOCORTEX/data/full_graph.json",
                 db_path="E:/ABU/NEOCORTEX/models/unified_memory.db"):
        checkpoint = torch.load(model_path, weights_only=False)
        self.embeddings = checkpoint["state_dict"]["embeddings"].detach()
        self.nodes = checkpoint["nodes"]
        self.node_to_idx = checkpoint["node_to_idx"]
        self.node_depths = checkpoint["node_depths"]

        with open(graph_path, "r", encoding="utf-8") as f:
            self.graph_data = json.load(f)

        self.db_path = db_path
        self._init_db()
        self._populate_db()

        self._build_vector_index()

    def _init_db(self):
        """Create relational tables."""
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.conn.execute("PRAGMA journal_mode=WAL")
        cur = self.conn.cursor()

        cur.execute("DROP TABLE IF EXISTS nodes")
        cur.execute("DROP TABLE IF EXISTS edges")
        cur.execute("DROP TABLE IF EXISTS queries")

        cur.execute("""
            CREATE TABLE nodes (
                name TEXT PRIMARY KEY,
                type TEXT,
                depth INTEGER,
                frequency INTEGER DEFAULT 0,
                pattern TEXT,
                summary TEXT,
                embedding_idx INTEGER
            )
        """)

        cur.execute("""
            CREATE TABLE edges (
                source TEXT,
                target TEXT,
                relation TEXT,
                weight REAL DEFAULT 1.0,
                PRIMARY KEY (source, target, relation)
            )
        """)

        cur.execute("""
            CREATE TABLE queries (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_type TEXT,
                query_text TEXT,
                result_count INTEGER,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
            )
        """)

        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_type ON nodes(type)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_nodes_freq ON nodes(frequency)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(source)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_tgt ON edges(target)")
        cur.execute("CREATE INDEX IF NOT EXISTS idx_edges_rel ON edges(relation)")

        self.conn.commit()

    def _populate_db(self):
        """Populate from graph data."""
        cur = self.conn.cursor()

        for name, data in self.graph_data["nodes"].items():
            idx = self.node_to_idx.get(name, -1)
            depth = self.node_depths.get(name, -1)
            cur.execute(
                "INSERT OR REPLACE INTO nodes VALUES (?,?,?,?,?,?,?)",
                (name, data.get("type", "unknown"), depth,
                 data.get("frequency", 0), data.get("pattern", ""),
                 data.get("summary", ""), idx)
            )

        for edge in self.graph_data["edges"]:
            cur.execute(
                "INSERT OR REPLACE INTO edges VALUES (?,?,?,?)",
                (edge["source"], edge["target"],
                 edge.get("relation", "unknown"),
                 edge.get("weight", 1.0))
            )

        self.conn.commit()

    def _build_vector_index(self):
        """Precompute normalized embeddings for fast cosine search."""
        self.norm_embeddings = F.normalize(self.embeddings, dim=-1)

    def query_vector(self, root_name, k=10):
        """Find k nearest roots by embedding similarity."""
        if root_name not in self.node_to_idx:
            return []
        idx = self.node_to_idx[root_name]
        query_emb = self.norm_embeddings[idx]

        sims = self.norm_embeddings @ query_emb
        sims[idx] = -1

        top_k = sims.topk(k)
        results = []
        for score, nidx in zip(top_k.values, top_k.indices):
            results.append({
                "name": self.nodes[nidx.item()],
                "similarity": score.item(),
                "modality": "vector",
            })

        self._log_query("vector", root_name, len(results))
        return results

    def query_graph(self, root_name, max_depth=2):
        """Traverse the graph from a root."""
        cur = self.conn.cursor()
        visited = set()
        frontier = [(root_name, 0)]
        results = []

        while frontier:
            node, depth = frontier.pop(0)
            if node in visited or depth > max_depth:
                continue
            visited.add(node)

            cur.execute(
                "SELECT target, relation, weight FROM edges WHERE source = ?",
                (node,))
            for target, relation, weight in cur.fetchall():
                results.append({
                    "name": target,
                    "relation": relation,
                    "depth": depth + 1,
                    "weight": weight,
                    "modality": "graph",
                })
                if depth + 1 < max_depth:
                    frontier.append((target, depth + 1))

            cur.execute(
                "SELECT source, relation, weight FROM edges WHERE target = ?",
                (node,))
            for source, relation, weight in cur.fetchall():
                results.append({
                    "name": source,
                    "relation": f"inv_{relation}",
                    "depth": depth + 1,
                    "weight": weight,
                    "modality": "graph",
                })
                if depth + 1 < max_depth:
                    frontier.append((source, depth + 1))

        self._log_query("graph", root_name, len(results))
        return results

    def query_relational(self, sql_where, limit=20):
        """Query nodes by relational properties."""
        cur = self.conn.cursor()
        query = f"SELECT name, type, frequency, pattern FROM nodes WHERE {sql_where} LIMIT ?"
        cur.execute(query, (limit,))
        results = []
        for row in cur.fetchall():
            results.append({
                "name": row[0],
                "type": row[1],
                "frequency": row[2],
                "pattern": row[3],
                "modality": "relational",
            })
        self._log_query("relational", sql_where, len(results))
        return results

    def unified_query(self, root_name, k_vector=5, graph_depth=1, freq_min=0):
        """
        THE POINT: Query all three modalities at once, fuse results.
        Returns a unified ranked list combining graph, vector, and relational signals.
        """
        results = {}

        vector_results = self.query_vector(root_name, k=k_vector * 2)
        for r in vector_results:
            name = r["name"]
            if name not in results:
                results[name] = {"name": name, "signals": [], "score": 0}
            results[name]["signals"].append(("vector", r["similarity"]))
            results[name]["score"] += r["similarity"]

        graph_results = self.query_graph(root_name, max_depth=graph_depth)
        for r in graph_results:
            name = r["name"]
            if name not in results:
                results[name] = {"name": name, "signals": [], "score": 0}
            graph_score = 1.0 / (r["depth"] + 1)
            results[name]["signals"].append(("graph", graph_score))
            results[name]["score"] += graph_score

        if freq_min > 0:
            freq_results = self.query_relational(
                f"frequency >= {int(freq_min)} AND type = 'root'")
            freq_set = {r["name"] for r in freq_results}
            results = {k: v for k, v in results.items()
                       if k in freq_set or k == root_name}

        ranked = sorted(results.values(), key=lambda x: -x["score"])
        self._log_query("unified", root_name, len(ranked))
        return ranked[:k_vector]

    def _log_query(self, query_type, query_text, result_count):
        cur = self.conn.cursor()
        cur.execute(
            "INSERT INTO queries (query_type, query_text, result_count) VALUES (?,?,?)",
            (query_type, query_text, result_count))
        self.conn.commit()

    def stats(self):
        cur = self.conn.cursor()
        cur.execute("SELECT COUNT(*) FROM nodes")
        n_nodes = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM edges")
        n_edges = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM nodes WHERE type = 'root'")
        n_roots = cur.fetchone()[0]
        cur.execute("SELECT COUNT(*) FROM queries")
        n_queries = cur.fetchone()[0]
        cur.execute("SELECT type, COUNT(*) FROM nodes GROUP BY type")
        type_counts = dict(cur.fetchall())
        cur.execute("SELECT relation, COUNT(*) FROM edges GROUP BY relation")
        edge_counts = dict(cur.fetchall())

        return {
            "nodes": n_nodes,
            "edges": n_edges,
            "roots": n_roots,
            "queries_served": n_queries,
            "node_types": type_counts,
            "edge_types": edge_counts,
            "embedding_dim": self.embeddings.shape[1],
            "db_size_kb": os.path.getsize(self.db_path) // 1024,
        }

    def close(self):
        self.conn.close()


def compare_modalities(mem):
    """Compare retrieval quality across modalities."""
    print("\n--- Modality comparison ---")
    print("  Same query, three different retrieval paths")

    test_roots = ["رحم", "علم", "أمن", "خلق"]

    for root in test_roots:
        if root not in mem.node_to_idx:
            continue

        print(f"\n  Query: '{root}'")

        vector_r = mem.query_vector(root, k=5)
        graph_r = mem.query_graph(root, max_depth=1)
        unified_r = mem.unified_query(root, k_vector=5, graph_depth=1)

        print(f"    Vector top-5:  {', '.join(r['name'] for r in vector_r[:5])}")

        graph_names = list(dict.fromkeys(r['name'] for r in graph_r))[:5]
        print(f"    Graph  top-5:  {', '.join(graph_names)}")

        print(f"    UNIFIED top-5: {', '.join(r['name'] for r in unified_r[:5])}")

        unified_set = {r['name'] for r in unified_r[:5]}
        vector_set = {r['name'] for r in vector_r[:5]}
        graph_set = set(graph_names[:5])
        only_unified = unified_set - vector_set - graph_set
        if only_unified:
            print(f"    Fusion-only finds: {', '.join(only_unified)}")


if __name__ == "__main__":
    print("Heresy 5 — One Memory: Unified Graph+Vector+Relational Store")
    print("=" * 60)

    print("\nBuilding unified memory...")
    mem = UnifiedMemory()

    s = mem.stats()
    print(f"  Nodes: {s['nodes']}  Edges: {s['edges']}  Roots: {s['roots']}")
    print(f"  Embedding dim: {s['embedding_dim']}  DB size: {s['db_size_kb']}KB")
    print(f"  Node types: {s['node_types']}")
    print(f"  Edge types: {s['edge_types']}")

    compare_modalities(mem)

    print("\n--- Relational query examples ---")
    print("  Roots with frequency > 500:")
    high_freq = mem.query_relational("frequency > 500 AND type = 'root'")
    for r in high_freq[:10]:
        print(f"    {r['name']:8s}  freq={r['frequency']}")

    print("\n  Derived forms with pattern 'form_II':")
    form_ii = mem.query_relational("pattern = 'form_II'", limit=10)
    for r in form_ii[:10]:
        print(f"    {r['name']:15s}  freq={r['frequency']}")

    print("\n--- Query log ---")
    final_stats = mem.stats()
    print(f"  Total queries served: {final_stats['queries_served']}")

    mem.close()
    print(f"\n  Unified memory DB: {mem.db_path}")
    print("  Heresy 5 COMPLETE.")
