"""Load the complete Quranic root graph from the pre-built JSON."""

import json
from pathlib import Path
import networkx as nx

GRAPH_PATH = Path(__file__).parent / "full_graph.json"


def load_graph(path=None):
    path = path or GRAPH_PATH
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    G = nx.DiGraph()
    for node_id, attrs in data["nodes"].items():
        G.add_node(node_id, **attrs)

    for edge in data["edges"]:
        src = edge.pop("source")
        tgt = edge.pop("target")
        G.add_edge(src, tgt, **edge)

    return G


def graph_stats(G):
    roots = [n for n, d in G.nodes(data=True) if d.get("type") == "root"]
    derived = [n for n, d in G.nodes(data=True) if d.get("type") == "derived"]
    deriv_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "derivation")
    sem_edges = sum(1 for _, _, d in G.edges(data=True) if d.get("relation") == "semantic")

    top_roots = sorted(roots, key=lambda n: G.nodes[n].get("frequency", 0), reverse=True)[:20]

    return {
        "total_nodes": G.number_of_nodes(),
        "root_nodes": len(roots),
        "derived_nodes": len(derived),
        "total_edges": G.number_of_edges(),
        "derivation_edges": deriv_edges,
        "semantic_edges": sem_edges,
        "top_roots": [
            {"root": r, "frequency": G.nodes[r].get("frequency", 0),
             "summary": G.nodes[r].get("summary", "")[:80]}
            for r in top_roots
        ],
    }
