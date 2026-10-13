"""
Full Quranic Corpus Parser — builds the complete root graph from real data.

Data sources:
  1. roots_lane.json — 1,651 roots with Lane's Lexicon definitions,
     frequencies, and morphological forms
  2. quran-morphology.txt — Quranic Arabic Corpus v0.4, per-word
     morphological annotation for all 77,430 words

Output:
  - Complete NetworkX graph with:
    - ALL root nodes (with frequency, semantic field, confidence)
    - ALL lemma/form nodes (with pattern, POS, frequency)
    - Derivation edges (root → lemma)
    - Semantic co-occurrence edges (root ↔ root, weighted by verse co-occurrence)
"""

import json
import re
from collections import defaultdict, Counter
from pathlib import Path
import networkx as nx

CORPUS_DIR = Path(__file__).parent / "corpus"


def parse_roots_lane():
    """Parse roots_lane.json — all 1,651 roots with morphological forms."""
    path = CORPUS_DIR / "roots_lane.json"
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    roots = {}
    for entry in data["roots"]:
        root_ar = entry["root"]
        roots[root_ar] = {
            "buckwalter": entry.get("root_buckwalter", ""),
            "frequency": entry.get("quran_frequency", 0),
            "summary_en": entry.get("summary_en", ""),
            "confidence": entry.get("confidence", ""),
            "source": entry.get("source", ""),
            "forms": [],
        }
        for form in entry.get("morphological_forms", []):
            roots[root_ar]["forms"].append({
                "pattern": form.get("form_pattern", ""),
                "arabic": form.get("form_arabic", ""),
                "name": form.get("form_name", ""),
                "category": form.get("form_category", ""),
                "example": form.get("example_word", ""),
                "occurrences": form.get("occurrences", 0),
            })

    return roots, data.get("metadata", {})


def parse_morphology_corpus():
    """
    Parse quran-morphology.txt — every word in the Quran with root + lemma.

    Returns:
      word_data: list of (chapter, verse, word_pos, arabic, pos, root, lemma, features_dict)
      verse_roots: dict mapping (chapter, verse) → set of roots in that verse
      root_lemmas: dict mapping root → Counter of lemma occurrences
      root_forms: dict mapping root → Counter of verb form (VF:N) occurrences
    """
    path = CORPUS_DIR / "quran-morphology.txt"

    word_data = []
    verse_roots = defaultdict(set)
    root_lemmas = defaultdict(Counter)
    root_forms = defaultdict(Counter)
    root_pos_counts = defaultdict(Counter)

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue

            parts = line.split("\t")
            if len(parts) < 4:
                continue

            position = parts[0]
            arabic = parts[1]
            pos = parts[2]
            features_str = parts[3]

            pos_parts = position.split(":")
            if len(pos_parts) < 3:
                continue
            chapter = int(pos_parts[0])
            verse = int(pos_parts[1])
            word_pos = int(pos_parts[2])

            features = {}
            for feat in features_str.split("|"):
                if ":" in feat:
                    key, val = feat.split(":", 1)
                    features[key] = val
                else:
                    features[feat] = True

            root = features.get("ROOT", None)
            lemma = features.get("LEM", None)
            verb_form = features.get("VF", None)

            if root:
                verse_roots[(chapter, verse)].add(root)
                if lemma:
                    root_lemmas[root][lemma] += 1
                if verb_form:
                    root_forms[root][verb_form] += 1
                root_pos_counts[root][pos] += 1

            word_data.append((chapter, verse, word_pos, arabic, pos, root, lemma, features))

    return word_data, dict(verse_roots), dict(root_lemmas), dict(root_forms), dict(root_pos_counts)


def build_cooccurrence_edges(verse_roots, min_cooccurrence=3):
    """
    Build semantic co-occurrence edges from verse-level root co-occurrence.

    Two roots that appear in the same verse are semantically related.
    Weight = number of verses they co-occur in.
    """
    cooccurrence = Counter()

    for (chapter, verse), roots in verse_roots.items():
        roots_list = sorted(roots)
        for i in range(len(roots_list)):
            for j in range(i + 1, len(roots_list)):
                pair = (roots_list[i], roots_list[j])
                cooccurrence[pair] += 1

    edges = []
    for (root_a, root_b), count in cooccurrence.items():
        if count >= min_cooccurrence:
            edges.append((root_a, root_b, count))

    edges.sort(key=lambda x: -x[2])
    return edges


def build_full_graph(min_cooccurrence=3):
    """Build the complete Quranic root graph from all data sources."""

    print("Parsing roots_lane.json...")
    roots_data, metadata = parse_roots_lane()
    print(f"  {len(roots_data)} roots loaded")

    print("Parsing quran-morphology.txt...")
    word_data, verse_roots, root_lemmas, root_forms, root_pos = parse_morphology_corpus()
    print(f"  {len(word_data)} word tokens parsed")
    print(f"  {len(verse_roots)} verses processed")
    print(f"  {len(root_lemmas)} unique roots found in morphology corpus")

    print("Building co-occurrence edges...")
    cooc_edges = build_cooccurrence_edges(verse_roots, min_cooccurrence)
    print(f"  {len(cooc_edges)} co-occurrence edges (min {min_cooccurrence} shared verses)")

    G = nx.DiGraph()

    all_roots_in_corpus = set(root_lemmas.keys())
    all_roots_in_lane = set(roots_data.keys())
    corpus_root_freq = Counter()
    for (ch, v), roots in verse_roots.items():
        for r in roots:
            corpus_root_freq[r] += 1

    for root_ar, data in roots_data.items():
        G.add_node(root_ar, type="root", depth=0,
                   frequency=data["frequency"],
                   summary=data["summary_en"],
                   buckwalter=data["buckwalter"],
                   confidence=data["confidence"],
                   num_forms=len(data["forms"]),
                   num_lemmas=len(root_lemmas.get(root_ar, {})))

    for root_ar in all_roots_in_corpus - all_roots_in_lane:
        freq = corpus_root_freq.get(root_ar, 0)
        if freq >= 2:
            G.add_node(root_ar, type="root", depth=0,
                       frequency=freq, summary="", buckwalter="",
                       confidence="corpus_only", num_forms=0,
                       num_lemmas=len(root_lemmas.get(root_ar, {})))

    for root_ar, data in roots_data.items():
        for form in data["forms"]:
            form_name = form["example"] or form["arabic"]
            if not form_name:
                continue
            node_id = f"{root_ar}:{form_name}"
            depth = 1 if form["pattern"].startswith("form_") else 2
            G.add_node(node_id, type="derived", depth=depth,
                       arabic=form_name, pattern=form["pattern"],
                       pattern_name=form["name"],
                       category=form["category"],
                       frequency=form["occurrences"],
                       source_root=root_ar)
            G.add_edge(root_ar, node_id, relation="derivation",
                       pattern=form["pattern"])

    for root_ar, lemmas in root_lemmas.items():
        if not G.has_node(root_ar):
            continue
        for lemma, count in lemmas.most_common():
            node_id = f"{root_ar}:{lemma}"
            if G.has_node(node_id):
                continue
            G.add_node(node_id, type="derived", depth=2,
                       arabic=lemma, pattern="corpus",
                       frequency=count, source_root=root_ar)
            G.add_edge(root_ar, node_id, relation="derivation",
                       pattern="corpus", weight=count)

    for root_a, root_b, count in cooc_edges:
        if G.has_node(root_a) and G.has_node(root_b):
            G.add_edge(root_a, root_b, relation="semantic",
                       co_occurrence=count)
            G.add_edge(root_b, root_a, relation="semantic",
                       co_occurrence=count)

    return G, metadata


def print_stats(G):
    """Print comprehensive graph statistics."""
    roots = [n for n, d in G.nodes(data=True) if d.get("type") == "root"]
    derived = [n for n, d in G.nodes(data=True) if d.get("type") == "derived"]

    deriv_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "derivation"]
    sem_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "semantic"]

    root_freqs = sorted([G.nodes[n].get("frequency", 0) for n in roots], reverse=True)

    print(f"\n{'='*60}")
    print(f"QURANIC ROOT GRAPH — COMPLETE")
    print(f"{'='*60}")
    print(f"  Root nodes:        {len(roots)}")
    print(f"  Derived nodes:     {len(derived)}")
    print(f"  Total nodes:       {G.number_of_nodes()}")
    print(f"  Derivation edges:  {len(deriv_edges)}")
    print(f"  Semantic edges:    {len(sem_edges)} (co-occurrence)")
    print(f"  Total edges:       {G.number_of_edges()}")
    print(f"\n  Top 20 roots by Quranic frequency:")
    top_roots = sorted(roots, key=lambda n: G.nodes[n].get("frequency", 0), reverse=True)[:20]
    for i, r in enumerate(top_roots, 1):
        d = G.nodes[r]
        summ = d.get("summary", "")[:60]
        print(f"    {i:2d}. {r:8s}  freq={d['frequency']:5d}  forms={d.get('num_forms',0):2d}  {summ}")

    print(f"\n  Frequency distribution:")
    print(f"    Roots with freq > 100: {sum(1 for f in root_freqs if f > 100)}")
    print(f"    Roots with freq 10-100: {sum(1 for f in root_freqs if 10 <= f <= 100)}")
    print(f"    Roots with freq 1-9:    {sum(1 for f in root_freqs if 1 <= f < 10)}")

    top_cooc = sorted(
        [(u, v, d["co_occurrence"]) for u, v, d in G.edges(data=True)
         if d.get("relation") == "semantic" and u < v],
        key=lambda x: -x[2]
    )[:15]
    print(f"\n  Top 15 co-occurring root pairs:")
    for r1, r2, count in top_cooc:
        s1 = G.nodes[r1].get("summary", "")[:25]
        s2 = G.nodes[r2].get("summary", "")[:25]
        print(f"    {r1:8s} <-> {r2:8s}  co_occ={count:4d}  ({s1} / {s2})")


def save_graph(G, path):
    """Save graph as JSON for reloading."""
    nodes = {}
    for n, d in G.nodes(data=True):
        nodes[n] = dict(d)

    edges = []
    for u, v, d in G.edges(data=True):
        edges.append({"source": u, "target": v, **d})

    with open(path, "w", encoding="utf-8") as f:
        json.dump({"nodes": nodes, "edges": edges}, f, ensure_ascii=False, indent=1)
    print(f"\n  Graph saved to {path}")


if __name__ == "__main__":
    import sys
    min_co = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    G, meta = build_full_graph(min_cooccurrence=min_co)
    print_stats(G)
    save_graph(G, "E:/ABU/NEOCORTEX/data/full_graph.json")
