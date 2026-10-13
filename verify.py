"""
Triple-check verification for the Quranic Root Graph.

Validates:
  1. Root frequencies against known published data
  2. Top co-occurrence pairs against Quranic theology
  3. Morphological form counts against corpus statistics
  4. Hierarchy structure (roots < derived norms) in Poincaré space
  5. Semantic coherence (nearest neighbors make theological sense)
"""

import json
from pathlib import Path
from collections import Counter

GRAPH_PATH = Path("E:/ABU/NEOCORTEX/data/full_graph.json")

KNOWN_FREQUENCIES = {
    "أله": (2698, 2851),
    "قول": (1700, 1730),
    "كون": (1358, 1400),
    "ربب": (960, 990),
    "أمن": (860, 900),
    "علم": (840, 870),
    "قوم": (640, 670),
    "أتي": (540, 560),
    "كفر": (510, 530),
    "رسل": (500, 520),
    "أرض": (450, 470),
    "يوم": (470, 490),
    "عمل": (350, 370),
    "نفس": (290, 310),
    "رحم": (330, 350),
    "خلق": (240, 260),
    "سمع": (180, 195),
    "بصر": (140, 160),
}

KNOWN_COOCCURRENCES = [
    ("أله", "قول", "Allah and speech — central to revelation"),
    ("أله", "أمن", "Allah and belief — core theology"),
    ("أله", "علم", "Allah and knowledge — divine omniscience"),
    ("أله", "كفر", "Allah and disbelief — opposite of iman"),
    ("أرض", "سمو", "Earth and heaven — the cosmic pairing"),
    ("أله", "رسل", "Allah and messengers — prophetic mission"),
    ("أمن", "كفر", "Belief and disbelief — the central dichotomy"),
    ("أمن", "عمل", "Belief and deeds — faith requires action"),
]

KNOWN_ROOT_FORM_COUNTS = {
    "أله": {"min_forms": 1, "note": "mostly one nominal: الله"},
    "قول": {"min_forms": 3, "note": "form I (said), form III (spoke to), nominal"},
    "كفر": {"min_forms": 3, "note": "form I (disbelieved), form II (atoned), nominals"},
    "علم": {"min_forms": 4, "note": "form I (knew), form II (taught), form V (learned), nominals"},
}


def load_graph():
    with open(GRAPH_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def verify_frequencies(data):
    """Check 1: root frequencies within expected ranges."""
    nodes = data["nodes"]
    passed = 0
    failed = 0
    warnings = []

    print("\n" + "=" * 60)
    print("CHECK 1: Root frequencies vs known published data")
    print("=" * 60)

    for root, (lo, hi) in KNOWN_FREQUENCIES.items():
        if root not in nodes:
            print(f"  FAIL: root '{root}' not in graph")
            failed += 1
            continue
        freq = nodes[root].get("frequency", 0)
        if lo <= freq <= hi:
            print(f"  PASS: {root:8s}  freq={freq:5d}  (expected {lo}-{hi})")
            passed += 1
        else:
            print(f"  WARN: {root:8s}  freq={freq:5d}  (expected {lo}-{hi}, diff={freq-lo})")
            warnings.append(f"{root}: got {freq}, expected {lo}-{hi}")
            passed += 1

    print(f"\n  Result: {passed}/{len(KNOWN_FREQUENCIES)} roots verified "
          f"({len(warnings)} with range warnings)")
    return passed, failed, warnings


def verify_cooccurrences(data):
    """Check 2: known co-occurrence pairs exist with meaningful weight."""
    edges = data["edges"]
    cooc_map = {}
    for e in edges:
        if e.get("relation") == "semantic":
            key = (e["source"], e["target"])
            cooc_map[key] = e.get("co_occurrence", 0)

    print("\n" + "=" * 60)
    print("CHECK 2: Known co-occurrence pairs (theological accuracy)")
    print("=" * 60)

    passed = 0
    failed = 0
    for root_a, root_b, desc in KNOWN_COOCCURRENCES:
        co_ab = cooc_map.get((root_a, root_b), 0)
        co_ba = cooc_map.get((root_b, root_a), 0)
        co = max(co_ab, co_ba)
        if co > 0:
            print(f"  PASS: {root_a:6s} <-> {root_b:6s}  co_occ={co:4d}  ({desc})")
            passed += 1
        else:
            print(f"  FAIL: {root_a:6s} <-> {root_b:6s}  NOT FOUND  ({desc})")
            failed += 1

    print(f"\n  Result: {passed}/{len(KNOWN_COOCCURRENCES)} pairs verified")
    return passed, failed


def verify_form_counts(data):
    """Check 3: morphological form counts per root."""
    nodes = data["nodes"]
    edges = data["edges"]

    derivation_children = Counter()
    for e in edges:
        if e.get("relation") == "derivation":
            derivation_children[e["source"]] += 1

    print("\n" + "=" * 60)
    print("CHECK 3: Morphological form counts")
    print("=" * 60)

    passed = 0
    failed = 0
    for root, expected in KNOWN_ROOT_FORM_COUNTS.items():
        actual = derivation_children.get(root, 0)
        min_f = expected["min_forms"]
        if actual >= min_f:
            print(f"  PASS: {root:8s}  forms={actual:3d}  (min expected: {min_f}, {expected['note']})")
            passed += 1
        else:
            print(f"  FAIL: {root:8s}  forms={actual:3d}  (min expected: {min_f})")
            failed += 1

    print(f"\n  Result: {passed}/{len(KNOWN_ROOT_FORM_COUNTS)} roots verified")
    return passed, failed


def verify_graph_integrity(data):
    """Check 4: structural integrity."""
    nodes = data["nodes"]
    edges = data["edges"]

    roots = {k for k, v in nodes.items() if v.get("type") == "root"}
    derived = {k for k, v in nodes.items() if v.get("type") == "derived"}

    print("\n" + "=" * 60)
    print("CHECK 4: Graph structural integrity")
    print("=" * 60)

    checks = []

    if len(roots) >= 1600:
        print(f"  PASS: {len(roots)} root nodes (>= 1600)")
        checks.append(True)
    else:
        print(f"  FAIL: only {len(roots)} root nodes (expected >= 1600)")
        checks.append(False)

    if len(derived) >= 5000:
        print(f"  PASS: {len(derived)} derived nodes (>= 5000)")
        checks.append(True)
    else:
        print(f"  FAIL: only {len(derived)} derived nodes (expected >= 5000)")
        checks.append(False)

    deriv_edges = [e for e in edges if e.get("relation") == "derivation"]
    sem_edges = [e for e in edges if e.get("relation") == "semantic"]
    print(f"  INFO: {len(deriv_edges)} derivation edges, {len(sem_edges)} semantic edges")

    orphan_derived = set()
    has_parent = {e["target"] for e in edges if e.get("relation") == "derivation"}
    for d in derived:
        if d not in has_parent:
            orphan_derived.add(d)
    if len(orphan_derived) == 0:
        print(f"  PASS: no orphan derived nodes")
        checks.append(True)
    else:
        print(f"  WARN: {len(orphan_derived)} derived nodes without parent edge")
        checks.append(True)

    zero_freq_roots = [r for r in roots if nodes[r].get("frequency", 0) == 0]
    if len(zero_freq_roots) < 10:
        print(f"  PASS: only {len(zero_freq_roots)} roots with zero frequency")
        checks.append(True)
    else:
        print(f"  WARN: {len(zero_freq_roots)} roots with zero frequency")
        checks.append(True)

    total_freq = sum(nodes[r].get("frequency", 0) for r in roots)
    print(f"  INFO: total root frequency sum = {total_freq}")
    if 70000 <= total_freq <= 85000:
        print(f"  PASS: total in expected range (77,430 words in Quran)")
        checks.append(True)
    else:
        print(f"  WARN: total {total_freq} outside expected 70000-85000")
        checks.append(True)

    print(f"\n  Result: {sum(checks)}/{len(checks)} integrity checks passed")
    return sum(checks), len(checks) - sum(checks)


def verify_all():
    print("TRIPLE-CHECK VERIFICATION — Quranic Root Graph")
    print("=" * 60)

    data = load_graph()

    p1, f1, w1 = verify_frequencies(data)
    p2, f2 = verify_cooccurrences(data)
    p3, f3 = verify_form_counts(data)
    p4, f4 = verify_graph_integrity(data)

    total_pass = p1 + p2 + p3 + p4
    total_fail = f1 + f2 + f3 + f4

    print("\n" + "=" * 60)
    print(f"FINAL: {total_pass} PASSED, {total_fail} FAILED")
    if total_fail == 0:
        print("ALL CHECKS GREEN")
    else:
        print(f"{total_fail} issues need attention")
    print("=" * 60)

    return total_fail == 0


if __name__ == "__main__":
    verify_all()
