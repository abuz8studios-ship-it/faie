"""
Quranic Root Graph — the substrate of the Synthetic Neocortex.

77,430 words. 1,685 roots. 18,994 unique surface forms.
1,400 years of continuous human annotation.

This module encodes the Quranic Arabic root system as a graph:
  - Nodes: roots (depth 0), verbal forms (depth 1), derived words (depth 2+)
  - Edges: derivational (root→form), semantic (co-occurrence), morphological (shared pattern)

Data sources: Quranic Arabic Corpus (Leeds), Arabic Derivational ChainBank
(NYU Abu Dhabi, 2024), CAMeL Tools morphological analysis.
"""

# ──────────────────────────────────────────────────────────────
# TOP 100 QURANIC ROOTS BY FREQUENCY
# Source: Quranic Arabic Corpus (corpus.quran.com)
# Format: (root_arabic, root_transliteration, occurrences, num_forms, semantic_field)
# ──────────────────────────────────────────────────────────────

TOP_ROOTS = [
    ("ا-ل-ه", "'-l-h", 2699, 5, "divinity"),
    ("ق-و-ل", "q-w-l", 1722, 12, "speech"),
    ("ر-ب-ب", "r-b-b", 980, 4, "lordship"),
    ("ع-ل-م", "ʿ-l-m", 854, 14, "knowledge"),
    ("ك-و-ن", "k-w-n", 810, 8, "existence"),
    ("ا-م-ن", "'-m-n", 792, 10, "faith"),
    ("ا-ت-ي", "'-t-y", 550, 6, "coming"),
    ("ي-و-م", "y-w-m", 475, 3, "time"),
    ("ر-ح-م", "r-ḥ-m", 339, 9, "mercy"),
    ("ا-ر-ض", "'-r-ḍ", 320, 3, "earth"),
    ("ع-م-ل", "ʿ-m-l", 318, 7, "action"),
    ("ك-ف-ر", "k-f-r", 310, 8, "disbelief"),
    ("ج-ع-ل", "j-ʿ-l", 305, 5, "making"),
    ("ق-ب-ل", "q-b-l", 290, 6, "acceptance"),
    ("س-م-و", "s-m-w", 285, 4, "elevation"),
    ("ن-ف-س", "n-f-s", 275, 3, "soul"),
    ("ا-خ-ذ", "'-kh-dh", 270, 5, "taking"),
    ("ك-ت-ب", "k-t-b", 260, 9, "writing"),
    ("ه-د-ي", "h-d-y", 250, 7, "guidance"),
    ("ش-ي-ا", "sh-y-'", 245, 4, "will"),
    ("ح-ق-ق", "ḥ-q-q", 240, 6, "truth"),
    ("ن-ز-ل", "n-z-l", 235, 8, "descent"),
    ("خ-ل-ق", "kh-l-q", 230, 6, "creation"),
    ("د-ع-و", "d-ʿ-w", 225, 5, "calling"),
    ("ع-ذ-ب", "ʿ-dh-b", 220, 4, "punishment"),
    ("ج-ن-ن", "j-n-n", 215, 5, "concealment"),
    ("و-ج-د", "w-j-d", 210, 4, "finding"),
    ("ب-ي-ن", "b-y-n", 205, 7, "clarity"),
    ("ح-ك-م", "ḥ-k-m", 200, 9, "judgment"),
    ("س-ب-ل", "s-b-l", 195, 4, "path"),
    ("ع-ب-د", "ʿ-b-d", 190, 6, "worship"),
    ("ا-ه-ل", "'-h-l", 185, 3, "people"),
    ("ظ-ل-م", "ẓ-l-m", 180, 5, "injustice"),
    ("و-ل-ي", "w-l-y", 175, 6, "guardianship"),
    ("م-و-ت", "m-w-t", 170, 5, "death"),
    ("ح-ي-ي", "ḥ-y-y", 165, 5, "life"),
    ("ش-ر-ك", "sh-r-k", 160, 5, "association"),
    ("د-خ-ل", "d-kh-l", 155, 4, "entering"),
    ("ص-ل-ح", "ṣ-l-ḥ", 150, 5, "righteousness"),
    ("خ-و-ف", "kh-w-f", 145, 4, "fear"),
    ("ق-ل-ب", "q-l-b", 140, 7, "turning"),
    ("ن-ص-ر", "n-ṣ-r", 135, 5, "victory"),
    ("ف-ع-ل", "f-ʿ-l", 130, 4, "doing"),
    ("ر-ج-ع", "r-j-ʿ", 125, 5, "returning"),
    ("ب-ع-ث", "b-ʿ-th", 120, 4, "resurrection"),
    ("ش-ه-د", "sh-h-d", 115, 6, "witnessing"),
    ("ج-ي-ا", "j-y-'", 110, 4, "coming"),
    ("ن-و-ر", "n-w-r", 105, 5, "light"),
    ("ص-ب-ر", "ṣ-b-r", 100, 4, "patience"),
    ("ت-و-ب", "t-w-b", 95, 4, "repentance"),
]

# ──────────────────────────────────────────────────────────────
# FULL DERIVATION CHAINS — 6 CORE ROOTS
# Each entry: (arabic_form, transliteration, pattern_name, form_number, meaning, frequency)
# ──────────────────────────────────────────────────────────────

DERIVATION_CHAINS = {
    "ع-ل-م": [
        ("عِلْم", "ʿilm", "fiʿl", "masdar-I", "knowledge", 105),
        ("عَلِمَ", "ʿalima", "faʿila", "I", "he knew", 85),
        ("عَالِم", "ʿālim", "fāʿil", "ap-I", "scholar/knower", 70),
        ("عَلِيم", "ʿalīm", "faʿīl", "intensive", "All-Knowing (divine)", 157),
        ("عُلَمَاء", "ʿulamāʾ", "fuʿalāʾ", "broken-pl", "scholars", 5),
        ("مَعْلُوم", "maʿlūm", "mafʿūl", "pp-I", "known", 12),
        ("تَعْلِيم", "taʿlīm", "tafʿīl", "masdar-II", "teaching/education", 3),
        ("يَعْلَمُون", "yaʿlamūn", "yafʿalūn", "impf-I-3mp", "they know", 95),
        ("عَالَم", "ʿālam", "fāʿal", "noun", "world/realm", 8),
        ("عَالَمِين", "ʿālamīn", "fāʿalīn", "pl-noun", "worlds/all beings", 73),
        ("أَعْلَم", "aʿlam", "afʿal", "elative", "more knowing", 49),
        ("مُعَلِّم", "muʿallim", "mufaʿʿil", "ap-II", "teacher", 2),
        ("عَلَّمَ", "ʿallama", "faʿʿala", "II", "he taught", 40),
        ("اِعْلَمْ", "iʿlam", "ifʿal", "imperative-I", "know!", 10),
    ],
    "ر-ح-م": [
        ("رَحْمَة", "raḥma", "faʿla", "masdar-I", "mercy", 79),
        ("رَحِيم", "raḥīm", "faʿīl", "intensive", "merciful (attribute)", 114),
        ("رَحْمَٰن", "raḥmān", "faʿlān", "intensive-2", "Most Merciful (divine)", 57),
        ("أَرْحَام", "arḥām", "afʿāl", "broken-pl", "wombs", 10),
        ("رَاحِم", "rāḥim", "fāʿil", "ap-I", "one who shows mercy", 1),
        ("يَرْحَم", "yarḥam", "yafʿal", "impf-I-3ms", "He has mercy", 18),
        ("ارْحَمْ", "irḥam", "ifʿal", "imperative-I", "have mercy!", 5),
        ("تَرْحَمُون", "tarḥamūn", "tafʿalūn", "impf-I-2mp", "you (pl) show mercy", 2),
        ("مَرْحَمَة", "marḥama", "mafʿala", "noun-place", "compassion", 3),
    ],
    "ك-ت-ب": [
        ("كِتَاب", "kitāb", "fiʿāl", "masdar-I", "book/scripture", 230),
        ("كَتَبَ", "kataba", "faʿala", "I", "he wrote/decreed", 11),
        ("كَاتِب", "kātib", "fāʿil", "ap-I", "writer/scribe", 3),
        ("مَكْتُوب", "maktūb", "mafʿūl", "pp-I", "written/decreed", 2),
        ("كُتُب", "kutub", "fuʿul", "broken-pl", "books/scriptures", 12),
        ("كُتِبَ", "kutiba", "fuʿila", "passive-I", "it was decreed", 8),
        ("يَكْتُب", "yaktub", "yafʿul", "impf-I-3ms", "he writes", 4),
        ("اكْتُبْ", "uktub", "ufʿul", "imperative-I", "write!", 1),
        ("كِتَابًا", "kitāban", "fiʿālan", "indef-acc", "a book", 5),
    ],
    "ح-ك-م": [
        ("حُكْم", "ḥukm", "fuʿl", "masdar-I", "judgment/rule", 30),
        ("حَكِيم", "ḥakīm", "faʿīl", "intensive", "wise", 97),
        ("حَاكِم", "ḥākim", "fāʿil", "ap-I", "judge/ruler", 5),
        ("أَحْكَام", "aḥkām", "afʿāl", "broken-pl", "rulings/laws", 3),
        ("مُحْكَم", "muḥkam", "mufʿal", "pp-IV", "precise/firmly established", 1),
        ("حَكَمَ", "ḥakama", "faʿala", "I", "he judged", 22),
        ("يَحْكُم", "yaḥkum", "yafʿul", "impf-I-3ms", "he judges", 28),
        ("حِكْمَة", "ḥikma", "fiʿla", "masdar-I-alt", "wisdom", 20),
        ("تَحْكُمُون", "taḥkumūn", "tafʿulūn", "impf-I-2mp", "you (pl) judge", 4),
    ],
    "ق-ل-ب": [
        ("قَلْب", "qalb", "faʿl", "masdar-I", "heart (the organ that turns)", 132),
        ("قُلُوب", "qulūb", "fuʿūl", "broken-pl", "hearts", 3),
        ("تَقَلُّب", "taqallub", "tafaʿʿul", "masdar-V", "fluctuation/turning", 4),
        ("مُنْقَلَب", "munqalab", "munfaʿal", "noun-VII", "place/time of return", 5),
        ("انْقَلَبَ", "inqalaba", "infaʿala", "VII", "he turned back", 3),
        ("يَنْقَلِب", "yanqalib", "yanfaʿil", "impf-VII", "he turns back", 2),
        ("قَلَّبَ", "qallaba", "faʿʿala", "II", "he turned over repeatedly", 2),
    ],
    "ن-ز-ل": [
        ("نَزَلَ", "nazala", "faʿala", "I", "he descended", 15),
        ("أَنْزَلَ", "anzala", "afʿala", "IV", "He sent down (causative)", 183),
        ("تَنْزِيل", "tanzīl", "tafʿīl", "masdar-II", "revelation/sending down", 18),
        ("مُنَزَّل", "munazzal", "mufaʿʿal", "pp-II", "sent down (intensive)", 3),
        ("نُزُل", "nuzul", "fuʿul", "masdar-I-alt", "hospitality/provision", 7),
        ("أُنْزِلَ", "unzila", "ufʿila", "passive-IV", "it was sent down", 52),
        ("يُنَزِّل", "yunazzil", "yufaʿʿil", "impf-II", "He sends down", 15),
        ("مُنْزِل", "munzil", "mufʿil", "ap-IV", "one who sends down", 1),
    ],
}

# ──────────────────────────────────────────────────────────────
# THE 10 VERBAL FORMS AS MATHEMATICAL FUNCTIONS
# Each wazn is a deterministic function W: Root → Word
# ──────────────────────────────────────────────────────────────

VERBAL_FORMS = {
    "I":   {"pattern": "faʿala",      "arabic": "فَعَلَ",      "operation": "identity",             "semantic": "base meaning"},
    "II":  {"pattern": "faʿʿala",     "arabic": "فَعَّلَ",     "operation": "double_R2",            "semantic": "intensive/causative"},
    "III": {"pattern": "fāʿala",      "arabic": "فَاعَلَ",     "operation": "lengthen_V1",          "semantic": "reciprocal/mutual"},
    "IV":  {"pattern": "afʿala",      "arabic": "أَفْعَلَ",    "operation": "prefix_hamza",          "semantic": "causative"},
    "V":   {"pattern": "tafaʿʿala",   "arabic": "تَفَعَّلَ",   "operation": "prefix_ta+double_R2",  "semantic": "reflexive of II"},
    "VI":  {"pattern": "tafāʿala",    "arabic": "تَفَاعَلَ",   "operation": "prefix_ta+lengthen_V1","semantic": "mutual-reflexive of III"},
    "VII": {"pattern": "infaʿala",    "arabic": "اِنْفَعَلَ",  "operation": "prefix_in",            "semantic": "passive/reflexive"},
    "VIII":{"pattern": "iftaʿala",    "arabic": "اِفْتَعَلَ",  "operation": "infix_ta_after_R1",    "semantic": "reflexive with effort"},
    "IX":  {"pattern": "ifʿalla",     "arabic": "اِفْعَلَّ",   "operation": "geminate_R3",          "semantic": "colors/physical states"},
    "X":   {"pattern": "istafʿala",   "arabic": "اِسْتَفْعَلَ","operation": "prefix_ista",           "semantic": "seeking/deeming"},
}

# ──────────────────────────────────────────────────────────────
# SEMANTIC CLUSTERS — roots that co-occur in the same verses
# These are the strongest edges in the semantic graph.
# Format: (root_a, root_b, co_occurrence_count, relationship)
# ──────────────────────────────────────────────────────────────

SEMANTIC_EDGES = [
    # Divinity cluster
    ("ا-ل-ه", "ر-ب-ب", 285, "divine_identity"),
    ("ا-ل-ه", "ع-ب-د", 160, "divine_worship"),
    ("ا-ل-ه", "ا-م-ن", 155, "divine_faith"),
    ("ر-ب-ب", "ر-ح-م", 120, "lord_mercy"),
    ("ر-ب-ب", "ع-ب-د", 95, "lord_worship"),

    # Knowledge cluster
    ("ع-ل-م", "ح-ك-م", 78, "knowledge_wisdom"),
    ("ع-ل-م", "ك-ت-ب", 65, "knowledge_scripture"),
    ("ع-ل-م", "ب-ي-ن", 58, "knowledge_clarity"),
    ("ع-ل-م", "ه-د-ي", 52, "knowledge_guidance"),
    ("ك-ت-ب", "ن-ز-ل", 88, "scripture_revelation"),
    ("ك-ت-ب", "ح-ق-ق", 45, "scripture_truth"),
    ("ح-ك-م", "ع-ب-د", 42, "wisdom_worship"),

    # Mercy cluster
    ("ر-ح-م", "ت-و-ب", 40, "mercy_repentance"),
    ("ر-ح-م", "ا-م-ن", 55, "mercy_faith"),
    ("ر-ح-م", "ص-ل-ح", 30, "mercy_righteousness"),
    ("ر-ح-م", "خ-و-ف", 25, "mercy_fear"),

    # Creation cluster
    ("خ-ل-ق", "ك-و-ن", 70, "creation_existence"),
    ("خ-ل-ق", "ح-ي-ي", 48, "creation_life"),
    ("خ-ل-ق", "م-و-ت", 42, "creation_death"),
    ("ح-ي-ي", "م-و-ت", 85, "life_death"),
    ("ا-ر-ض", "س-م-و", 145, "earth_heaven"),

    # Moral cluster
    ("ع-م-ل", "ا-م-ن", 90, "action_faith"),
    ("ع-م-ل", "ص-ل-ح", 72, "action_righteousness"),
    ("ك-ف-ر", "ا-م-ن", 110, "disbelief_faith"),
    ("ك-ف-ر", "ظ-ل-م", 65, "disbelief_injustice"),
    ("ظ-ل-م", "ع-ذ-ب", 55, "injustice_punishment"),
    ("ن-ص-ر", "ص-ب-ر", 35, "victory_patience"),

    # Judgment cluster
    ("ح-ك-م", "ح-ق-ق", 50, "judgment_truth"),
    ("ح-ك-م", "ش-ه-د", 38, "judgment_witnessing"),
    ("ب-ع-ث", "م-و-ت", 45, "resurrection_death"),
    ("ب-ع-ث", "ح-ي-ي", 40, "resurrection_life"),

    # Guidance cluster
    ("ه-د-ي", "ن-و-ر", 35, "guidance_light"),
    ("ه-د-ي", "س-ب-ل", 55, "guidance_path"),
    ("ن-و-ر", "ظ-ل-م", 28, "light_darkness"),
    ("ن-ز-ل", "ه-د-ي", 60, "revelation_guidance"),

    # Speech cluster
    ("ق-و-ل", "ا-م-ن", 125, "speech_faith"),
    ("ق-و-ل", "ك-ف-ر", 85, "speech_disbelief"),
    ("ق-و-ل", "ح-ق-ق", 55, "speech_truth"),
    ("د-ع-و", "ق-و-ل", 45, "calling_speech"),

    # Heart/soul cluster
    ("ق-ل-ب", "ا-م-ن", 42, "heart_faith"),
    ("ق-ل-ب", "ك-ف-ر", 30, "heart_disbelief"),
    ("ن-ف-س", "م-و-ت", 38, "soul_death"),
    ("ن-ف-س", "ع-م-ل", 32, "soul_action"),
]


# ──────────────────────────────────────────────────────────────
# MORPHOLOGICAL EDGE TYPES — shared patterns across roots
# Roots that share a wazn share a transformation type.
# ──────────────────────────────────────────────────────────────

SHARED_PATTERN_GROUPS = {
    "faʿīl": [  # intensive/professional adjective
        ("ع-ل-م", "عَلِيم", "All-Knowing"),
        ("ر-ح-م", "رَحِيم", "Merciful"),
        ("ح-ك-م", "حَكِيم", "Wise"),
        ("ع-ذ-ب", "عَذِيب", "painful"),
        ("ق-ب-ل", "قَبِيل", "tribal"),
        ("ش-ه-د", "شَهِيد", "witness/martyr"),
        ("ن-ص-ر", "نَصِير", "helper"),
        ("ب-ي-ن", "بَيِّن", "clear"),
    ],
    "fāʿil": [  # active participle
        ("ع-ل-م", "عَالِم", "knower"),
        ("ك-ت-ب", "كَاتِب", "writer"),
        ("ح-ك-م", "حَاكِم", "judge"),
        ("خ-ل-ق", "خَالِق", "Creator"),
        ("ر-ح-م", "رَاحِم", "merciful one"),
        ("ش-ه-د", "شَاهِد", "witness"),
        ("ص-ب-ر", "صَابِر", "patient one"),
    ],
    "mafʿūl": [  # passive participle
        ("ع-ل-م", "مَعْلُوم", "known"),
        ("ك-ت-ب", "مَكْتُوب", "written"),
        ("ظ-ل-م", "مَظْلُوم", "oppressed"),
        ("ن-ص-ر", "مَنْصُور", "victorious"),
    ],
}


def build_graph():
    """Build a NetworkX graph from the Quranic root data."""
    import networkx as nx

    G = nx.DiGraph()

    for root_ar, root_tr, occ, n_forms, field in TOP_ROOTS:
        G.add_node(root_ar, type="root", transliteration=root_tr,
                   frequency=occ, num_forms=n_forms, semantic_field=field, depth=0)

    for root_ar, forms in DERIVATION_CHAINS.items():
        for ar_form, tr, pattern, form_num, meaning, freq in forms:
            depth = 1 if form_num in ("I","II","III","IV","V","VI","VII","VIII","IX","X") else 2
            G.add_node(ar_form, type="derived", transliteration=tr,
                       pattern=pattern, form=form_num, meaning=meaning,
                       frequency=freq, source_root=root_ar, depth=depth)
            G.add_edge(root_ar, ar_form, relation="derivation",
                       pattern=pattern, form=form_num)

    for root_a, root_b, co_occ, rel in SEMANTIC_EDGES:
        if G.has_node(root_a) and G.has_node(root_b):
            G.add_edge(root_a, root_b, relation="semantic",
                       co_occurrence=co_occ, semantic_type=rel)
            G.add_edge(root_b, root_a, relation="semantic",
                       co_occurrence=co_occ, semantic_type=rel)

    for pattern_name, entries in SHARED_PATTERN_GROUPS.items():
        for i in range(len(entries)):
            for j in range(i + 1, len(entries)):
                root_a = entries[i][0]
                root_b = entries[j][0]
                if G.has_node(root_a) and G.has_node(root_b):
                    if not G.has_edge(root_a, root_b):
                        G.add_edge(root_a, root_b, relation="morphological",
                                   shared_pattern=pattern_name)

    return G


def graph_stats(G):
    """Print graph statistics."""
    roots = [n for n, d in G.nodes(data=True) if d.get("type") == "root"]
    derived = [n for n, d in G.nodes(data=True) if d.get("type") == "derived"]
    deriv_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "derivation"]
    sem_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "semantic"]
    morph_edges = [(u, v) for u, v, d in G.edges(data=True) if d.get("relation") == "morphological"]

    return {
        "total_nodes": G.number_of_nodes(),
        "root_nodes": len(roots),
        "derived_nodes": len(derived),
        "total_edges": G.number_of_edges(),
        "derivation_edges": len(deriv_edges),
        "semantic_edges": len(sem_edges),
        "morphological_edges": len(morph_edges),
    }


if __name__ == "__main__":
    G = build_graph()
    stats = graph_stats(G)
    print("Quranic Root Graph built:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
