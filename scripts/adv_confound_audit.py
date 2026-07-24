#!/usr/bin/env python
"""Adversarial confound-control audit for BCS v2. Everything recomputed from raw
stimuli text; the stored `gate` field is IGNORED. Per (family x N x difficulty)
cell we hunt for any residual endpoint/role/frequency/position confound."""
from __future__ import annotations
import json, re, sys
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr, pearsonr

STIM = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/galyukshev/Desktop/claude/manifolds_research/workspace_mirror/data/bcs_pilot/stimuli.jsonl"
NULL = "/Users/galyukshev/Desktop/claude/manifolds_research/workspace_mirror/data/bcs_pilot/stimuli_null.jsonl"

# relation surfaces (inv checked first; inv implies first-named is the LATER one)
REL = {
    "s0_zib":  ("zibs", "is zibbed by"),
    "s1_size": ("is smaller than", "is larger than"),
    "s1_loud": ("is quieter than", "is louder than"),
    "s1_heat": ("is cooler than", "is hotter than"),
}


def claimed_edges(stim):
    """Reconstruct CLAIMED directed edges (earlier -> later) purely from card text."""
    fwd, inv = REL[stim["family"]]
    edges = []
    for c in stim["cards"]:
        a, b, t = c["entity"], c["entity_b"], c["text"]
        if inv in t:
            earlier, later = b, a          # 'The b inv the a' => a earlier? NO: inv => first(b) is later
        elif fwd in t:
            earlier, later = a, b
        else:
            raise ValueError(f"unparseable card: {t}")
        edges.append((earlier, later))
    return edges


def topo_unique(nodes, edges):
    """Kahn; returns (unique_bool, order or None). unique iff never >1 source."""
    succ = defaultdict(set); indeg = {n: 0 for n in nodes}
    for u, v in edges:
        if v not in succ[u]:
            succ[u].add(v); indeg[v] += 1
    avail = sorted([n for n in nodes if indeg[n] == 0])
    order = []; unique = True
    while avail:
        if len(avail) > 1:
            unique = False
        u = avail.pop(0); order.append(u)
        newly = []
        for w in succ[u]:
            indeg[w] -= 1
            if indeg[w] == 0:
                newly.append(w)
        avail = sorted(avail + newly)
    complete = len(order) == len(nodes)
    return (unique and complete), (order if complete else None)


def has_cycle(nodes, edges):
    succ = defaultdict(set); indeg = {n: 0 for n in nodes}
    for u, v in edges:
        succ[u].add(v)
    color = {n: 0 for n in nodes}  # 0 white 1 gray 2 black
    def dfs(u):
        color[u] = 1
        for w in succ[u]:
            if color[w] == 1: return True
            if color[w] == 0 and dfs(w): return True
        color[u] = 2; return False
    return any(color[n] == 0 and dfs(n) for n in nodes)


def name_positions(prompt, name):
    return [m.start() for m in re.finditer(r"\b" + re.escape(name) + r"\b", prompt)]


def analyze_total_shuffle(stim):
    ents = stim["latent_order"]
    n = len(ents)
    rank = {e: i + 1 for i, e in enumerate(ents)}
    ranks = np.array([rank[e] for e in ents], float)
    prompt = stim["prompt"]
    cards = stim["cards"]

    # --- mention frequency: card mentions (as entity or entity_b) + roster (once)
    card_cnt = {e: 0 for e in ents}
    first_cnt = {e: 0 for e in ents}
    for c in cards:
        card_cnt[c["entity"]] += 1
        card_cnt[c["entity_b"]] += 1
        first_cnt[c["entity"]] += 1
    roster = stim.get("readout_order") or []
    ment = np.array([card_cnt[e] + (1 if e in roster else 0) for e in ents], float)
    degree = np.array([card_cnt[e] for e in ents], float)  # degree in comparison graph
    # regex mention count over whole prompt (independent cross-check)
    rgx_cnt = np.array([len(name_positions(prompt, e)) for e in ents], float)

    subj = np.array([first_cnt[e] / max(card_cnt[e], 1) for e in ents], float)

    # --- position: mean CHAR position over ALL mentions in prompt (cards+roster)
    meanchar = np.array([np.mean(name_positions(prompt, e)) for e in ents], float)
    # mean CARD-index position (presentation slot) of its card mentions only
    slot_pos = {e: [] for e in ents}
    for c in cards:
        slot_pos[c["entity"]].append(c["presentation_slot"])
        slot_pos[c["entity_b"]].append(c["presentation_slot"])
    meancard = np.array([np.mean(slot_pos[e]) for e in ents], float)
    # roster slot = index in readout_order
    ros_idx = {e: i for i, e in enumerate(roster)}
    roster_slot = np.array([ros_idx.get(e, -1) for e in ents], float)

    def sp(x):
        return 0.0 if np.std(x) == 0 else spearmanr(x, ranks)[0]

    return {
        "n": n,
        "ment_constant": bool(np.all(ment == ment[0])),
        "ment_val": ment.tolist() if not np.all(ment == ment[0]) else float(ment[0]),
        "degree_regular": bool(np.all(degree == degree[0])),
        "degree_val": float(degree[0]) if np.all(degree == degree[0]) else degree.tolist(),
        "rgx_constant": bool(np.all(rgx_cnt == rgx_cnt[0])),
        "rgx_val": float(rgx_cnt[0]) if np.all(rgx_cnt == rgx_cnt[0]) else rgx_cnt.tolist(),
        "subj_all_half": bool(np.allclose(subj, 0.5)),
        "subj_max_dev": float(np.max(np.abs(subj - 0.5))),
        "corr_subj": float(sp(subj)),
        "corr_meanchar": float(sp(meanchar)),
        "corr_meancard": float(sp(meancard)),
        "corr_roster_slot": float(sp(roster_slot)),
    }


def load(path):
    return [json.loads(l) for l in open(path)]


def main():
    stims = load(STIM)
    total_shuffle = [s for s in stims
                     if s.get("structure") is None and s.get("condition") == "shuffle"]
    total_all = [s for s in stims if s.get("structure") is None]

    # ---------- structural checks over ALL total-order stimuli (both conditions)
    n_deg_fail = n_uniq_fail = n_uniq_mismatch = n_ment_vary = n_subj_bad = 0
    for s in total_all:
        a = analyze_total_shuffle(s) if s["condition"] == "shuffle" else None
        # structural (condition-independent)
        ents = s["latent_order"]
        ce = claimed_edges(s)
        uniq, order = topo_unique(ents, ce)
        if not uniq:
            n_uniq_fail += 1
        elif order != ents:
            n_uniq_mismatch += 1
        # degree regular & mention
        card_cnt = defaultdict(int); first_cnt = defaultdict(int)
        for c in s["cards"]:
            card_cnt[c["entity"]] += 1; card_cnt[c["entity_b"]] += 1
            first_cnt[c["entity"]] += 1
        degs = [card_cnt[e] for e in ents]
        if len(set(degs)) != 1:
            n_deg_fail += 1
        roster = s.get("readout_order") or []
        ments = [card_cnt[e] + (1 if e in roster else 0) for e in ents]
        if len(set(ments)) != 1:
            n_ment_vary += 1
        subj = np.array([first_cnt[e] / max(card_cnt[e], 1) for e in ents])
        if not np.allclose(subj, 0.5):
            n_subj_bad += 1

    print("=== STRUCTURAL (all total-order, both conditions; N=%d) ===" % len(total_all))
    print(f"  degree not-regular stimuli : {n_deg_fail}")
    print(f"  mention (deg+roster) varies: {n_ment_vary}")
    print(f"  subj_frac != 0.5 stimuli   : {n_subj_bad}")
    print(f"  NON-unique topo order      : {n_uniq_fail}")
    print(f"  unique-but != latent_order : {n_uniq_mismatch}")

    # ---------- per-cell correlation hunt (SHUFFLE only)
    cells = defaultdict(list)
    for s in total_shuffle:
        cells[(s["family"], s["n_items"], s["difficulty"])].append(analyze_total_shuffle(s))

    print("\n=== PER-CELL (family, N, diff) SHUFFLE — max|corr| and mean(signed) ===")
    hdr = f"{'cell':28s} {'n':>4} {'mSubj':>7} {'mChar':>7} {'meanChar':>9} {'mCard':>7} {'mRoster':>8} {'meanRos':>8} {'mentK?':>6} {'subj.5?':>7}"
    print(hdr)
    worst = defaultdict(lambda: (0.0, None))
    signed_all = defaultdict(list)
    for key in sorted(cells):
        rows = cells[key]
        subj = np.array([r["corr_subj"] for r in rows])
        char = np.array([r["corr_meanchar"] for r in rows])
        card = np.array([r["corr_meancard"] for r in rows])
        ros = np.array([r["corr_roster_slot"] for r in rows])
        ment_ok = all(r["ment_constant"] for r in rows)
        subj_ok = all(r["subj_all_half"] for r in rows)
        for nm, arr in (("subj", subj), ("char", char), ("card", card), ("roster", ros)):
            mx = np.max(np.abs(arr))
            if mx > worst[nm][0]:
                worst[nm] = (mx, key)
            signed_all[nm].extend(arr.tolist())
        cellname = f"{key[0]},N={key[1]},{key[2]}"
        print(f"{cellname:28s} {len(rows):>4} {np.max(np.abs(subj)):>7.3f} "
              f"{np.max(np.abs(char)):>7.3f} {np.mean(char):>9.3f} "
              f"{np.max(np.abs(card)):>7.3f} {np.max(np.abs(ros)):>8.3f} {np.mean(ros):>8.3f} "
              f"{str(ment_ok):>6} {str(subj_ok):>7}")

    print("\n=== WORST CELL per feature (max over all shuffle cells) ===")
    for nm in ("subj", "char", "card", "roster"):
        mx, key = worst[nm]
        print(f"  {nm:8s} max|corr|={mx:.3f}  at cell {key}")
    print("\n=== POPULATION signed-mean corr (systematic-bias check; want ~0) ===")
    for nm in ("subj", "char", "card", "roster"):
        v = np.array(signed_all[nm])
        # one-sample-ish: is the mean far from 0 relative to its SE?
        se = np.std(v) / np.sqrt(len(v))
        print(f"  {nm:8s} mean={np.mean(v):+.4f}  sd={np.std(v):.3f}  se={se:.4f}  "
              f"mean/se={np.mean(v)/se:+.2f}  (n={len(v)})")


if __name__ == "__main__":
    main()
