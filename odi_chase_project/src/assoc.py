"""Phase 4: association rule mining (Apriori and FP-Growth implemented from scratch).

mlxtend is not installed in the execution environment and there is no network access, so both
algorithms are implemented in pure Python. Their outputs are cross-checked against each other.
Rules are DESCRIPTIVE associations mined on all 47 matches; they are not causal and are not used as
model inputs.
"""
import time
from collections import defaultdict
from itertools import combinations
from math import ceil
import numpy as np
import pandas as pd

# ------------------------------------------------------------------ transactions / binning
def build_transactions(st, min_balls=18):
    """One transaction per END OF OVER (the delivery that completes the 6th legal ball of an over),
    for live (eligible) states with at least `min_balls` legal balls so that all momentum windows
    exist. One state per over rather than per delivery limits (but does not remove) the strong
    within-match dependence between rows."""
    s = st[st.eligible & (st.legal_balls >= min_balls) & (st.legal_balls % 6 == 0) & (st.is_legal == 1)].copy()
    s = s.drop_duplicates(["match_id", "legal_balls"], keep="first")

    def cut(x, bins, labels):
        return pd.cut(x, bins=bins, labels=labels).astype(str)

    s["STAGE"] = cut(s.pct_complete, [0, 1 / 3, 2 / 3, 1.0001], ["early", "middle", "late"])
    s["RRR"] = cut(s.rrr, [-np.inf, 5.5, 8.0, np.inf], ["low(<=5.5)", "medium(5.5-8)", "high(>8)"])
    s["WKTS_LEFT"] = cut(s.wkts_remaining, [-1, 3, 6, 10], ["few(<=3)", "moderate(4-6)", "many(>=7)"])
    s["MOMENTUM"] = cut(s.rate_change_12_vs_earlier, [-np.inf, -1.0, 1.0, np.inf],
                        ["negative(<-1)", "stable", "positive(>+1)"])
    s["PACE_VS_REQ"] = cut(s.rate_last12_minus_rrr, [-np.inf, -1.0, 1.0, np.inf],
                           ["behind(<-1)", "level", "ahead(>+1)"])
    s["OUTCOME"] = np.where(s.chase_win == 1, "Win", "Loss")
    attrs = ["STAGE", "RRR", "WKTS_LEFT", "MOMENTUM", "PACE_VS_REQ", "OUTCOME"]
    for a in attrs:
        s[a] = a + "=" + s[a]
    return s.reset_index(drop=True), attrs


def to_item_lists(s, attrs):
    return [frozenset(r) for r in s[attrs].itertuples(index=False, name=None)]


# ------------------------------------------------------------------ Apriori (horizontal counting)
def apriori(transactions, min_support, max_len=None):
    n = len(transactions)
    minc = ceil(min_support * n)
    counts = defaultdict(int)
    for t in transactions:
        for i in t:
            counts[frozenset([i])] += 1
    freq = {k: v for k, v in counts.items() if v >= minc}
    allf = dict(freq)
    k = 2
    prev = sorted(tuple(sorted(x)) for x in freq)
    while prev and (max_len is None or k <= max_len):
        prev_set = set(prev)
        cands = set()
        for a, b in combinations(prev, 2):          # join step: share first k-2 items
            if a[:-1] == b[:-1]:
                c = a + (b[-1],)
                if all(sub in prev_set for sub in combinations(c, k - 1)):   # prune step
                    cands.add(c)
        cnt = defaultdict(int)
        for t in transactions:
            for c in cands:
                if t.issuperset(c):
                    cnt[c] += 1
        new = {c: v for c, v in cnt.items() if v >= minc}
        for c, v in new.items():
            allf[frozenset(c)] = v
        prev = sorted(new)
        k += 1
    return {k: v / n for k, v in allf.items()}, allf


# ------------------------------------------------------------------ FP-Growth
class _Node:
    __slots__ = ("item", "count", "parent", "children", "link")
    def __init__(self, item, parent):
        self.item, self.count, self.parent, self.children, self.link = item, 0, parent, {}, None


def _build_tree(weighted_transactions, minc):
    cnt = defaultdict(int)
    for t, w in weighted_transactions:
        for i in t:
            cnt[i] += w
    freq = {i: c for i, c in cnt.items() if c >= minc}
    order = {i: r for r, (i, _) in enumerate(sorted(freq.items(), key=lambda x: (-x[1], str(x[0]))))}
    root, header = _Node(None, None), {}
    for t, w in weighted_transactions:
        items = sorted((i for i in t if i in freq), key=lambda i: order[i])
        node = root
        for i in items:
            if i not in node.children:
                child = _Node(i, node)
                node.children[i] = child
                child.link = header.get(i)
                header[i] = child
            node = node.children[i]
            node.count += w
    return header, freq, order


def _mine(header, freq, order, minc, suffix, out, max_len):
    for item in sorted(freq, key=lambda i: -order[i]):         # least frequent first
        new = suffix | {item}
        out[frozenset(new)] = freq[item]
        if max_len is not None and len(new) >= max_len:
            continue
        cond = []
        node = header[item]
        while node is not None:
            path, p = [], node.parent
            while p is not None and p.item is not None:
                path.append(p.item)
                p = p.parent
            if path:
                cond.append((path, node.count))
            node = node.link
        if cond:
            h2, f2, o2 = _build_tree(cond, minc)
            if f2:
                _mine(h2, f2, o2, minc, new, out, max_len)


def fpgrowth(transactions, min_support, max_len=None):
    n = len(transactions)
    minc = ceil(min_support * n)
    header, freq, order = _build_tree([(t, 1) for t in transactions], minc)
    out = {}
    _mine(header, freq, order, minc, frozenset(), out, max_len)
    return {k: v / n for k, v in out.items()}, out


# ------------------------------------------------------------------ rules
def outcome_rules(counts, transactions, s, outcome_prefix="OUTCOME=", min_conf=0.6, min_lift=1.0,
                  min_matches=1):
    """Rules  antecedent -> OUTCOME=x  (single-item consequent), with support / confidence / lift and
    match coverage (how many distinct matches the antecedent occurs in; match-level precision)."""
    n = len(transactions)
    match_ids = s.match_id.to_numpy()
    match_outcome = s.groupby("match_id").chase_win.first()
    rows = []
    for iset, c in counts.items():
        cons = [i for i in iset if i.startswith(outcome_prefix)]
        if len(cons) != 1 or len(iset) < 2:
            continue
        cons = cons[0]
        ante = iset - {cons}
        if any(i.startswith(outcome_prefix) for i in ante):
            continue
        ca = counts.get(ante)
        if ca is None:                                  # antecedent below min support cannot happen (downward closure)
            ca = sum(1 for t in transactions if ante <= t)
        cc = counts.get(frozenset([cons])) or sum(1 for t in transactions if cons in t)
        conf, supp, lift = c / ca, c / n, (c / ca) / (cc / n)
        if conf < min_conf or lift < min_lift:
            continue
        mask = np.array([ante <= t for t in transactions])
        covered = np.unique(match_ids[mask])
        target_cls = 1 if cons.endswith("Win") else 0
        mprec = float((match_outcome.loc[covered] == target_cls).mean())
        if len(covered) < min_matches:
            continue
        rows.append({"antecedent": " & ".join(sorted(ante)), "consequent": cons, "n_antecedent": ca,
                     "support": supp, "confidence": conf, "lift": lift, "base_rate": cc / n,
                     "n_matches": len(covered), "match_level_precision": mprec, "_mask": mask})
    r = pd.DataFrame(rows)
    if len(r):
        r = r.sort_values(["lift", "support"], ascending=False).reset_index(drop=True)
    return r


def bootstrap_rules(rules, s, n_boot=300, seed=0):
    """Match-cluster bootstrap of rule confidence and lift (resample whole matches)."""
    rng = np.random.default_rng(seed)
    mids = s.match_id.to_numpy()
    uniq = np.unique(mids)
    pos = {m: np.where(mids == m)[0] for m in uniq}
    y_win = (s.OUTCOME == "OUTCOME=Win").to_numpy()
    masks = np.vstack(rules["_mask"].to_numpy())
    is_win = rules.consequent.str.endswith("Win").to_numpy()
    confs = np.zeros((n_boot, len(rules)))
    lifts = np.zeros((n_boot, len(rules)))
    for b in range(n_boot):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([pos[m] for m in pick])
        mk = masks[:, idx]
        yy = y_win[idx]
        na = mk.sum(1)
        hit = np.where(is_win, (mk & yy).sum(1), (mk & ~yy).sum(1))
        base = np.where(is_win, yy.mean(), 1 - yy.mean())
        with np.errstate(divide="ignore", invalid="ignore"):
            confs[b] = hit / na
            lifts[b] = (hit / na) / base
    out = rules.copy()
    out["conf_boot_p05"] = np.nanpercentile(confs, 5, axis=0)
    out["lift_boot_p05"] = np.nanpercentile(lifts, 5, axis=0)
    return out


def time_algo(fn, trans, sup, reps=5):
    ts = []
    for _ in range(reps):
        t = time.perf_counter(); res = fn(trans, sup); ts.append(time.perf_counter() - t)
    return res, float(np.median(ts))


def compare_algorithms(trans, supports=(0.20, 0.10, 0.05, 0.03, 0.02)):
    rows = []
    for sup in supports:
        (_, ca), ta = time_algo(apriori, trans, sup)
        (_, cf), tf = time_algo(fpgrowth, trans, sup)
        same = ca == cf
        rows.append({"min_support": sup, "n_frequent_itemsets_apriori": len(ca),
                     "n_frequent_itemsets_fpgrowth": len(cf), "identical_itemsets_and_counts": same,
                     "apriori_sec": ta, "fpgrowth_sec": tf})
    return pd.DataFrame(rows)
