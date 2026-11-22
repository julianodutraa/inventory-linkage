"""From accepted pairs to entities, and how to score both."""
from __future__ import annotations

from collections import defaultdict
from itertools import combinations

import numpy as np

from .generate import Record


class _DSU:
    def __init__(self, n: int):
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x


def connected_components(n: int, edges: list[tuple[int, int]]) -> list[int]:
    """Transitive closure: the textbook default, and the source of chaining."""
    d = _DSU(n)
    for i, j in edges:
        a, b = d.find(i), d.find(j)
        if a != b:
            d.parent[a] = b
    return [d.find(i) for i in range(n)]


def constrained_merge(records: list[Record], edges: list[tuple[int, int]],
                      weights: np.ndarray) -> list[int]:
    """Greedy merge by descending weight, refusing any union that would put two
    records from the same source system into one entity.

    Each source lists an asset at most once, so a cluster holding two cloud
    records is wrong by construction. This is the one piece of domain
    knowledge that the pairwise model cannot express.
    """
    d = _DSU(len(records))
    sources = {i: {r.source} for i, r in enumerate(records)}
    for k in np.argsort(-weights, kind="stable"):
        i, j = edges[k]
        a, b = d.find(i), d.find(j)
        if a == b or sources[a] & sources[b]:
            continue
        d.parent[a] = b
        sources[b] |= sources.pop(a)
    return [d.find(i) for i in range(len(records))]


def pair_prf(pred: set[tuple[int, int]], true: set[tuple[int, int]]) -> dict:
    tp = len(pred & true)
    p = tp / len(pred) if pred else 1.0
    r = tp / len(true) if true else 1.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return {"precision": round(p, 4), "recall": round(r, 4), "f1": round(f, 4),
            "tp": tp, "fp": len(pred) - tp, "fn": len(true) - tp}


def cluster_pairs(labels: list[int]) -> set[tuple[int, int]]:
    groups = defaultdict(list)
    for i, c in enumerate(labels):
        groups[c].append(i)
    out = set()
    for members in groups.values():
        out.update(combinations(sorted(members), 2))
    return out


def true_pairs(records: list[Record]) -> set[tuple[int, int]]:
    return cluster_pairs([r.entity for r in records])


def cluster_stats(labels: list[int]) -> dict:
    sizes = defaultdict(int)
    for c in labels:
        sizes[c] += 1
    s = sorted(sizes.values(), reverse=True)
    return {"n_clusters": len(s), "largest": s[0], "over_3": sum(1 for x in s if x > 3)}
