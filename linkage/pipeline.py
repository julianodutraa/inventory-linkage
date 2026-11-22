"""End to end: block, compare, fit, decide, cluster."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .cluster import (cluster_pairs, cluster_stats, connected_components,
                      constrained_merge, pair_prf, true_pairs)
from .compare import FIELDS, N_LEVELS, block, comparison_matrix
from .fellegi_sunter import FSModel, fit_em, model_thresholds
from .generate import Record


@dataclass
class Run:
    records: list[Record]
    pairs: list[tuple[int, int]]
    G: np.ndarray
    model: FSModel
    weights: np.ndarray
    post: np.ndarray
    upper: float
    lower: float
    is_match: np.ndarray  # ground truth per candidate pair, evaluation only


ID_FIELDS = ["name_alpha", "name_num", "ip"]


def fit(records: list[Record], fields: list[str] | None = None, k: int = 3,
        mu: float = 0.01, lam: float = 0.01, seed: int = 0) -> Run:
    pairs = block(records)
    G = comparison_matrix(records, pairs)
    idx = [FIELDS.index(f) for f in fields] if fields else list(range(len(FIELDS)))
    Gs = G[:, idx]
    names = [FIELDS[i] for i in idx]
    model = fit_em(Gs, [N_LEVELS[i] for i in idx], k=k, seed=seed,
                   id_fields=[names.index(f) for f in ID_FIELDS if f in names])
    w = model.weights(Gs)
    post = model.posterior(Gs)
    upper, lower = model_thresholds(post, w, mu, lam)
    truth = np.array([records[i].entity == records[j].entity for i, j in pairs])
    return Run(records, pairs, Gs, model, w, post, upper, lower, truth)


def decide(run: Run, review: dict[int, bool] | None = None, cut: str = "upper") -> np.ndarray:
    """Pairwise link decisions. ``cut='upper'`` links only above the upper
    threshold (review band rejected); ``cut='half'`` links where the posterior
    is above 0.5. ``review`` overrides decisions for reviewed pair indices."""
    if cut == "upper":
        link = run.weights >= run.upper
    else:
        link = run.post >= 0.5
    link = link.copy()
    for k, v in (review or {}).items():
        link[k] = v
    return link


def score(run: Run, link: np.ndarray) -> dict:
    tp = true_pairs(run.records)
    edges = [run.pairs[k] for k in np.where(link)[0]]
    ew = run.weights[link]
    pairwise = pair_prf(set(edges), tp & set(run.pairs))
    cc = connected_components(len(run.records), edges)
    cm = constrained_merge(run.records, edges, ew)
    return {
        "pairwise_on_candidates": pairwise,
        "entities_transitive_closure": {**pair_prf(cluster_pairs(cc), tp), **cluster_stats(cc)},
        "entities_constrained_merge": {**pair_prf(cluster_pairs(cm), tp), **cluster_stats(cm)},
    }


def calibration(post: np.ndarray, truth: np.ndarray, bins: int = 10) -> dict:
    edges = np.linspace(0, 1, bins + 1)
    ece, rows = 0.0, []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (post >= lo) & (post < hi if hi < 1 else post <= hi)
        if sel.sum() == 0:
            continue
        conf, acc = float(post[sel].mean()), float(truth[sel].mean())
        ece += sel.sum() / len(post) * abs(conf - acc)
        rows.append({"bin": f"{lo:.1f}-{hi:.1f}", "n": int(sel.sum()),
                     "mean_posterior": round(conf, 4), "observed_match_rate": round(acc, 4)})
    return {"ece": round(ece, 4), "bins": rows}
