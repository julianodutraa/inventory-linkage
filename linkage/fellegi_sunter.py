"""Fellegi Sunter record linkage as a latent class model fitted by EM.

Each candidate pair carries a discrete comparison vector gamma, one level per
field, with -1 meaning "missing on at least one side" (marginalised out).
The classic model (Fellegi and Sunter 1969; Winkler 1988; Jaro 1989) is a
two class mixture with conditional independence inside each class:

    P(gamma) = p * prod_f m_f[gamma_f] + (1 - p) * prod_f u_f[gamma_f]

Nothing in the likelihood says that the "m" class means "same entity". EM
finds whichever two group split explains the data best. In an infrastructure
inventory the dominant structure among candidate pairs is "same service
group" (siblings agree on team, env, kind, region and subnet), so the two
class fit happily learns that instead. ``fit_em`` therefore takes ``k``:
with k = 3 the mixture has room for match, sibling and unrelated pairs, and
the match class is identified afterwards as the class that agrees most often
on the identity bearing fields.

The match weight is the log2 likelihood ratio between the match class and
the mixture of all other classes, so it reduces to the textbook
sum_f log2(m_f / u_f) when k = 2.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class FSModel:
    pi: np.ndarray            # class priors, shape (k,)
    theta: list[np.ndarray]   # per field, shape (k, levels)
    match_class: int
    n_iter: int
    loglik: float

    @property
    def p(self) -> float:
        return float(self.pi[self.match_class])

    def _class_loglik(self, G: np.ndarray) -> np.ndarray:
        L = np.zeros((len(G), len(self.pi)))
        for f, th in enumerate(self.theta):
            g = G[:, f]
            ok = g >= 0
            L[ok] += np.log(th[:, g[ok]]).T
        return L

    def posterior(self, G: np.ndarray) -> np.ndarray:
        L = self._class_loglik(G) + np.log(self.pi)
        L -= L.max(axis=1, keepdims=True)
        R = np.exp(L)
        R /= R.sum(axis=1, keepdims=True)
        return R[:, self.match_class]

    def weights(self, G: np.ndarray) -> np.ndarray:
        """log2 P(gamma | match) / P(gamma | not match)."""
        L = self._class_loglik(G)
        c = self.match_class
        others = [j for j in range(len(self.pi)) if j != c]
        po = self.pi[others] / self.pi[others].sum()
        lo = L[:, others] + np.log(po)
        mx = lo.max(axis=1, keepdims=True)
        lnon = (mx + np.log(np.exp(lo - mx).sum(axis=1, keepdims=True)))[:, 0]
        return (L[:, c] - lnon) / np.log(2)

    def field_weights(self) -> list[np.ndarray]:
        """Per field log2(m/u) with u the prior weighted mix of non match classes."""
        c = self.match_class
        others = [j for j in range(len(self.pi)) if j != c]
        po = self.pi[others] / self.pi[others].sum()
        return [np.log2(th[c] / (po @ th[others])) for th in self.theta]


def fit_em(G: np.ndarray, n_levels: list[int], k: int = 2, id_fields: list[int] | None = None,
           max_iter: int = 1000, tol: float = 1e-10, seed: int = 0) -> FSModel:
    """Latent class EM. ``id_fields`` are the columns used to pick the match
    class after fitting (the class with the highest mean probability of top
    level agreement on those columns); defaults to all columns."""
    rng = np.random.default_rng(seed)
    n, F = G.shape
    eps = 1e-6
    onehots = []
    for f, L in enumerate(n_levels):
        oh = np.zeros((n, L))
        ok = G[:, f] >= 0
        oh[np.where(ok)[0], G[ok, f]] = 1.0
        onehots.append(oh)
    # Class c starts tilted towards agreement by an amount that grows with c.
    theta = []
    for L in n_levels:
        base = np.vstack([np.linspace(1, 1 + 3 * c / max(k - 1, 1), L) ** (1 + c) for c in range(k)])
        base = base * rng.uniform(0.9, 1.1, size=base.shape)
        theta.append(base / base.sum(axis=1, keepdims=True))
    pi = np.full(k, 1.0 / k)
    prev = -np.inf
    for it in range(1, max_iter + 1):
        L = np.tile(np.log(pi), (n, 1))
        for f in range(F):
            L += onehots[f] @ np.log(theta[f]).T
        mx = L.max(axis=1, keepdims=True)
        ll = mx[:, 0] + np.log(np.exp(L - mx).sum(axis=1))
        R = np.exp(L - ll[:, None])
        loglik = float(ll.sum())
        pi = np.clip(R.mean(axis=0), eps, None)
        pi /= pi.sum()
        for f in range(F):
            C = R.T @ onehots[f] + eps
            theta[f] = C / C.sum(axis=1, keepdims=True)
        if loglik - prev < tol * abs(loglik):
            break
        prev = loglik
    ids = id_fields if id_fields is not None else list(range(F))
    agree = np.mean([theta[f][:, -1] for f in ids], axis=0)
    return FSModel(pi, theta, int(np.argmax(agree)), it, loglik)


def model_thresholds(post: np.ndarray, weights: np.ndarray, mu: float, lam: float) -> tuple[float, float]:
    """Fellegi Sunter decision thresholds from the model's own error estimates.

    Upper: lowest weight such that the model estimated false match rate among
    accepted pairs, mean(1 - posterior), stays at or below ``mu``.
    Lower: the weight below which the model estimated share of true matches
    left behind, sum(posterior below) / sum(posterior), stays at or below ``lam``.
    Accept w >= upper, reject w < lower, review in between.
    """
    order = np.argsort(-weights, kind="stable")
    w, q = weights[order], post[order]
    fdr = np.cumsum(1 - q) / np.arange(1, len(q) + 1)
    # Weights are discrete, so a cut can only fall at the end of a run of ties.
    group_end = np.append(w[1:] != w[:-1], True)
    ok = np.where((fdr <= mu) & group_end)[0]
    upper = float(w[ok.max()]) if len(ok) else float(w[0] + 1)
    below = np.cumsum(q[::-1]) / q.sum()       # mass in the bottom j+1 pairs
    wr = w[::-1]
    okl = np.where((below <= lam) & np.append(wr[1:] != wr[:-1], True))[0]
    lower = float(w[::-1][okl.max() + 1]) if len(okl) and okl.max() + 1 < len(w) else float(w[-1])
    return upper, min(lower, upper)
