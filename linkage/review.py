"""Clerical review of the uncertain band, by an LLM or by a person.

The queue is every candidate pair whose match posterior falls in
[lo, hi). Each item is exported with the public record fields only (no
entity id, no posterior, no comparison vector) so that a reviewer, human
or model, judges the two records cold.
"""
from __future__ import annotations

import hashlib
import json

import numpy as np

from .pipeline import Run

REVIEW_INSTRUCTIONS = """You are reconciling an infrastructure asset inventory.
Three systems describe the same fleet: `cloud` (provider inventory), `monitor`
(monitoring scrape targets, sometimes registered as a bare ip:port) and `cmdb`
(hand maintained, may contain typos, underscores, upper case, long form team
and environment names, stale IP addresses and other human errors). Each system
lists a given asset at most once. Services are often deployed as several
numbered sibling instances (for example -01, -02) that share team,
environment, kind, region and subnet. For each pair, decide whether record A
and record B describe the SAME physical asset. Answer with a JSON list of
objects {"id": ..., "same": true|false, "confidence": 0..1, "reason": "..."}."""


def queue(run: Run, lo: float = 0.02, hi: float = 0.98) -> np.ndarray:
    return np.where((run.post >= lo) & (run.post < hi))[0]


def pair_id(run: Run, k: int) -> str:
    i, j = run.pairs[k]
    return hashlib.sha1(f"{run.records[i].rid}|{run.records[j].rid}".encode()).hexdigest()[:10]


def export(run: Run, ks: np.ndarray) -> list[dict]:
    items = []
    for k in ks:
        i, j = run.pairs[k]
        a, b = run.records[i].public(), run.records[j].public()
        for d in (a, b):
            d.pop("rid")
        items.append({"id": pair_id(run, k), "A": a, "B": b})
    return items


def prompt_sha(items: list[dict]) -> str:
    blob = REVIEW_INSTRUCTIONS + json.dumps(items, sort_keys=True)
    return hashlib.sha256(blob.encode()).hexdigest()


def evaluate(run: Run, ks: np.ndarray, judgments: dict[str, dict]) -> dict:
    """Compare model decisions, reviewer decisions and simple combinations on the queue."""
    truth = run.is_match[ks]
    model = run.post[ks] >= 0.5
    ids = [pair_id(run, k) for k in ks]
    missing = [x for x in ids if x not in judgments]
    if missing:
        raise ValueError(f"{len(missing)} queue items have no judgment")
    llm = np.array([bool(judgments[x]["same"]) for x in ids])
    conf = np.array([float(judgments[x].get("confidence", 0.5)) for x in ids])

    def acc(pred):
        return round(float((pred == truth).mean()), 4)

    # Log odds fusion: treat the reviewer as one more piece of evidence.
    lo = np.log(np.clip(run.post[ks], 1e-6, 1 - 1e-6) / np.clip(1 - run.post[ks], 1e-6, 1))
    c = np.clip(conf, 0.5, 0.99)
    lr = np.where(llm, np.log(c / (1 - c)), -np.log(c / (1 - c)))
    fused = (lo + lr) >= 0
    return {
        "n": int(len(ks)), "true_matches": int(truth.sum()),
        "accuracy_model": acc(model), "accuracy_llm": acc(llm), "accuracy_fused": acc(fused),
        "llm_disagrees_with_model": int((llm != model).sum()),
        "when_they_disagree_llm_right": int(((llm != model) & (llm == truth)).sum()),
        "llm_false_merges": int((llm & ~truth).sum()), "llm_false_splits": int((~llm & truth).sum()),
        "model_false_merges": int((model & ~truth).sum()), "model_false_splits": int((~model & truth).sum()),
        "decisions": {"model": model, "llm": llm, "fused": fused},
    }
