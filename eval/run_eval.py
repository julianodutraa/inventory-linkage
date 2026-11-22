"""Reproduce every number in the README: python eval/run_eval.py"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from linkage import pipeline as P, review as R  # noqa: E402
from linkage.cluster import true_pairs  # noqa: E402
from linkage.compare import FIELDS  # noqa: E402
from linkage.generate import generate  # noqa: E402

MAIN_SEED = 20261122
NO_NUM = [f for f in FIELDS if f != "name_num"]
VARIANTS = {
    "two_class_fs": dict(k=2),
    "three_class_without_instance_number": dict(k=3, fields=NO_NUM),
    "three_class_full": dict(k=3),
}


def summarize(run: P.Run) -> dict:
    link = P.decide(run, cut="half")
    accepted = run.weights >= run.upper
    s = P.score(run, link)
    return {
        "match_class_prior": round(run.model.p, 4),
        "em_iterations": run.model.n_iter,
        "ece": P.calibration(run.post, run.is_match)["ece"],
        "fs_upper_threshold": round(run.upper, 3),
        "fdr_promised_at_upper": round(float((1 - run.post[accepted]).mean()), 4),
        "fdr_delivered_at_upper": round(float(1 - run.is_match[accepted].mean()), 4),
        **s,
    }


def main() -> None:
    t0 = time.time()
    recs, meta = generate(MAIN_SEED)
    out: dict = {"data": meta}

    base = P.fit(recs, k=3)
    tp = true_pairs(recs)
    n = len(recs)
    out["blocking"] = {
        "candidate_pairs": len(base.pairs),
        "all_cross_source_pairs": int(sum(1 for i in range(n) for j in range(i + 1, n)
                                          if recs[i].source != recs[j].source)),
        "true_pairs": len(tp),
        "pair_completeness": round(len(tp & set(base.pairs)) / len(tp), 4),
    }
    out["variants"] = {name: summarize(P.fit(recs, **kw)) for name, kw in VARIANTS.items()}
    out["field_weights_three_class"] = {f: np.round(w, 2).tolist() for f, w in
                                        zip(FIELDS, base.model.field_weights())}
    out["calibration_three_class"] = P.calibration(base.post, base.is_match)
    out["calibration_two_class"] = P.calibration(P.fit(recs, k=2).post, base.is_match)

    seeds = list(range(1, 11))
    rob = {}
    for name in ("two_class_fs", "three_class_full"):
        f1 = [P.score(r, P.decide(r, cut="half"))["entities_constrained_merge"]["f1"]
              for r in (P.fit(generate(s)[0], **VARIANTS[name]) for s in seeds)]
        rob[name] = {"seeds": seeds, "entity_f1_constrained": f1,
                     "mean": round(float(np.mean(f1)), 4), "min": round(float(np.min(f1)), 4)}
    out["robustness"] = rob

    ks = R.queue(base)
    items = R.export(base, ks)
    out["review_queue"] = {"size": len(ks), "prompt_sha256": R.prompt_sha(items),
                           "always_same_accuracy": round(float(base.is_match[ks].mean()), 4),
                           "true_matches_outside_queue_below": int(base.is_match[base.post < 0.02].sum())}
    (ROOT / "eval" / "review_queue.json").write_text(
        json.dumps({"instructions": R.REVIEW_INSTRUCTIONS, "items": items}, indent=1))
    jpath = ROOT / "eval" / "llm_judgments.json"
    if jpath.exists():
        j = json.loads(jpath.read_text())
        assert j["provenance"]["prompt_sha256"] == out["review_queue"]["prompt_sha256"], \
            "judgments were made on a different queue"
        ev = R.evaluate(base, ks, j["judgments"])
        decisions = ev.pop("decisions")
        ev["end_to_end"] = {}
        for who in ("model", "llm", "fused"):
            override = {int(k): bool(v) for k, v in zip(ks, decisions[who])}
            s = P.score(base, P.decide(base, review=override, cut="half"))
            ev["end_to_end"][who] = {
                "pairwise_f1": s["pairwise_on_candidates"]["f1"],
                "entity_f1_closure": s["entities_transitive_closure"]["f1"],
                "entity_f1_constrained": s["entities_constrained_merge"]["f1"],
                "entity_fp_constrained": s["entities_constrained_merge"]["fp"],
                "entity_fn_constrained": s["entities_constrained_merge"]["fn"],
            }
        oracle = {int(k): bool(base.is_match[k]) for k in ks}
        s = P.score(base, P.decide(base, review=oracle, cut="half"))
        ev["end_to_end"]["oracle"] = {"entity_f1_constrained": s["entities_constrained_merge"]["f1"],
                                      "entity_f1_closure": s["entities_transitive_closure"]["f1"]}
        out["llm_review"] = {"provenance": j["provenance"], **ev}
    out["runtime_seconds"] = round(time.time() - t0, 1)
    (ROOT / "eval" / "results.json").write_text(json.dumps(out, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k not in ("calibration_three_class",
                                                               "calibration_two_class")}, indent=1))


if __name__ == "__main__":
    main()
