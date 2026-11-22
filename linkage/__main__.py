"""python -m linkage : run the pipeline on the synthetic fleet and explain one link."""
from __future__ import annotations

import argparse

import numpy as np

from . import pipeline as P
from .cluster import constrained_merge
from .compare import FIELDS
from .generate import generate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seed", type=int, default=20261122)
    ap.add_argument("--classes", type=int, default=3)
    args = ap.parse_args()

    recs, meta = generate(args.seed)
    run = P.fit(recs, k=args.classes)
    link = P.decide(run, cut="half")
    s = P.score(run, link)
    print(f"records {meta['n_records']}  candidate pairs {len(run.pairs)}  "
          f"latent classes {args.classes}  EM iterations {run.model.n_iter}")
    for key, label in (("pairwise_on_candidates", "pairwise"),
                       ("entities_transitive_closure", "entities, closure"),
                       ("entities_constrained_merge", "entities, constrained")):
        m = s[key]
        print(f"{label:<24} P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}")

    edges = [run.pairs[k] for k in np.where(link)[0]]
    labels = constrained_merge(recs, edges, run.weights[link])
    # Show the three source entity whose weakest internal link is lowest:
    # the least obvious merge the system still made.
    best = None
    for k in np.where(link)[0]:
        i, j = run.pairs[k]
        if labels[i] != labels[j]:
            continue
        members = [x for x in range(len(recs)) if labels[x] == labels[i]]
        if len(members) == 3 and (best is None or run.weights[k] < best[0]):
            best = (run.weights[k], k, members)
    if best:
        w, k, members = best
        print("\nleast obvious three source entity")
        for x in members:
            r = recs[x]
            print(f"  {r.source:<8} {r.name:<36} {str(r.ip):<15} {str(r.env):<12} {str(r.team)}")
        fw = run.model.field_weights()
        parts = [f"{f}={fw[c][g]:+.1f}" for c, (f, g) in enumerate(zip(FIELDS, run.G[k])) if g >= 0]
        print(f"  weakest link: weight {w:.1f} bits, posterior {run.post[k]:.3f}")
        print("  " + "  ".join(parts))


if __name__ == "__main__":
    main()
