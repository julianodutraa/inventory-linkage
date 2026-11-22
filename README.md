# inventory-linkage

Probabilistic record linkage for infrastructure inventories. Three systems that each claim to know what runs in production (the cloud provider inventory, the monitoring scrape targets and the CMDB) are reconciled into one list of assets, without a single labelled pair, using a Fellegi Sunter model fitted by EM. A blind LLM reviewer handles the uncertain band, and the evaluation measures honestly how much that reviewer is worth.

## Executive summary

Every infrastructure organisation runs on at least three inventories that disagree. Finance reconciles cloud spend against the provider inventory, on-call trusts what monitoring scrapes, and audit and change management trust the CMDB. When the three are not joined correctly, the consequences are concrete: hosts that nobody monitors because the CMDB entry never matched a scrape target, licences and capacity paid twice for the same database, and change tickets approved against the wrong instance. Joining them by hand does not scale past a few hundred assets, and joining them by exact hostname fails on the first underscore.

This project shows that the classical statistical answer to this problem works very well, but only after fixing one specific modelling mistake that is easy to make and invisible in the usual diagnostics. The textbook two class model, run on a realistic fleet, promises that 1 in 100 accepted links is wrong and delivers 64 in 100, because it learns "same service group" instead of "same asset". Adding a third latent class for sibling instances brings the delivered false link rate to 0.6% against 0.95% promised, and entity level F1 from 0.915 to 0.977 (0.984 on average across ten independently generated fleets).

The LLM reviewer is the part most people would expect to matter most, and it matters least. On the 224 pairs the model is unsure about it is right 96.4% of the time against 94.2% for the model, but answering "same" to everything already scores 92.9% on that queue. Once a one line structural rule (no entity can contain two records from the same source) is applied, the LLM review changes entity F1 from 0.9772 to 0.9743, a small loss, and even a perfect reviewer would only reach 0.9799. The remaining errors sit in pairs the model is confidently wrong about (mostly an exact IP match outvoted by a mistyped instance number), which no review queue built on the model's own uncertainty will ever surface. The recommendation is to spend the engineering budget on the model structure and the constraint, and to use the reviewer as an explanation layer, not as the decision maker.

## What it does

```
cloud inventory ─┐
monitor targets ─┼─> normalise ─> block ─> compare ─> EM (k latent classes) ─> posterior
CMDB ────────────┘                                                              │
                                                     review queue (0.02 ≤ p < 0.98) ─> blind LLM
                                                                                │
                                         constrained merge (one record per source per asset)
```

1. **Normalise** hostnames into an alphabetic stem and numeric instance tokens (`PG_Orders_prod_2.prod.internal` becomes `pg-orders-prod` plus `(2,)`), map long form environments and team names, and recognise bare `ip:port` scrape targets as having no hostname at all.
2. **Block** on service token, on kind plus environment, and on /24 subnet, keeping only cross source pairs. On the reference fleet this keeps 28,786 of 250,396 cross source pairs (an 88.5% reduction) and 97.3% of the true pairs. The 2.7% lost are mostly address only scrape targets whose IP is also stale: there is nothing left to block on.
3. **Compare** seven fields into discrete levels: name stem (exact, Jaro Winkler at least 0.94, at least 0.85, other), instance number (equal, different), IP (exact, same /24, other), environment, team (exact, partial, other), kind and region. Missing values are marginalised out.
4. **Fit** a latent class model by EM. With `k = 2` this is the classic Fellegi Sunter model. With `k = 3` the mixture has room for match, sibling and unrelated pairs; the match class is identified after fitting as the class most likely to agree fully on name, instance number and IP.
5. **Decide** with Fellegi Sunter thresholds derived from the model's own error estimates, or at posterior 0.5.
6. **Review** the uncertain band with a blind LLM, optionally fused with the model posterior in log odds.
7. **Cluster** with a constrained greedy merge: edges are added in descending weight and any union that would put two records of the same source into one asset is refused.

## Method in depth

Each candidate pair has a comparison vector γ. Under the latent class model with conditional independence inside each class,

```
P(γ) = Σ_c π_c Π_f θ_{c,f}[γ_f]
```

and the match weight of a pair is the log likelihood ratio between the match class M and the prior weighted mixture of all other classes,

```
w(γ) = log2 P(γ | M) − log2 Σ_{c≠M} (π_c / (1 − π_M)) P(γ | c)
```

which reduces to the textbook `Σ_f log2(m_f / u_f)` when k = 2. EM alternates class responsibilities and closed form updates of π and θ; it needs no labels. The Fellegi Sunter upper threshold is the lowest weight at which the model estimated false link rate among accepted pairs, the mean of (1 − posterior), stays at or below μ = 1%; the lower threshold bounds the model estimated share of true matches rejected at λ = 1%. Because weights are discrete, cuts are only placed at the end of a run of tied weights.

### Why two classes fail here

Nothing in the likelihood says that the "m" class means "same asset". EM finds the two group split that best explains the candidate pairs, and in an infrastructure fleet the dominant structure is the service group: sibling instances `-01`, `-02`, `-03` share team, environment, kind, region and subnet, and their names differ by one digit. Conditional on "not the same asset", these fields are strongly correlated, which is exactly what the two class model assumes away. It therefore learns a match class of about 6.9% of pairs (the true match rate among candidates is about 2.5%), gives the instance number field a weight near zero, and accepts siblings as matches with full confidence. Its expected calibration error is 0.044, which looks acceptable, because more than 97% of candidate pairs are easy negatives that dominate the average. The failure is only visible as the gap between promised and delivered false link rate at the decision threshold.

Removing the instance number field from the three class model collapses it back to the two class behaviour, which confirms that the third class is identified by exactly that field.

## Results

All numbers come from `python eval/run_eval.py` on seed 20261122 (341 assets, 868 records) unless stated otherwise. They are written to `eval/results.json`.

| Model | False links promised at FS upper threshold | False links delivered | ECE | Pairwise F1 | Entity F1, transitive closure | Entity F1, constrained merge |
|---|---|---|---|---|---|---|
| Two classes (textbook) | 0.93% | 63.9% | 0.044 | 0.529 | 0.414 | 0.915 |
| Three classes, no instance number | 0.46% | 63.6% | 0.044 | 0.529 | 0.420 | 0.891 |
| Three classes | 0.95% | 0.58% | 0.001 | 0.975 | 0.956 | 0.977 |

Across ten more fleets (seeds 1 to 10), entity F1 with constrained merge averages 0.915 for two classes (minimum 0.888) and 0.984 for three classes (minimum 0.973).

Transitive closure is the default in most linkage tooling and it is the wrong default here: with the two class model a few sibling links chain into clusters of up to 14 records, and even with three classes the largest closure cluster has 7 records although no asset can have more than 3. The constraint costs one line of code and is worth more than every other post processing step.

### LLM review of the uncertain band

The queue holds the 224 pairs with posterior between 0.02 and 0.98. Each pair was judged blind by a Claude Sonnet model in two independent sessions of 112 pairs, seeing only the review instructions in `linkage/review.py` and the public record fields (no posterior, no comparison vector, no entity id). Judgments are cached with provenance in `eval/llm_judgments.json`, and the evaluation refuses to run if the queue's prompt hash differs from the one the judgments were made on. One session returned the same generic reason string for 107 of its 112 decisions; the decisions are kept as returned.

| On the 224 queued pairs | Accuracy | False merges | False splits |
|---|---|---|---|
| Answer "same" for every pair | 92.9% | 16 | 0 |
| Model, posterior ≥ 0.5 | 94.2% | 13 | 0 |
| LLM reviewer | 96.4% | 6 | 2 |
| Log odds fusion of model and LLM | 97.3% | not separated | not separated |

| End to end | Pairwise F1 | Entity F1, closure | Entity F1, constrained |
|---|---|---|---|
| No review (model decides) | 0.9747 | 0.9561 | 0.9772 |
| LLM decides the queue | 0.9781 | 0.9704 | 0.9743 |
| Fusion decides the queue | 0.9795 | 0.9732 | 0.9772 |
| Ground truth decides the queue (ceiling) | | 0.9799 | 0.9799 |

The LLM disagreed with the model on 9 pairs and was right on 7. Its value is real under transitive closure, where every false merge it removes prevents a chain. Under the constrained merge it is not: all 7 false merges it caught were already refused by the constraint, and its two false splits cost recall. The ceiling shows the deeper limit. 23 true matches sit below posterior 0.02, all of them pairs whose instance numbers disagree, typically a CMDB entry with the wrong instance number. In 18 of the 23 the IP matches exactly. Conditional independence adds a −17.4 bit instance number penalty to a +10.1 bit exact IP agreement and the sum says no, although the combination "exact IP, different instance number" never occurs between siblings and is close to proof of a match. The model cannot express that interaction, it is confidently wrong, and a queue built from its own uncertainty never surfaces these pairs.

### Example output

```
$ python -m linkage
records 868  candidate pairs 28786  latent classes 3  EM iterations 179
pairwise                 P 0.982  R 0.968  F1 0.975
entities, closure        P 0.952  R 0.961  F1 0.956
entities, constrained    P 0.997  R 0.958  F1 0.977

least obvious three source entity
  cmdb     kfk-warehouso-prod-1                 10.10.121.20    production   Observability
  cloud    kfk-warehouse-prod-01                10.10.24.38     prod         None
  monitor  10.10.24.38:9100                     10.10.24.38     prod         o11y
  weakest link: weight 5.7 bits, posterior 0.565
  name_alpha=+3.8  name_num=+1.7  ip=-2.2  env=+0.6  kind=+0.3
```

The CMDB record has a typo, no zero padding and a stale IP; the monitoring record has no hostname at all. All three are the same broker, and the system links them correctly.

### A bug worth recording

The first version tokenised `10.20.131.188:9100` like any hostname and read its digits as instance numbers. Every address only scrape target therefore registered a hard instance number disagreement against its own cloud record, and pairs with an exact IP match came out with posterior 0.0. The fix treats bare `ip:port` targets as having no name at all, and a regression test pins it.

## Limitations

The fleet is synthetic. The noise processes (typos, padding, stale IPs, wrong instance numbers, address only targets) are modelled on real inventory drift but their rates are invented, and the absolute numbers should not be read as a forecast for any real estate. The team comparator is crude (`o11y` and `observability` only match through a first letter rule). Conditional independence still fails inside the match class (name stem and instance number come from the same string), so posteriors near the boundary are not perfectly calibrated. The LLM review is one model, one prompt, one run; its reasons were not audited pair by pair.

## Run it

```
python -m pip install -r requirements.txt
python -m pytest            # 17 tests, about 3 seconds
python -m linkage           # pipeline on the reference fleet
python eval/run_eval.py     # every number in this README, about 30 seconds
```

Python 3.10 or later and NumPy are the only requirements.

## Layout

```
linkage/generate.py        synthetic three source fleet with ground truth
linkage/compare.py         normalisation, Jaro Winkler, blocking, comparison levels
linkage/fellegi_sunter.py  latent class EM, match weights, FS thresholds
linkage/cluster.py         transitive closure, constrained merge, pairwise metrics
linkage/review.py          blind review queue export and evaluation
linkage/pipeline.py        end to end fit, decide, score, calibration
eval/run_eval.py           reproduces eval/results.json
eval/llm_judgments.json    cached blind LLM judgments with provenance
```

## References

Fellegi, I. P. and Sunter, A. B. (1969). A theory for record linkage. Journal of the American Statistical Association 64(328).
Winkler, W. E. (1988). Using the EM algorithm for weight computation in the Fellegi Sunter model of record linkage. Proceedings of the Section on Survey Research Methods, ASA.
Jaro, M. A. (1989). Advances in record linkage methodology as applied to matching the 1985 census of Tampa, Florida. JASA 84(406).
Winkler, W. E. (1990). String comparator metrics and enhanced decision rules in the Fellegi Sunter model of record linkage.
Enamorado, T., Fifield, B. and Imai, K. (2019). Using a probabilistic model to assist merging of large scale administrative records. American Political Science Review 113(2).

## License

MIT
