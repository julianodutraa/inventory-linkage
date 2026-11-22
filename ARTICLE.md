# The model promised 1% and delivered 64%

*Reconciling three infrastructure inventories with Fellegi Sunter, EM and a blind LLM reviewer, and what each of them is actually worth.*

Ask three systems how many databases you run in production and you will get three answers. The cloud provider inventory knows what is billed. The monitoring system knows what it scrapes. The CMDB knows what someone typed into it during a change window. None of them is wrong in isolation, and joining them is one of those problems that every platform team solves once by hand, then again by script, and then quietly stops trusting.

I wanted to see how far the oldest serious answer to this problem goes, and where an LLM fits on top of it. The answer turned out to be more interesting than "the LLM helps".

## The problem is not the noise

Record linkage on hostnames looks like a string matching problem. `PG_Orders_prod_2` in the CMDB, `pg-orders-prod-02` in the cloud console, `pg-orders-prod-02.prod.internal` in monitoring, maybe `10.10.90.156:9100` if the scrape target was registered by address. Normalise case, underscores, domain suffixes and zero padding, add a Jaro Winkler comparator for typos, and most of it falls into place.

The hard part is the siblings. Services are deployed as numbered instances that share everything except the trailing digit: same team, same environment, same kind, same region, same subnet. Two siblings agree on almost every attribute a linkage model looks at. Any approach that scores records by how much they agree will love them.

To study this honestly I built a synthetic fleet with ground truth: 341 assets across three sources, 868 records, with typos, dropped padding, long form team names, stale IPs after re-IP work, address only scrape targets and the occasional CMDB entry with the wrong instance number. Every number below comes from that fleet and is reproducible with one command.

## Fellegi Sunter, the way the textbook says

The Fellegi Sunter model (1969) is the backbone of census and administrative record linkage. For each candidate pair you compute a discrete comparison vector: does the name stem agree exactly, approximately, or not at all, does the instance number agree, is the IP identical or only in the same /24, and so on. You then assume two populations of pairs, matches and non matches, with fields independent inside each, and score each pair by the log likelihood ratio between them. Winkler showed in 1988 that EM can estimate all the parameters from unlabelled pairs. No training data, a principled weight per field, a posterior per pair, and two thresholds with explicit error rates. It is a beautiful piece of statistics.

I ran it, set the upper threshold so that the model's own estimate of false links among accepted pairs was 1%, and looked at what it actually accepted. 64% of those links were wrong.

The expected calibration error was 0.044. That number looks fine, and it is the reason this failure survives in production systems. More than 97% of candidate pairs are trivial negatives, they are calibrated perfectly, and they dominate the average. The disaster lives entirely in the 3% that matter.

## What EM actually learned

Nothing in the likelihood says that the "match" class means "same asset". EM finds the two group split that best explains the data, and the strongest structure among candidate pairs is not identity, it is the service group. The model learned a match class covering 6.9% of pairs when the true rate is 2.5%. It gave the instance number field a weight close to zero, because inside its "match" class siblings disagree on that digit all the time. It then accepted `vm-audit-prod-01` and `vm-audit-prod-02` as the same machine with complete confidence.

This is the conditional independence assumption failing in a very specific way. Given that two records are not the same asset, their team, environment, kind, region and subnet are not independent at all: siblings agree on all of them at once. A two component mixture has no way to represent a population of "related but distinct" pairs, so it folds them into the match class.

The fix is to give the mixture somewhere to put them. With three latent classes, EM separates unrelated pairs, sibling pairs and matches, and the match class can be identified afterwards as the one most likely to agree on name, instance number and IP. Same data, same fields, no labels. The delivered false link rate at the threshold drops to 0.58% against 0.95% promised, and entity level F1 goes from 0.915 to 0.977 on the reference fleet, and from 0.915 to 0.984 on average over ten more fleets. Remove the instance number field and the three class model collapses back to the two class behaviour, which confirms what the third class is made of.

## The one line that beats every heuristic

Turning pairwise links into entities is where most pipelines reach for transitive closure: if A links to B and B links to C, they are one asset. With the textbook model that produced clusters of 14 records. Even with three classes the largest closure cluster had 7, although no asset can appear more than three times.

Each source lists an asset at most once. Encoding that as a constraint during a greedy merge in descending weight order (refuse any union that would put two cloud records into one entity) lifted precision at the entity level to 0.997. It is one condition in a union find loop, and it is the most valuable line in the project.

## Then the LLM

The obvious next step is clerical review. Fellegi Sunter has a review band built in, and an LLM is a tireless clerk. I exported the 224 pairs with posterior between 0.02 and 0.98, stripped everything except the public record fields, and had a Claude Sonnet model judge them blind in two independent sessions, with the same domain description an engineer would give a new colleague: three sources, typos in the CMDB, stale IPs, numbered siblings.

On the queue the LLM was right 96.4% of the time, against 94.2% for the model. It disagreed with the model on 9 pairs and was right on 7. That sounds like a clear win until you add the baseline that should always be in the table: answering "same" to every pair scores 92.9%, because the queue is mostly true matches.

End to end, with transitive closure, the LLM lifted entity F1 from 0.956 to 0.970. With the constrained merge it moved F1 from 0.9772 to 0.9743, slightly down. Every one of the 7 false merges it caught had already been refused by the constraint, and its 2 false splits cost recall. A log odds fusion of model and reviewer recovered the loss but did not improve on the model alone. Even a perfect reviewer, replacing every decision in the queue with ground truth, only reaches 0.9799.

## Where the remaining errors live

The ceiling is low because the remaining errors are not uncertain. 23 true matches sit below posterior 0.02. All of them have disagreeing instance numbers, typically a CMDB entry with the wrong digit. In 18 of them the IP matches exactly.

Under conditional independence the model adds a −17.4 bit penalty for the instance number to a +10.1 bit reward for the exact IP and concludes "different asset". But siblings never share an IP. The combination "exact IP, different instance number" is close to proof of a match, and it is an interaction that a model with independent fields cannot express. The model is confidently wrong, so a queue built from its own uncertainty never routes these pairs to anyone, human or LLM.

That is the most transferable lesson here. An LLM in the loop inherits the blind spots of whatever decides what it gets to see.

## What I would do in production

Fit the latent class model with enough classes to represent the real structure of the fleet, and check delivered against promised error at the threshold on a small audited sample instead of trusting ECE. Encode every hard structural fact you have as a constraint in the merge, starting with one record per source per asset. Add interaction fields where the domain tells you two signals are jointly decisive, such as exact IP with conflicting names. Use the LLM where it is strong: writing the explanation of each merge for the person approving a change, and auditing a random sample outside the queue, not deciding the queue the model already handles.

The code, the synthetic fleet, the cached blind judgments with their prompt hash and every number in this article are in the repository: https://github.com/julianodutraa/inventory-linkage
