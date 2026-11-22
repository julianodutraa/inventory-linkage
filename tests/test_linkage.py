import numpy as np
import pytest

from linkage import pipeline as P, review as R
from linkage.cluster import connected_components, constrained_merge, pair_prf
from linkage.compare import compare, host_tokens, jaro_winkler, team_level
from linkage.fellegi_sunter import fit_em, model_thresholds
from linkage.generate import Record, generate


@pytest.fixture(scope="module")
def fleet():
    return generate(20261122)[0]


@pytest.fixture(scope="module")
def run3(fleet):
    return P.fit(fleet, k=3)


@pytest.fixture(scope="module")
def run2(fleet):
    return P.fit(fleet, k=2)


def test_jaro_winkler_reference_values():
    # Winkler (1990) textbook examples.
    assert jaro_winkler("martha", "marhta") == pytest.approx(0.9611, abs=1e-4)
    assert jaro_winkler("dwayne", "duane") == pytest.approx(0.84, abs=1e-4)
    assert jaro_winkler("dixon", "dicksonx") == pytest.approx(0.8133, abs=1e-4)
    assert jaro_winkler("abc", "abc") == 1.0
    assert jaro_winkler("", "abc") == 0.0


def test_host_tokens_normalises_fqdn_underscores_and_padding():
    assert host_tokens("PG_Orders_prod_2.prod.internal") == ("pg-orders-prod", (2,))
    assert host_tokens("pg-orders-prod-02") == ("pg-orders-prod", (2,))


def test_ip_scrape_target_is_not_an_instance_number():
    # Regression: the digits of 10.20.131.188:9100 used to be read as instance
    # numbers, so an exact IP match was scored as a hard name disagreement.
    assert host_tokens("10.20.131.188:9100") == ("", ())
    a = Record("a", "cloud", "k8s-invoices-stg-02", "10.20.131.188", "stg", None, "k8s-node", None)
    b = Record("b", "monitor", "10.20.131.188:9100", "10.20.131.188", "stg", None, None, None)
    g = compare(a, b)
    assert g[0] == -1 and g[1] == -1 and g[2] == 2


def test_team_levels():
    assert team_level("data-platform", "Data Platform") == 2
    assert team_level("data-platform", "dataplat") == 1
    assert team_level("sre-core", "Payments Engineering") == 0
    assert team_level(None, "sre") == -1


def test_generator_is_deterministic_and_does_not_leak(fleet):
    again = generate(20261122)[0]
    assert [r.public() for r in fleet] == [r.public() for r in again]
    assert all("entity" not in r.public() for r in fleet)
    # Every source lists each entity at most once.
    seen = set()
    for r in fleet:
        assert (r.source, r.entity) not in seen
        seen.add((r.source, r.entity))


def test_em_recovers_parameters_from_its_own_model():
    rng = np.random.default_rng(7)
    n, p = 40000, 0.1
    m = [np.array([0.05, 0.15, 0.8]), np.array([0.1, 0.9]), np.array([0.2, 0.8]),
         np.array([0.1, 0.9]), np.array([0.05, 0.25, 0.7])]
    u = [np.array([0.7, 0.25, 0.05]), np.array([0.85, 0.15]), np.array([0.6, 0.4]),
         np.array([0.7, 0.3]), np.array([0.8, 0.15, 0.05])]
    z = rng.random(n) < p
    G = np.column_stack([np.where(z, rng.choice(len(a), n, p=a), rng.choice(len(b), n, p=b))
                         for a, b in zip(m, u)]).astype(np.int8)
    model = fit_em(G, [3, 2, 2, 2, 3], k=2)
    assert model.p == pytest.approx(p, abs=0.01)
    c = model.match_class
    for f in range(5):
        assert np.allclose(model.theta[f][c], m[f], atol=0.03)
        assert np.allclose(model.theta[f][1 - c], u[f], atol=0.03)


def test_missing_fields_contribute_zero_weight(run3):
    G = np.array([[-1, -1, -1, -1, -1, -1, -1]], dtype=np.int8)
    assert run3.model.weights(G)[0] == pytest.approx(0.0, abs=1e-9)


def test_weights_match_textbook_sum_for_two_classes(run2):
    fw = run2.model.field_weights()
    k = 123
    manual = sum(fw[f][g] for f, g in enumerate(run2.G[k]) if g >= 0)
    assert run2.model.weights(run2.G[k:k + 1])[0] == pytest.approx(manual, abs=1e-9)


def test_thresholds_ordered_and_respect_model_fdr(run3):
    assert run3.lower <= run3.upper
    acc = run3.weights >= run3.upper
    assert (1 - run3.post[acc]).mean() <= 0.01 + 1e-9


def test_threshold_helper_on_toy_scores():
    w = np.array([10.0, 8, 6, 4, 2, 0])
    q = np.array([0.999, 0.99, 0.9, 0.5, 0.05, 0.001])
    upper, lower = model_thresholds(q, w, mu=0.01, lam=0.05)
    assert upper == 8.0
    assert lower <= upper


def test_constrained_merge_never_joins_two_records_of_one_source():
    recs = [Record(str(i), s, "x", None, None, None, None, None) for i, s in
            enumerate(["cloud", "cmdb", "cloud", "monitor"])]
    edges = [(0, 1), (1, 2), (2, 3)]
    labels = constrained_merge(recs, edges, np.array([3.0, 2.0, 1.0]))
    assert labels[0] == labels[1] and labels[2] == labels[3] and labels[0] != labels[2]
    cc = connected_components(4, edges)
    assert len(set(cc)) == 1  # closure chains everything together


def test_pair_prf():
    m = pair_prf({(0, 1), (2, 3)}, {(0, 1), (4, 5)})
    assert (m["precision"], m["recall"], m["f1"]) == (0.5, 0.5, 0.5)


def test_two_class_model_learns_service_groups_not_assets(run2):
    """The headline failure: the model promises ~1% false links and delivers far more."""
    acc = run2.weights >= run2.upper
    promised = (1 - run2.post[acc]).mean()
    delivered = 1 - run2.is_match[acc].mean()
    assert promised < 0.02
    assert delivered > 0.5


def test_three_classes_fix_it(run2, run3):
    acc = run3.weights >= run3.upper
    assert 1 - run3.is_match[acc].mean() < 0.02
    f2 = P.score(run2, P.decide(run2, cut="half"))["entities_constrained_merge"]["f1"]
    f3 = P.score(run3, P.decide(run3, cut="half"))["entities_constrained_merge"]["f1"]
    assert f3 > f2 + 0.03
    assert f3 > 0.97


def test_constraint_beats_transitive_closure(run3):
    s = P.score(run3, P.decide(run3, cut="half"))
    cc, cm = s["entities_transitive_closure"], s["entities_constrained_merge"]
    assert cm["precision"] > cc["precision"]
    assert cm["largest"] <= 3 < cc["largest"]


def test_review_export_is_blind(run3):
    ks = R.queue(run3)
    items = R.export(run3, ks[:5])
    for it in items:
        assert set(it) == {"id", "A", "B"}
        for side in ("A", "B"):
            assert set(it[side]) == {"source", "name", "ip", "env", "team", "kind", "region"}


def test_oracle_review_never_hurts(run3):
    ks = R.queue(run3)
    oracle = {int(k): bool(run3.is_match[k]) for k in ks}
    base = P.score(run3, P.decide(run3, cut="half"))["pairwise_on_candidates"]["f1"]
    rev = P.score(run3, P.decide(run3, review=oracle, cut="half"))["pairwise_on_candidates"]["f1"]
    assert rev >= base
