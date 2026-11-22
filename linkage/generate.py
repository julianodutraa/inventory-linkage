"""Synthetic infrastructure inventories with known ground truth.

Three systems describe the same fleet of assets (VMs, database instances,
Kafka brokers, Kubernetes nodes) and disagree in realistic ways:

* ``cloud``   the cloud provider inventory: canonical names, current IPs,
              team tags sometimes missing.
* ``monitor`` the monitoring scrape targets: fully qualified names (or a
              bare ip:port target), team labels abbreviated, a fraction of
              IPs stale after re-IP work.
* ``cmdb``    the hand maintained CMDB: typos, underscores, dropped zero
              padding, the occasional wrong instance number, long form team
              and environment names, stale IPs.

The hard part is not the noise, it is the siblings. Every service is deployed
as one to four numbered instances that share team, environment, kind, region
and subnet. Two siblings agree on almost every attribute and differ only in a
trailing digit, which is exactly the situation a generic record linkage model
is worst at.

Record ids are random hex strings so that nothing in a record leaks the
entity it belongs to.
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict

KINDS = {"vm": "vm", "pg": "postgres", "ora": "oracle", "kfk": "kafka", "k8s": "k8s-node"}
SERVICES = [
    "orders", "billing", "ingest", "telemetry", "catalog", "search", "auth",
    "ledger", "events", "reports", "metrics", "features", "payments", "notify",
    "invoices", "routing", "pricing", "audit", "sessions", "warehouse",
]
TEAMS = {
    "data-platform": ("dataplat", "Data Platform"),
    "sre-core": ("sre", "SRE Core"),
    "payments-eng": ("pay", "Payments Engineering"),
    "observability": ("o11y", "Observability"),
    "search-infra": ("search", "Search Infrastructure"),
}
ENVS = {"prod": ("prod", "production"), "stg": ("stg", "staging"), "dev": ("dev", "development")}
REGIONS = ["us-east-1", "eu-west-1", "sa-east-1"]
SOURCES = ("cloud", "monitor", "cmdb")


@dataclass
class Record:
    rid: str
    source: str
    name: str
    ip: str | None
    env: str | None
    team: str | None
    kind: str | None
    region: str | None
    entity: int = field(repr=False, default=-1)  # ground truth, never shown to models

    def public(self) -> dict:
        d = asdict(self)
        d.pop("entity")
        return d


def _typo(s: str, rng: random.Random) -> str:
    letters = [i for i, c in enumerate(s) if c.isalpha()]
    if len(letters) < 4:
        return s
    i = rng.choice(letters[1:-1])
    op = rng.random()
    if op < 0.4 and i + 1 < len(s) and s[i + 1].isalpha():  # transposition
        return s[:i] + s[i + 1] + s[i] + s[i + 2:]
    if op < 0.7:  # deletion
        return s[:i] + s[i + 1:]
    return s[:i] + rng.choice("aeiorstn") + s[i + 1:]  # substitution


def generate(seed: int = 20261122, n_services: int = 150) -> tuple[list[Record], dict]:
    """Return shuffled records from all sources plus generation metadata."""
    rng = random.Random(seed)
    entities = []
    subnet_counter = 0
    used = set()
    while len({(e["kind"], e["service"], e["env"]) for e in entities}) < n_services:
        kind = rng.choice(list(KINDS))
        service = rng.choice(SERVICES)
        env = rng.choices(list(ENVS), weights=[0.5, 0.3, 0.2])[0]
        if (kind, service, env) in used:
            continue
        used.add((kind, service, env))
        team = rng.choice(list(TEAMS))
        region = rng.choice(REGIONS)
        subnet_counter += 1
        env_oct = {"prod": 10, "stg": 20, "dev": 30}[env]
        n_inst = rng.choices([1, 2, 3, 4], weights=[0.3, 0.35, 0.2, 0.15])[0]
        base_host = rng.randint(10, 200)
        for idx in range(1, n_inst + 1):
            entities.append(dict(
                kind=kind, service=service, env=env, team=team, region=region, idx=idx,
                ip=f"10.{env_oct}.{subnet_counter % 250}.{base_host + idx}",
                stale_ip=f"10.{env_oct}.{(subnet_counter + 97) % 250}.{rng.randint(10, 250)}",
            ))

    records: list[Record] = []

    def rid() -> str:
        return "%012x" % rng.getrandbits(48)

    for eid, e in enumerate(entities):
        host = f"{e['kind']}-{e['service']}-{e['env']}-{e['idx']:02d}"
        short_team, long_team = TEAMS[e["team"]]
        if rng.random() < 0.9:
            records.append(Record(
                rid(), "cloud", host, e["ip"], e["env"],
                e["team"] if rng.random() > 0.2 else None,
                KINDS[e["kind"]], e["region"], eid))
        if rng.random() < 0.85:
            ip = e["stale_ip"] if rng.random() < 0.15 else e["ip"]
            # Some scrape targets are registered by address only.
            mname = f"{ip}:9100" if rng.random() < 0.12 else f"{host}.{e['env']}.internal"
            records.append(Record(
                rid(), "monitor", mname, ip, e["env"],
                short_team if rng.random() > 0.1 else None,
                KINDS[e["kind"]] if rng.random() > 0.1 else None, None, eid))
        if rng.random() < 0.8:
            idx = e["idx"]
            if rng.random() < 0.06:  # human error: the wrong instance number
                idx = idx + 1 if rng.random() < 0.5 or idx == 1 else idx - 1
            name = f"{e['kind']}-{e['service']}-{e['env']}-{idx:02d}"
            if rng.random() < 0.3:
                name = name.replace("-", "_")
            if rng.random() < 0.3:
                name = f"{e['kind']}-{e['service']}-{e['env']}-{idx}".replace(
                    "-", "_" if "_" in name else "-")
            if rng.random() < 0.15:
                name = _typo(name, rng)
            if rng.random() < 0.2:
                name = name.upper()
            ip = e["stale_ip"] if rng.random() < 0.3 else e["ip"]
            if rng.random() < 0.05:
                ip = None
            records.append(Record(
                rid(), "cmdb", name, ip, ENVS[e["env"]][1],
                long_team if rng.random() > 0.05 else None,
                KINDS[e["kind"]], e["region"] if rng.random() > 0.3 else None, eid))
    rng.shuffle(records)
    meta = {"seed": seed, "n_entities": len(entities), "n_records": len(records),
            "n_services": n_services}
    return records, meta
