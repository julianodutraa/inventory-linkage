"""Normalisation, blocking and discrete comparison vectors."""
from __future__ import annotations

import re
from collections import defaultdict
from itertools import combinations

import numpy as np

from .generate import Record

ENV_MAP = {"production": "prod", "staging": "stg", "development": "dev"}
_TOKEN = re.compile(r"[a-z]+|\d+")
_IP_TARGET = re.compile(r"^\d{1,3}(\.\d{1,3}){3}(:\d+)?$")


def jaro(a: str, b: str) -> float:
    if a == b:
        return 1.0
    la, lb = len(a), len(b)
    if not la or not lb:
        return 0.0
    window = max(la, lb) // 2 - 1
    ma, mb = [False] * la, [False] * lb
    matches = 0
    for i, ca in enumerate(a):
        lo, hi = max(0, i - window), min(i + window + 1, lb)
        for j in range(lo, hi):
            if not mb[j] and b[j] == ca:
                ma[i] = mb[j] = True
                matches += 1
                break
    if not matches:
        return 0.0
    t, k = 0, 0
    for i in range(la):
        if ma[i]:
            while not mb[k]:
                k += 1
            if a[i] != b[k]:
                t += 1
            k += 1
    t /= 2
    return (matches / la + matches / lb + (matches - t) / matches) / 3


def jaro_winkler(a: str, b: str, p: float = 0.1) -> float:
    j = jaro(a, b)
    prefix = 0
    for ca, cb in zip(a[:4], b[:4]):
        if ca != cb:
            break
        prefix += 1
    return j + prefix * p * (1 - j)


def host_tokens(name: str) -> tuple[str, tuple[int, ...]]:
    """Split a hostname into its alphabetic stem and its numeric tokens.

    ``PG_Orders_prod_2.prod.internal`` -> ("pg-orders-prod", (2,))
    """
    if _IP_TARGET.match(name):
        # A bare ip:port scrape target carries no hostname at all. Its digits
        # are an address, not an instance number, and must read as missing.
        return "", ()
    short = name.lower().split(".")[0].replace("_", "-")
    toks = _TOKEN.findall(short)
    alpha = "-".join(t for t in toks if t.isalpha())
    nums = tuple(int(t) for t in toks if t.isdigit())
    return alpha, nums


def norm_team(t: str | None) -> str | None:
    if t is None:
        return None
    t = t.lower().replace(" ", "-")
    return t


def team_level(a: str | None, b: str | None) -> int:
    a, b = norm_team(a), norm_team(b)
    if a is None or b is None:
        return -1
    if a == b:
        return 2
    ta, tb = set(a.split("-")), set(b.split("-"))
    initials_a = "".join(w[0] for w in a.split("-"))
    initials_b = "".join(w[0] for w in b.split("-"))
    if (ta & tb or a.startswith(b[:4]) or b.startswith(a[:4])
            or initials_a in b or initials_b in a):
        return 1
    return 0


FIELDS = ["name_alpha", "name_num", "ip", "env", "team", "kind", "region"]
N_LEVELS = [4, 2, 3, 2, 3, 2, 2]


def compare(r1: Record, r2: Record) -> list[int]:
    a1, n1 = host_tokens(r1.name)
    a2, n2 = host_tokens(r2.name)
    if not a1 or not a2:
        name_alpha = -1
    else:
        jw = jaro_winkler(a1, a2)
        name_alpha = 3 if a1 == a2 else 2 if jw >= 0.94 else 1 if jw >= 0.85 else 0
    name_num = -1 if not n1 or not n2 else int(n1 == n2)
    if r1.ip is None or r2.ip is None:
        ip = -1
    elif r1.ip == r2.ip:
        ip = 2
    else:
        ip = int(r1.ip.rsplit(".", 1)[0] == r2.ip.rsplit(".", 1)[0])
    e1, e2 = (ENV_MAP.get(x, x) if x else None for x in (r1.env, r2.env))
    env = -1 if e1 is None or e2 is None else int(e1 == e2)
    team = team_level(r1.team, r2.team)
    kind = -1 if r1.kind is None or r2.kind is None else int(r1.kind == r2.kind)
    region = -1 if r1.region is None or r2.region is None else int(r1.region == r2.region)
    return [name_alpha, name_num, ip, env, team, kind, region]


def block(records: list[Record]) -> list[tuple[int, int]]:
    """Cross source candidate pairs that share a blocking key.

    Keys: the first four letters of the service token, the kind prefix plus
    the environment prefix, and the /24 subnet. Any one shared key is enough.
    """
    keys: dict[str, list[int]] = defaultdict(list)
    for i, r in enumerate(records):
        alpha, _ = host_tokens(r.name)
        toks = alpha.split("-")
        if len(toks) >= 2:
            keys["s:" + toks[1][:4]].append(i)
            keys["k:" + toks[0][:2] + toks[-1][:2]].append(i)
        if r.ip:
            keys["n:" + r.ip.rsplit(".", 1)[0]].append(i)
    pairs = set()
    for members in keys.values():
        for i, j in combinations(members, 2):
            if records[i].source != records[j].source:
                pairs.add((min(i, j), max(i, j)))
    return sorted(pairs)


def comparison_matrix(records: list[Record], pairs: list[tuple[int, int]]) -> np.ndarray:
    return np.array([compare(records[i], records[j]) for i, j in pairs], dtype=np.int8)
