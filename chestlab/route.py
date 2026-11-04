"""Fastest chest-farming routes over the whole map.

A run = land next to the first chest, walk the route, open every chest on it, leave. The score of a route is how many chests
you expect to open per second: sum of value / (walking time + opening time). Value is a chest's chance of existing in a
match (100%-spawn chests, rare and sprite chests = 1; the other normal chests = P_OTHER; buried chests need digging).
Walking time is the straight line between chests at WALK_MPS (heights are not in the data, so floors are ignored).
    python tools/make_routes.py"""
import functools
import os

import cv2
import numpy as np

from .paths import BASE


MPU = 8.75                                           # metres per fortnite.gg map unit (0.0875 m per island px, 100 px per unit)
WALK_MPS = 6.5                                         # sprinting, with jumps and turns
OPEN_S = 1.2                                           # time to open one chest
OVERHEAD_S = 40.0                                      # per run: leave, rejoin, bus / drop, land (the more of it, the longer the best route)
P_OTHER = 0.5                                          # chance that a normal (<100% spawn) chest exists in a given match
SWIM_MPS = 3.0                                         # deep water: you swim (shallows are waded at walking speed)
DIG_S = 4.0                                            # a buried chest has to be dug up first
P_KEY = 0.5                                            # chance you have a key or a bomb for a vault (its 4 chests are behind a locked door)
VAULT_S = 0.75                                         # extra seconds per vault chest: the door
VALUE = dict(chest_sure=1.0, rare=1.0, sprite=1.0, vault=P_KEY, chest=P_OTHER, buried=0.8)


def metres(units):
    return np.asarray(units, float) * MPU


EXTRA = None          # optional per-chest extra seconds (aligned with P), e.g. digging; set by the caller
DEEP = None           # optional bool image of deep water at DEEP_PX_PER_M pixels per metre, see use_terrain()
DEEP_PX_PER_M = 16.0 / MPU


def use_terrain(on=True):
    """Legs through deep water cost swimming time. Deep water is read off the map tiles: dark saturated blue (the shallows
    are light turquoise and are waded)."""
    global DEEP
    _leg.cache_clear()
    if not on:
        DEEP = None
        return
    img = cv2.imread(os.path.join(BASE, "assets", "gg", "full_z4.png"))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    DEEP = (hsv[..., 0] >= 91) & (hsv[..., 0] <= 125) & (hsv[..., 1] > 170) & (hsv[..., 2] < 170)


@functools.lru_cache(maxsize=2_000_000)
def _leg(ax, ay, bx, by):
    """Seconds to get from (ax, ay) to (bx, by) metres in a straight line."""
    d = float(np.hypot(bx - ax, by - ay))
    if DEEP is None or d < 1.0:
        return d / WALK_MPS
    t = np.linspace(0.0, 1.0, max(2, int(d / 1.5)))
    x = np.clip(((ax + (bx - ax) * t) * DEEP_PX_PER_M).astype(int), 0, DEEP.shape[1] - 1)
    y = np.clip(((ay + (by - ay) * t) * DEEP_PX_PER_M).astype(int), 0, DEEP.shape[0] - 1)
    f = float(DEEP[y, x].mean())
    return d * ((1.0 - f) / WALK_MPS + f / SWIM_MPS)


def leg_seconds(P, a, b):
    return _leg(round(float(P[a][0]), 2), round(float(P[a][1]), 2), round(float(P[b][0]), 2), round(float(P[b][1]), 2))


def route_time(P, order):
    """Seconds for one whole run: overhead, walking `order` (indices into P, metres), opening every chest on it."""
    walk = sum(leg_seconds(P, a, b) for a, b in zip(order[:-1], order[1:]))
    extra = float(EXTRA[order].sum()) if EXTRA is not None else 0.0
    return OVERHEAD_S + walk + OPEN_S * len(order) + extra


def rate(P, v, order):
    return float(v[order].sum() / route_time(P, order))


def two_opt(P, order, fixed_start=False):
    """Shorten an open path by reversing segments until nothing improves (the path may start and end anywhere)."""
    order = list(order)
    n = len(order)
    d = lambda a, b: float(np.hypot(*(P[a] - P[b])))
    better = True
    while better and n > 3:
        better = False
        for i in range(0 if not fixed_start else 1, n - 1):
            for j in range(i + 1, n):
                a = order[i - 1] if i > 0 else None
                b, c = order[i], order[j]
                e = order[j + 1] if j + 1 < n else None
                old = (d(a, b) if a is not None else 0) + (d(c, e) if e is not None else 0)
                new = (d(a, c) if a is not None else 0) + (d(b, e) if e is not None else 0)
                if new < old - 1e-9:
                    order[i:j + 1] = order[i:j + 1][::-1]
                    better = True
    return order


def best_from(P, v, s, near=70, kmax=60):
    """Best route that starts at chest s: nearest-neighbour walk over its neighbourhood, cut at the best prefix, then 2-opt and
    pruning of chests that cost more time than they are worth."""
    d = np.hypot(*(P - P[s]).T)
    cand = np.argsort(d)[:near]
    order, left = [s], [c for c in cand if c != s]
    while left and len(order) < kmax:
        last = P[order[-1]]
        j = int(np.argmin([np.hypot(*(P[c] - last)) for c in left]))
        order.append(left.pop(j))
    legs = np.cumsum([0.0] + [leg_seconds(P, a, b) for a, b in zip(order[:-1], order[1:])])
    extra = np.cumsum(EXTRA[order]) if EXTRA is not None else np.zeros(len(order))
    ks = np.arange(1, len(order) + 1)
    r = np.cumsum(v[order]) / (OVERHEAD_S + legs + OPEN_S * ks + extra)           # rate of every prefix of the walk
    order = order[:int(np.argmax(r)) + 1]
    for _ in range(3):                                   # tidy: 2-opt, then drop any chest whose removal raises the rate
        order = two_opt(P, order, fixed_start=False)
        changed = False
        for c in list(order):
            if len(order) > 2:
                trial = [x for x in order if x != c]
                if rate(P, v, trial) > rate(P, v, order) + 1e-9:
                    order, changed = trial, True
        if not changed:
            break
    return rate(P, v, order), order


def best_routes(P, v, top=8, min_chests=6, separation_m=250.0):
    """Best distinct routes over all starting chests -> [(rate, order)], best first."""
    found = []
    for s in range(len(P)):
        r, o = best_from(P, v, s)
        if len(o) >= min_chests:
            found.append((r, o))
    found.sort(key=lambda t: -t[0])
    out = []
    for r, o in found:
        c = P[o].mean(axis=0)
        if all(np.hypot(*(c - P[o2].mean(axis=0))) > separation_m for _, o2 in out):
            out.append((r, o))
        if len(out) == top:
            break
    return out


def best_pair(P, v, s, t, budget_s, near=110, iters=60, seed=0, rounds=2):
    """Two players, `budget_s` each, starting at chests s and t and never taking the same chest: each route is re-optimised with
    the other's chests taken away, a few times over. -> (combined value, route of s, route of t) (indices into P)."""
    v = np.asarray(v, float)
    vb = v.copy()
    vb[s] = 0.0
    b = best_budget(P, vb, t, budget_s, near, iters, seed)[1]
    for r in range(rounds):
        va = v.copy()
        va[b] = 0.0
        va[s] = v[s]
        a = best_budget(P, va, s, budget_s, near, iters, seed + r)[1]
        vb = v.copy()
        vb[a] = 0.0
        vb[t] = v[t]
        b = best_budget(P, vb, t, budget_s, near, iters, seed + r)[1]
    return float(v[a].sum() + v[b].sum()), a, b


def best_budget(P, v, s, budget_s, near=90, iters=0, seed=0):
    """Most value you can collect in `budget_s` seconds of walking + opening, starting at chest s (no leave/rejoin overhead
    inside the budget): cheapest-insertion by value per second, then 2-opt and refilling the time it freed, repeated. iters > 0
    adds an iterated local search on top: drop a few chests, refill with a little randomness, 2-opt, keep what is better."""
    d = np.hypot(*(P - P[s]).T)
    cand = np.argsort(d)[:near]
    cand = np.concatenate([[s], cand[cand != s]])
    m = len(cand)
    T = np.zeros((m, m))
    for i in range(m):
        for j in range(i + 1, m):
            T[i, j] = T[j, i] = leg_seconds(P, cand[i], cand[j])
    ex = EXTRA[cand] if EXTRA is not None else np.zeros(m)
    cost = OPEN_S + ex
    vv = v[cand]

    def total(route):
        return sum(T[a, b] for a, b in zip(route[:-1], route[1:])) + cost[route].sum()

    def fill(route, rng=None):
        route = list(route)
        t = total(route)
        while True:
            used = np.zeros(m, bool)
            used[route] = True
            free = np.nonzero(~used & (vv > 0))[0]
            if not len(free):
                return route
            R = np.array(route)
            # insert after position i (i = last: append): added walking time per (position, free chest)
            prev = T[R[:, None], free[None, :]]
            nxt = np.vstack([T[free[None, :], R[1:, None]], np.zeros((1, len(free)))])
            old = np.append(T[R[:-1], R[1:]], 0.0)[:, None]
            delta = prev + nxt - old + cost[free][None, :]
            ok = t + delta <= budget_s
            if not ok.any():
                return route
            ratio = np.where(ok, vv[free][None, :] / np.maximum(delta, 1e-6), -1.0)
            flat = ratio.ravel()
            pick = int(np.argmax(flat)) if rng is None else int(rng.choice(np.argpartition(-flat, min(2, len(flat) - 1))[:3]))
            if flat[pick] < 0:                                       # (a random pick among the top 3 may be one that does not fit)
                pick = int(np.argmax(flat))
            i, j = np.unravel_index(pick, ratio.shape)
            route.insert(int(i) + 1, int(free[j]))
            t += float(delta[i, j])

    def tidy(route):
        """2-opt on the open path, the start fixed, each move priced from the leg table."""
        r, n = list(route), len(route)
        improved = True
        while improved:
            improved = False
            for i in range(1, n - 1):
                for j in range(i + 1, n):
                    b, c = r[i], r[j]
                    old = T[r[i - 1], b] + (T[c, r[j + 1]] if j + 1 < n else 0.0)
                    new = T[r[i - 1], c] + (T[b, r[j + 1]] if j + 1 < n else 0.0)
                    if new < old - 1e-9:
                        r[i:j + 1] = r[i:j + 1][::-1]
                        improved = True
        return r

    route = [0]
    for _ in range(3):
        route = tidy(fill(route))
    if iters:
        rng = np.random.default_rng(seed)
        cur = best = route
        stale = 0
        for _ in range(iters):
            r = list(cur)
            if len(r) > 1:
                k = int(rng.integers(1, min(4, len(r) - 1) + 1))
                if rng.random() < 0.5:                                # drop a stretch of the walk ...
                    a = int(rng.integers(1, len(r) - k + 1))
                    del r[a:a + k]
                else:                                                 # ... or a few chests anywhere on it
                    gone = set(rng.choice(np.arange(1, len(r)), k, replace=False).tolist())
                    r = [x for q, x in enumerate(r) if q not in gone]
            r = tidy(fill(r, rng))
            rv, rt, bv, bt = vv[r].sum(), total(r), vv[best].sum(), total(best)
            if rv > bv + 1e-9 or (abs(rv - bv) <= 1e-9 and rt < bt - 1e-9):
                best, stale = r, 0
            else:
                stale += 1
            if rv >= vv[cur].sum() - 1e-9 or rng.random() < 0.05:
                cur = r
            if stale >= 60:                                           # lost: start again from the best one
                cur, stale = best, 0
        route = best
    keep = [int(cand[i]) for i in route]
    return float(v[keep].sum()), keep, total(route)
