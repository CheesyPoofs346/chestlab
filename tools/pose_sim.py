"""Position tracking with a scripted minimap fitter: standing, walking, sprinting, lone outliers, a teleport.
Compares the tracker as it was (no outlier gate, no calm mode) with the new one. Units: map px (1 px ~ 0.0875 m)."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chestlab.estimate import Tracker
from chestlab.live import Pipeline


class Fake:
    adaptive = False
    last_score = 0.9
    zoom = 0.0698
    def __init__(self): self.next = None
    def fit(self, mm, guess_uv=None):
        return self.next


class F:
    def __init__(self, u, v, adaptive=False): self.u, self.v, self.adaptive = u, v, adaptive


def path(dt=1 / 30):
    pts, p, v = [], np.array([500.0, 400.0]), np.zeros(2)
    plan = [(3.0, 0, 0), (4.0, 82, 40), (2.0, 0, 0), (3.0, 170, -100), (2.0, 0, 0), (4.0, 82, 200)]   # secs, speed px/s, heading deg
    for dur, sp, hd in plan:
        v = sp * np.array([np.sin(np.radians(hd)), -np.cos(np.radians(hd))])
        for _ in range(int(dur / dt)):
            p = p + v * dt; pts.append((p.copy(), v.copy()))
    return np.array([x[0] for x in pts]), np.array([x[1] for x in pts])


def run(new, seed=0, sigma=6.0, outl=0.03, dt=1 / 30):
    rng = np.random.default_rng(seed)
    tr = Tracker(Pipeline(reader=object()).world); tr.fitter = Fake()
    if not new:
        Tracker.GATE_PX = 1e9; tr.filt.CALM_S = 1e9
    else:
        Tracker.GATE_PX = 50.0; tr.filt.CALM_S = 0.5
    P, V = path(dt)
    tele = len(P) // 2 + 20                                    # teleport 400 px away here, for good
    off = np.zeros(2); out, truth, still, rej = [], [], [], 0
    for i, (p, v) in enumerate(zip(P, V)):
        if i == tele: off = np.array([400.0, -250.0])
        meas = p + off + rng.normal(0, sigma, 2)
        if rng.random() < outl: meas = meas + rng.uniform(60, 200) * np.array([np.cos(a := rng.uniform(0, 6.28)), np.sin(a)])
        tr.fitter.next = F(*meas)
        r = tr.step(None, dt, heading=0.0)
        out.append(r["pos"] if r["pos"] is not None else (np.nan, np.nan)); truth.append(p + off); still.append(np.hypot(*v) < 1)
    out, truth, still = np.array(out), np.array(truth), np.array(still)
    e = np.hypot(*(out - truth).T)
    post = np.arange(len(e)) > tele + 25                          # after the teleport has been confirmed
    pre = np.arange(len(e)) < tele
    ok = ~np.isnan(e)
    return e, still, post | pre, tele


if __name__ == "__main__":
    for name, new in (("old tracker", False), ("new tracker", True)):
        E, S, M = [], [], []
        for s in range(8):
            e, still, mask, tele = run(new, seed=s)
            E.append(e[mask]); S.append(still[mask])
            rec = np.where(e[tele:tele + 60] < 25)[0]
            if s == 0: rec0 = rec[0] if len(rec) else -1
        e, st = np.concatenate(E), np.concatenate(S)
        print("%s: error px (m) p50 %.1f (%.2f)  p95 %.1f (%.2f)  max %.0f (%.1f) | standing: p95 %.1f px | moving: p95 %.1f px | teleport recovered after %d frames" % (
            name, np.nanpercentile(e, 50), np.nanpercentile(e, 50) * .0875, np.nanpercentile(e, 95), np.nanpercentile(e, 95) * .0875, np.nanmax(e), np.nanmax(e) * .0875,
            np.nanpercentile(e[st], 95), np.nanpercentile(e[~st], 95), rec0))
