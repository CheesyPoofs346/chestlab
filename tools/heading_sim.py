"""How well does the heading follow a real mouse? Synthetic turns (flicks, pans, stops) read through the compass bar's
integer readout, with dropouts and misreads. Compares the old scheme (OCR every 0.12 s, alpha-beta) with the new one."""
import os, sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from chestlab.heading import HeadingFilter, wrap180


def truth(dur=40.0, seed=1):
    rng = np.random.default_rng(seed)
    t = np.arange(0, dur, 0.001)
    om = np.zeros_like(t)
    i = 0
    while i < len(t):
        kind = rng.choice(["still", "pan", "flick", "wobble"], p=[.25, .3, .25, .2])
        n = int(rng.uniform(0.3, 1.6) * 1000)
        seg = np.zeros(min(n, len(t) - i))
        w = np.hanning(len(seg)) if len(seg) > 2 else np.ones(len(seg))
        if kind == "pan": seg = w * rng.uniform(15, 90) * rng.choice([-1, 1])
        elif kind == "flick": seg = w * rng.uniform(250, 700) * rng.choice([-1, 1]); seg = seg[: int(0.4 * 1000)] if False else seg
        elif kind == "wobble": seg = w * rng.uniform(5, 30) * np.sin(np.linspace(0, 6 * np.pi, len(seg)))
        om[i:i + len(seg)] = seg; i += len(seg)
    th = (np.cumsum(om) * 0.001 + rng.uniform(0, 360)) % 360
    return t, th, om


def sample(t, th, ts):
    ang = np.unwrap(np.radians(th))
    return np.degrees(np.interp(ts, t, ang)) % 360


def run(seed=1, ocr_hz=80, drop=0.10, misread=0.01, latency=0.008, render_hz=60, old=False, comp=0.0):
    rng = np.random.default_rng(seed + 100)
    t, th, om = truth(seed=seed)
    f = HeadingFilter()
    # old scheme state
    a_th = a_om = None; a_last = 0.0; cached = None; cached_t = -1
    errs, rates = [], []
    ts_meas = np.arange(0.05, t[-1] - 0.1, 1.0 / ocr_hz) + rng.uniform(-0.002, 0.002, int((t[-1] - 0.15) * ocr_hz) + 1)[: len(np.arange(0.05, t[-1] - 0.1, 1.0 / ocr_hz))]
    ts_render = np.arange(0.5, t[-1] - 0.1, 1.0 / render_hz)
    ev = [(x, 'm') for x in ts_meas] + [(x, 'r') for x in ts_render]
    ev.sort()
    for x, k in ev:
        if k == 'm':
            if rng.random() < drop: continue
            true_at = sample(t, th, np.array([x - latency]))[0]      # the HUD shows the heading from `latency` ago
            r = round(true_at) % 360
            if rng.random() < misread: r = int(rng.integers(0, 360))
            if old:
                if x - cached_t >= 0.12: cached, cached_t = r, x
            else:
                f.update(x - comp, r)
        else:
            tr = sample(t, th, np.array([x]))[0]
            if old:
                # old: every frame (~33 ms) alpha-beta toward the cached reading; render uses the filtered value
                if cached is None: continue
                if a_th is None: a_th, a_om = cached, 0.0
                dtf = 1.0 / render_hz
                a_th = (a_th + a_om * dtf) % 360
                e = wrap180(cached - a_th); a_th = (a_th + 0.35 * e) % 360; a_om += 0.08 * e / dtf
                errs.append(abs(wrap180(a_th - tr))); rates.append(abs(om[int(x * 1000)]))
            else:
                r = f.at(x)
                if r is None: continue
                errs.append(abs(wrap180(r[0] - tr))); rates.append(abs(om[int(x * 1000)]))
    e, rt = np.array(errs), np.array(rates)
    return e, rt


if __name__ == "__main__":
    for name, kw in (("old (8 Hz OCR, alpha-beta)", dict(old=True)), ("new filter, 80 Hz OCR", dict()), ("new + latency comp (known)", dict(comp=0.008)),
                     ("new, 25 ms latency, comp 25", dict(latency=0.025, comp=0.025)), ("new, 25 ms latency, NO comp", dict(latency=0.025)),
                     ("new, 40 Hz OCR, 25% dropouts", dict(ocr_hz=40, drop=0.25)), ("new, 100 Hz, 5% dropouts", dict(ocr_hz=100, drop=0.05))):
        E, R = [], []
        for s in range(1, 6):
            e, r = run(seed=s, **kw); E.append(e); R.append(r)
        e, r = np.concatenate(E), np.concatenate(R)
        mv = r > 100
        print("%-32s all: p50 %.2f p95 %.2f max %.1f deg | turning>100deg/s: p50 %.2f p95 %.2f | still(<5): p95 %.2f" % (
            name, np.percentile(e, 50), np.percentile(e, 95), e.max(), np.percentile(e[mv], 50), np.percentile(e[mv], 95), np.percentile(e[r < 5], 95)))
