"""The sensor loops in their own processes. Python threads share one GIL, and the renderer measured 4-5x slower with
the OCR / minimap-fit threads running next to it. Separate processes publish their results through shared memory
(a seqlock: odd sequence = being written) and the renderer reads them without ever waiting for a sensor.
perf_counter is machine-wide on Windows, so timestamps from the children are directly comparable."""
import multiprocessing as mp
import os
import threading
import time
from collections import deque

import numpy as np

from .paths import APP
from .engine import RECENT, MssGrabber, Rate, Snapshots, compass_loop, pose_loop, pose_record
from .heading import HeadingFilter, extrapolate

H_N, P_BASE = 12, 24
P_N = P_BASE + 2 * RECENT
SRC = {0: "-", 1: "number", 2: "bar"}


def _write(arr, vals):
    """Seqlock write: values[1:] then the sequence number goes even again."""
    arr[0] += 1
    for i, v in enumerate(vals, 1):
        arr[i] = v
    arr[0] += 1


def _read(arr, n):
    for _ in range(50):
        s = arr[0]
        if s % 2:
            continue
        vals = arr[:n]
        if arr[0] == s:
            return vals
    return None


def _logger(path):
    def log(msg):
        try:
            with open(path, "a", encoding="utf-8") as f:
                f.write(time.strftime("%H:%M:%S ") + msg + "\n")
        except OSError:
            pass
    return log


def _alive():
    parent = mp.parent_process()
    return lambda: parent is None or parent.is_alive()


# ---- children
def NO_LOCK():
    return dict(locked=False, coasting=False, on_island=False, raw=None, pos=None, heading=None)


def compass_main(H, M, stop, grabber, log_path, hold=None):
    from .compass import CompassReader
    hf, reader, rate, alive = HeadingFilter(), CompassReader(), Rate(), _alive()
    mon = lambda: dict(left=M[0], top=M[1], width=M[2], height=M[3])

    def on_cycle(src, ocr, err):
        rate.tick()
        ex = hf.export()
        _write(H, [1.0 if ex else 0.0, *(ex or (0, 0, 0, 0)), -1.0 if ocr is None else ocr,
                   {"-": 0, "number": 1, "bar": 2}[src], rate.v, 1.0 if err else 0.0])
    compass_loop(grabber(), mon, lambda: stop.is_set() or not alive(), hf, reader, on_cycle, _logger(log_path), lambda: hold is not None and bool(hold.value))


def pose_main(P, M, stop, grabber, log_path, hold=None):
    from .estimate import Tracker
    from .world import World
    from . import worldmap
    tr, rate, alive = (worldmap.WorldTracker(World()) if worldmap.available() else Tracker(World())), Rate(), _alive()
    recent, total = deque(maxlen=RECENT), [0]
    mon = lambda: dict(left=M[0], top=M[1], width=M[2], height=M[3])

    held = [False]

    def step(mm, dt):
        if hold is not None and hold.value:          # the full map is open: the minimap is not on screen, what is there matches nothing
            held[0] = True
            return NO_LOCK()
        if held[0]:                                  # back in the game: forget whatever was said before, find the position again
            tr.filt.reset()
            tr.cand, tr.misses, held[0] = None, 0, False
        r = tr.step(mm, dt)
        if r["raw"] is not None and not r["fit"].adaptive:          # storm-mode fixes are ~2 m rougher: not for recording chests
            recent.append(r["raw"])
            total[0] += 1
        return r

    def on_pose(t, r, brightness, err):
        if r is None:
            P[P_BASE - 1] = 1.0
            return
        rate.tick()
        d = pose_record(t, r, tr, brightness)
        pos, vel = d["pos"], d["vel"] or (0.0, 0.0)
        flat = [x for fix in recent for x in fix]
        _write(P, [1.0, t, d["locked"], d["on_island"], pos[0] if pos else 0.0, pos[1] if pos else 0.0, pos is not None, vel[0], vel[1],
                   d["storm"], d["coasting"], d["score"], d["zoom"] or 0.0, brightness, rate.v, len(recent), total[0], 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
                  + flat + [0.0] * (2 * RECENT - len(flat)))
    pose_loop(grabber(), mon, lambda: stop.is_set() or not alive(), step, on_pose, _logger(log_path))


class _Hz:
    def __init__(self, get):
        self._get = get

    @property
    def v(self):
        return self._get()


class ProcessEngine(Snapshots):
    """Same interface as engine.Engine, sensors in two child processes."""

    def __init__(self, pipe, rect, log=lambda m: None, grabber=MssGrabber(), log_path=None):
        self.pipe, self.log, self.grabber = pipe, log, grabber
        ctx = mp.get_context("spawn")
        self.ctx = ctx
        self.H, self.P, self.M = ctx.RawArray("d", H_N), ctx.RawArray("d", P_N), ctx.RawArray("i", 4)
        self.stop_ev = ctx.Event()
        self.hold = ctx.RawValue("i", 0)        # 1 while the full map is open: the pose process pauses the minimap fit
        self.log_path = log_path or os.path.join(APP, "user", "live.log")
        self.lock = threading.Lock()
        self.mon = rect
        self.procs = []
        self._seen = -1
        self._pose = None
        self.rate_c = _Hz(lambda: self.H[8])       # H layout: 1 valid, 2 th, 3 om, 4 t, 5 tm, 6 ocr, 7 src, 8 rate, 9 err
        self.rate_p = _Hz(lambda: self.P[15] if self.P[0] else 0.0)

    # mon: dict of the capture rectangle, shared with the children
    @property
    def mon(self):
        return dict(left=self.M[0], top=self.M[1], width=self.M[2], height=self.M[3])

    @mon.setter
    def mon(self, r):
        for i, k in enumerate(("left", "top", "width", "height")):
            self.M[i] = int(r[k])

    @property
    def stop(self):
        return self.stop_ev.is_set()

    @stop.setter
    def stop(self, v):
        if v:
            self.stop_ev.set()

    def start(self):
        for target, arr in ((compass_main, self.H), (pose_main, self.P)):
            p = self.ctx.Process(target=target, args=(arr, self.M, self.stop_ev, self.grabber, self.log_path, self.hold), daemon=True)
            p.start()
            self.procs.append(p)

    def close(self):
        self.stop_ev.set()
        for p in self.procs:
            p.join(timeout=2.0)
            if p.is_alive():
                p.terminate()

    # ---- readers
    def _h(self):
        v = _read(self.H, H_N)
        return v if v and v[1] else None

    @property
    def ocr(self):
        v = self._h()
        return None if v is None or v[6] < 0 else int(v[6])

    @property
    def heading_src(self):
        v = self._h()
        return "-" if v is None else SRC.get(int(v[7]), "-")

    @property
    def err_c(self):
        return "compass sensor error (see user/live.log)" if self.H[9] else None

    @property
    def err_p(self):
        return "minimap sensor error (see user/live.log)" if self.P[P_BASE - 1] else None

    def get_heading(self, t):
        v = self._h()
        return None if v is None else extrapolate(v[2], v[3], v[4], v[5], t)

    def get_pose(self):
        v = _read(self.P, P_N)
        if not v or not v[1]:
            return None
        if v[0] != self._seen:
            self._seen = v[0]
            n = int(v[16])
            fixes = [(v[P_BASE + 2 * i], v[P_BASE + 2 * i + 1]) for i in range(min(n, RECENT))]
            self.pipe.recent.clear()
            self.pipe.recent.extend(fixes)
            self._pose = dict(t=v[2], locked=bool(v[3]), on_island=bool(v[4]), pos=(v[5], v[6]) if v[7] else None, vel=(v[8], v[9]),
                              storm=bool(v[10]), coasting=bool(v[11]), score=v[12], zoom=v[13] or None, brightness=v[14])
        return self._pose

    @property
    def pose(self):
        return self.get_pose()
