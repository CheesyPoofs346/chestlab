"""Live sensing, split by how fast each quantity changes. Heading changes in milliseconds (a mouse flick), so a small
loop captures just the compass bar at ~100 Hz and feeds a HeadingFilter. Position changes in seconds, so another loop
fits the minimap at ~30 Hz. The renderer asks both for "now" every frame (60+ Hz) and gets values extrapolated to the
instant the picture will be on screen.

The two loops are plain functions. Engine runs them on threads (simple; used by the tests); procs.ProcessEngine runs
the same functions in separate processes, which is what the live app uses so the sensors can't slow the renderer."""
import threading
import time

import cv2
import numpy as np
import mss

from .compass import BarComb, region_px
from .heading import BAR_SIGMA, OCR_SIGMA, HeadingFilter
from .live import MINIMAP_BOX, crop_minimap

MSS = getattr(mss, "MSS", None) or mss.mss
COMPASS_HZ, POSE_HZ = 60, 30           # 60 Hz is as often as the screen is redrawn: reading the compass faster only costs CPU
BAR_MIN_QUALITY = 1.25
MAX_EXTRAPOLATION = 0.15     # s: past this a position is stale rather than "predicted"
RECENT = 30                  # raw clear-weather fixes kept for recording a chest


class MssGrabber:
    """Screen capture. Calling it returns grab(region dict) -> BGRA ndarray with its own mss handle (once per thread)."""

    def __call__(self):
        sct = MSS()
        return lambda r: np.asarray(sct.grab(r))


class FrameSource:
    """Stands in for the screen in offline tests: serves regions of a fixed BGR frame, no window involved."""

    def __init__(self, frame):
        self.bgra = cv2.cvtColor(frame, cv2.COLOR_BGR2BGRA)

    def __call__(self):
        return lambda r: self.bgra[r["top"]:r["top"] + r["height"], r["left"]:r["left"] + r["width"]]


class Rate:
    """Exponentially smoothed events per second."""

    def __init__(self):
        self.t, self.v, self.n = None, 0.0, 0

    def tick(self):
        self.n += 1
        now = time.perf_counter()
        if self.t is not None:
            self.v += 0.1 * (1.0 / max(now - self.t, 1e-4) - self.v)
        self.t = now


# ---- the two sensor loops (thread- or process-agnostic)
def compass_loop(grab, get_mon, stopped, hf, reader, on_cycle, log=lambda m: None, paused=lambda: False):
    """Capture the compass strip, read the heading number (label comb as fallback), feed `hf`. on_cycle(src, ocr, err)."""
    bar = BarComb()
    period = 1.0 / COMPASS_HZ
    src, ocr = "-", None
    while not stopped():
        t0 = time.perf_counter()
        err = None
        if paused():                                   # the full map is open: the compass is not on screen, nothing there to read
            time.sleep(0.05)
            continue
        try:
            mon = get_mon()
            x0, y0, w, h = region_px(mon["width"])
            reg = cv2.cvtColor(grab(dict(left=mon["left"] + x0, top=mon["top"] + y0, width=w, height=h)), cv2.COLOR_BGRA2BGR)
            t = time.perf_counter()
            geo = dict(s=mon["width"] / 2000.0, ox=x0, oy=y0)
            prior = hf.prior(t)
            n = reader.read(reg, tries=2 if prior is not None else None, **geo)
            if n is not None:
                ocr = n
                hf.update(t, n, OCR_SIGMA)
                src = "number"
            elif prior is not None:
                est = bar.estimate(reg, prior, **geo)
                if est[1] >= BAR_MIN_QUALITY and hf.update(t, est[0], BAR_SIGMA, gate=6.0):
                    src = "bar"
        except Exception as e:
            err = "compass " + repr(e)
            log("ERROR " + err)
            time.sleep(0.2)
        on_cycle(src, ocr, err)
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def pose_loop(grab, get_mon, stopped, step, on_pose, log=lambda m: None):
    """Capture the minimap, fit it, publish. step(minimap, dt) -> tracker result; on_pose(t, r, brightness, err)."""
    last = time.perf_counter()
    period = 1.0 / POSE_HZ
    while not stopped():
        t0 = time.perf_counter()
        try:
            mon = get_mon()
            s = mon["width"] / 2000.0
            x0, y0, x1, y1 = (int(round(v * s)) for v in MINIMAP_BOX)
            reg = cv2.cvtColor(grab(dict(left=mon["left"] + x0, top=mon["top"] + y0, width=x1 - x0, height=y1 - y0)), cv2.COLOR_BGRA2BGR)
            t = time.perf_counter()
            r = step(crop_minimap(reg, s=s, ox=x0, oy=y0), max(t - last, 1e-3))
            last = t
            on_pose(t, r, float(reg[::8, ::8].mean()), None)
        except Exception as e:
            err = "minimap " + repr(e)
            log("ERROR " + err)
            on_pose(time.perf_counter(), None, 0.0, err)
            time.sleep(0.2)
        time.sleep(max(0.0, period - (time.perf_counter() - t0)))


def pose_record(t, r, tracker, brightness):
    """Plain dict for the renderer from one tracker result."""
    f = tracker.filt
    return dict(t=t, locked=bool(r["locked"]), on_island=bool(r["on_island"]), pos=r["pos"],
                vel=None if f.x is None else (float(f.x[2]), float(f.x[3])),
                storm=bool(r["raw"] is not None and r["fit"].adaptive), coasting=bool(r["coasting"]),
                score=tracker.fitter.last_score, zoom=tracker.fitter.zoom, brightness=brightness)


# ---- what the renderer asks for (same for both engines)
class Snapshots:
    @property
    def err(self):
        return self.err_c or self.err_p

    def snapshot(self, t, W, H, lead=0.0):
        """State for a frame that will be on screen at about t + lead."""
        p = self.get_pose()
        hd = self.get_heading(t + lead)
        st = dict(locked=False, on_island=False, pos=None, heading=None, compass=self.ocr, markers=[], score=0.0, zoom=None,
                  brightness=0.0, storm=False, src=self.heading_src, hrate=0.0, hage=None)
        if hd is not None:
            st.update(heading=hd[0], hrate=hd[1], hage=hd[2])
        if p is not None:
            st.update(locked=p["locked"], on_island=p["on_island"], score=p["score"], zoom=p["zoom"], brightness=p["brightness"],
                      storm=p["storm"])
            pos = p["pos"]
            if pos is not None:
                dt = min(max(t + lead - p["t"], 0.0), MAX_EXTRAPOLATION)
                v = p["vel"] or (0.0, 0.0)
                pos = (pos[0] + v[0] * dt, pos[1] + v[1] * dt)
                st["pos"] = pos
        self.pipe.live_pos = st["pos"]
        fresh = hd is not None and hd[2] < 1.0 and p is not None and t - p["t"] < 1.0
        if fresh and st["locked"] and st["on_island"]:
            st["markers"] = self.pipe.project(W, H, st["pos"], st["heading"])
            st.update(self.pipe.view(st["heading"], st["pos"], st["storm"]))
        return st


class Engine(Snapshots):
    """Both sensor loops on threads of this process."""

    def __init__(self, pipe, rect, log=lambda m: None, grabber=MssGrabber()):
        self.pipe, self.mon, self.log, self.grabber = pipe, dict(rect), log, grabber
        self.hf = HeadingFilter()
        self.lock = threading.Lock()          # guards the tracker + chest list (sensor thread vs hotkey actions)
        self.stop = False
        self.pose = None
        self.ocr = None
        self.heading_src = "-"
        self.hold = self._held = False
        self.err_c = self.err_p = None
        self.rate_c, self.rate_p = Rate(), Rate()
        self.threads = [threading.Thread(target=self._compass, daemon=True, name="compass"),
                        threading.Thread(target=self._pose, daemon=True, name="minimap")]

    def start(self):
        for t in self.threads:
            t.start()

    def get_pose(self):
        return self.pose

    def get_heading(self, t):
        return self.hf.at(t)

    def _compass(self):
        def on_cycle(src, ocr, err):
            self.heading_src, self.ocr, self.err_c = src, ocr, err
            self.rate_c.tick()
        compass_loop(self.grabber(), lambda: self.mon, lambda: self.stop, self.hf, self.pipe.reader, on_cycle, self.log, lambda: self.hold)

    def _pose(self):
        def step(mm, dt):
            with self.lock:
                if self.hold:                                    # the full map is open: the minimap is not on screen
                    self._held = True
                    return dict(locked=False, coasting=False, on_island=False, raw=None, pos=None, heading=None)
                if self._held:                                   # back in the game: find the position again from scratch
                    tr = self.pipe.tracker
                    tr.filt.reset()
                    tr.cand, tr.misses, self._held = None, 0, False
                return self.pipe.step_minimap(mm, dt)

        def on_pose(t, r, brightness, err):
            self.err_p = err
            if r is not None:
                self.pose = pose_record(t, r, self.pipe.tracker, brightness)
                self.rate_p.tick()
        pose_loop(self.grabber(), lambda: self.mon, lambda: self.stop, step, on_pose, self.log)
