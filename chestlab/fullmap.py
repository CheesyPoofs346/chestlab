"""The in-game full map (M) as a surface for the chest overlay.

assets/gg/full_z4.png is the whole season map as fortnite.gg draws it (4096 px = 256 map units, 16 px a unit: the coordinates the chest
data uses). The in-game map is the same terrain, but drawn softer, with labels, a vignette and a different sea, so feature matching (SIFT)
does not hold up on it. What both share is the SHAPE of the land, so the map is found by correlating land / water silhouettes: a coarse
search over zoom and position (once, ~0.5 s), then a fine search (zoom and shift, sub-pixel) that also tells how sure we are. While the
map is open, optical flow follows your panning and zooming between those fine searches.

The map never rotates, so a registration is just (s, tx, ty): screen px = s * z4 px + (tx, ty).

MapFinder  - coarse() / refine() / register(): the maths, offline-testable on any frame
MapWatch   - a thread that runs while the map is open and publishes .state() = dict(M = z4 px -> overlay px (2x3), score, age)"""
import os
import threading
import time

import cv2
import numpy as np

from .paths import BASE as ROOT
REF = os.path.join(ROOT, "assets", "gg", "full_z4.png")
ZREF = 4096.0                  # side of the full map in z4 px
FW_COARSE, FW_FINE, WORK_W = 320, 640, 640       # frame widths for the coarse search, the fine search, and the optical flow
S_RANGE = (0.12, 2.0)          # zoom range, screen px per z4 px at a 1125 px tall screen (it scales with the screen height)
ACCEPT, KEEP = 0.70, 0.45      # silhouette correlation needed to take a new map / to keep believing in the one we have
MIN_STRUCT = 18.0              # a frame with a flatter silhouette (all land, say) says nothing about where we are


def land(bgr):
    """255 where the picture is land, 0 on water and black: sea is saturated blue / turquoise, land is everything else that is lit."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    water = ((h >= 82) & (h <= 128) & (s > 80) & (v > 40)) | ((h > 128) & (h <= 142) & (s > 110) & (v > 60))      # the 2nd: a storm tints the sea purple (the land goes pink, hue 145+)
    return ((~water) & (v >= 45)).astype(np.uint8) * 255


CROP = (0.20, 0.08, 0.80, 0.93)         # the part of the screen the silhouette is cut from: the game's side panels and bars stay out of it
METHODS = ("hue", "cluster")


def cluster_land(small):
    """Land / sea without assuming what colour the sea is (a storm tints the whole map purple): split the middle of the picture into two
    colour groups; the sea is the group that fills the edge of that region (the island sits in the middle of it)."""
    h, w = small.shape[:2]
    x0, y0, x1, y1 = int(w * CROP[0]), int(h * CROP[1]), int(w * CROP[2]), int(h * CROP[3])
    lab = cv2.cvtColor(cv2.GaussianBlur(small, (0, 0), 1.2), cv2.COLOR_BGR2LAB).astype(np.float32)
    pts = lab[y0:y1, x0:x1].reshape(-1, 3)
    pts = pts[::max(1, len(pts) // 6000)]
    cv2.setRNGSeed(1)
    _, _, cent = cv2.kmeans(pts, 2, None, (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 25, 0.5), 2, cv2.KMEANS_PP_CENTERS)
    lab_of = (((lab[..., None, :] - cent[None, None]) ** 2).sum(-1)).argmin(-1)
    ring = np.zeros((h, w), bool)
    ring[y0:y1, x0:x1] = True
    ry, rx = max(2, int((y1 - y0) * 0.14)), max(2, int((x1 - x0) * 0.14))
    ring[y0 + ry:y1 - ry, x0 + rx:x1 - rx] = False
    sea = int(np.bincount(lab_of[ring], minlength=2).argmax())
    return ((lab_of != sea).astype(np.uint8)) * 255


def silhouette(bgr, width, method="hue"):
    """The frame's land mask at `width` px, softened so that correlation is smooth and cut to the middle of the screen: (mask, shrink factor,
    (ox, oy) = where that cut starts in the full-width picture)."""
    f = width / bgr.shape[1]
    small = cv2.resize(bgr, (width, max(1, int(round(bgr.shape[0] * f)))), interpolation=cv2.INTER_AREA)
    m = cv2.GaussianBlur(land(small) if method == "hue" else cluster_land(small), (0, 0), 1.5)
    h, w = m.shape
    ox, oy = int(w * CROP[0]), int(h * CROP[1])
    return np.ascontiguousarray(m[oy:int(h * CROP[3]), ox:int(w * CROP[2])]), f, (ox, oy)


def _h3(M):
    return np.vstack([M, [0.0, 0.0, 1.0]])


def compose(*Ms):
    """Apply 2x3 transforms right to left (the last one first)."""
    out = np.eye(3)
    for M in Ms:
        out = out @ _h3(M)
    return out[:2]


def matrix(s, tx, ty):
    return np.array([[s, 0.0, tx], [0.0, s, ty]])


def scale_of(M):
    return float(np.hypot(M[0, 0], M[1, 0]))


def _peak(res):
    """Best position of a correlation map with sub-pixel refinement: (score, x, y)."""
    _, mx, _, (x, y) = cv2.minMaxLoc(res)
    fx = fy = 0.0
    if 0 < x < res.shape[1] - 1:
        a, b, c = res[y, x - 1], res[y, x], res[y, x + 1]
        d = a - 2 * b + c
        fx = 0.5 * (a - c) / d if d < -1e-9 else 0.0
    if 0 < y < res.shape[0] - 1:
        a, b, c = res[y - 1, x], res[y, x], res[y + 1, x]
        d = a - 2 * b + c
        fy = 0.5 * (a - c) / d if d < -1e-9 else 0.0
    return float(mx), x + float(np.clip(fx, -1, 1)), y + float(np.clip(fy, -1, 1))


class MapFinder:
    def __init__(self, ref=None):
        ref = cv2.imread(REF) if ref is None else ref
        if ref is None:
            raise FileNotFoundError(REF)
        blur = lambda k: cv2.GaussianBlur(land(cv2.resize(ref, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)), (0, 0), 1.5)
        self.coarse_ref, self.fine_ref = blur(0.25), blur(0.5)

    def coarse(self, frame, method="hue"):
        """Search every zoom and position: (score, s, tx, ty) of the best silhouette match, or None for a frame with no structure."""
        tpl, f, (ox, oy) = silhouette(frame, FW_COARSE, method)
        if tpl.std() < MIN_STRUCT:
            return None
        th, tw = tpl.shape
        k = frame.shape[0] / 1125.0
        found = []
        s = S_RANGE[0] * k
        while s <= S_RANGE[1] * k:
            n = int(round(ZREF * s * f))
            if n >= 24:
                R = cv2.resize(self.coarse_ref, (n, n), interpolation=cv2.INTER_AREA if n < self.coarse_ref.shape[0] else cv2.INTER_LINEAR)
                R = cv2.copyMakeBorder(R, th, th, tw, tw, cv2.BORDER_CONSTANT, value=0)
                sc, x, y = _peak(cv2.matchTemplate(R, tpl, cv2.TM_CCOEFF_NORMED))
                found.append((sc, s, (ox + tw - x) / f, (oy + th - y) / f))
            s *= 1.12
        inner = found[1:-1]                                  # a best fit at the very edge of the zoom range is "wants to be something else", not a map
        return max(inner, key=lambda c: c[0]) if inner else None

    def _at(self, tpl, f, off, shape, s, pivot, m):
        """Correlate the frame's silhouette with the map at zoom s, the map point `pivot` (z4 px) staying where it was on screen; searches
        +-m work px. -> (score, tx, ty) of the best shift."""
        th, tw = tpl.shape
        H, W = shape
        base = self.fine_ref
        k = ZREF * s * f / base.shape[0]                                   # base px -> work px
        cx, cy = f * (W / 2 - s * pivot[0]), f * (H / 2 - s * pivot[1])    # where the map's origin lands in the work frame
        win = cv2.warpAffine(base, np.array([[k, 0, cx + m - off[0]], [0, k, cy + m - off[1]]]), (tw + 2 * m, th + 2 * m), flags=cv2.INTER_LINEAR, borderValue=0)
        sc, x, y = _peak(cv2.matchTemplate(win, tpl, cv2.TM_CCOEFF_NORMED))
        return sc, (cx - (x - m)) / f, (cy - (y - m)) / f

    def refine(self, frame, s0, tx0, ty0, ds=0.06, m=16, steps=9, method="hue"):
        """Fine search around a guess: (score, s, tx, ty), or None when the frame's silhouette is too flat to say anything."""
        tpl, f, off = silhouette(frame, FW_FINE, method)
        if tpl.std() < MIN_STRUCT:
            return None
        H, W = frame.shape[:2]
        pivot = ((W / 2 - tx0) / s0, (H / 2 - ty0) / s0)
        grid = s0 * np.exp(np.linspace(-ds, ds, steps))
        scores = np.array([self._at(tpl, f, off, (H, W), s, pivot, m)[0] for s in grid])
        i = int(np.argmax(scores))
        s = float(grid[i])
        if 0 < i < len(grid) - 1:                                          # parabola through the three best zooms
            a, b, c = scores[i - 1], scores[i], scores[i + 1]
            d = a - 2 * b + c
            if d < -1e-9:
                s = float(grid[i] * np.exp(np.clip(0.5 * (a - c) / d, -1, 1) * (np.log(grid[1] / grid[0]))))
        sc, tx, ty = self._at(tpl, f, off, (H, W), s, pivot, m)
        return sc, s, tx, ty

    def register(self, frame):
        """Find the map in a frame from scratch: (2x3 z4 px -> frame px, score, method) or None. Tries the colour rule first (a normal map), then
        the colour-free one (a storm tints everything)."""
        for method in METHODS:
            c = self.coarse(frame, method)
            if c is None or c[0] < 0.45:
                continue
            r = self.refine(frame, c[1], c[2], c[3], ds=0.10, m=24, method=method)
            if r is not None and r[0] >= ACCEPT:
                return matrix(*r[1:]), r[0], method
        return None


def grow(mask, k=11):
    """A free-of-overlay mask with the overlay's pixels fattened by k px: our icons sit where the map was one capture ago, so they are a few px off."""
    return None if mask is None else cv2.erode(mask, np.ones((k, k), np.uint8))


def track(prev, cur, mask=None, mask_cur=None):
    """Motion of the map between two grayscale frames: 2x3 prev-px -> cur-px, or None when there is too little to follow. Corners under our
    own overlay (mask = 0, in the previous frame / mask_cur in this one) are not used: they do not move with the map."""
    p0 = cv2.goodFeaturesToTrack(prev, 250, 0.01, 9, mask=mask)
    if p0 is None or len(p0) < 12:
        return None
    p1, ok, _ = cv2.calcOpticalFlowPyrLK(prev, cur, p0, None, winSize=(31, 31), maxLevel=4)
    ok = ok.ravel() == 1
    if mask_cur is not None:
        h, w = mask_cur.shape
        xi = np.clip(np.round(p1[:, 0, 0]).astype(int), 0, w - 1)
        yi = np.clip(np.round(p1[:, 0, 1]).astype(int), 0, h - 1)
        ok &= mask_cur[yi, xi] > 0
    if ok.sum() < 12:
        return None
    D, inl = cv2.estimateAffinePartial2D(p0[ok], p1[ok], method=cv2.RANSAC, ransacReprojThreshold=1.5)
    if D is None or inl.sum() < 10 or abs(np.arctan2(D[1, 0], D[0, 0])) > 0.05:
        return None
    return D


def _power(A, k):
    """The screen-space similarity A (2x3, a uniform scale s plus a shift) applied k times (k real): the camera keeps moving the same way."""
    s = float(np.hypot(A[0, 0], A[1, 0]))
    b = A[:, 2]
    if abs(s - 1.0) < 1e-4:
        return np.array([[1.0, 0, b[0] * k], [0, 1.0, b[1] * k]])
    sk = s ** k
    g = (1.0 - sk) / (1.0 - s)
    return np.array([[sk, 0, b[0] * g], [0, sk, b[1] * g]])


class MapWatch:
    """Runs while the full map is open (start() on the M key). make_grab() -> grab() -> BGRA frame of the game screen (made inside the thread);
    mask() -> uint8 (255 = free of our overlay) or None; out_wh = the overlay's size. state() is None, or a dict with M (z4 px -> overlay px,
    2x3), score (how well the land outline fits; 0 when the view is too flat to tell) and age (s since the last fine fit)."""

    FLOW_REGION = (0.25, 0.08, 0.79, 0.90)   # the part of the screen the optical flow may use (fractions x0, y0, x1, y1): between the left panel and the legend
    LOOK_EVERY = 0.4           # s between fine fits
    GIVE_UP_S = 3.0            # lost for this long (map closed / covered) -> stop
    NEVER_S = 8.0              # never found the map in this long -> stop (M pressed in chat, or a menu covers it)
    LOST_S = 1.0               # the outline stops fitting for this long -> not the map any more
    LOST_FLAT_S = 2.5          # ... and nothing to fit or follow (zoomed in on all land, say): the map may well just be standing still, give it longer
    RECOVER = 0.60             # a wide look around the tracked map that fits this well takes it back (a quick zoom outruns the flow) instead of searching from scratch
    SURE, SURE_ZOOM = 0.85, 0.8        # a map found from scratch with a weaker fit, or zoomed in this far (screen px per z4 px, at 1125 px tall), must be found twice in a row
    CONFIRM_S = 2.0
    LEAD_S = 0.03              # the overlay shows a frame this long after state() is read (compositor + scan-out): aim there
    MAX_AHEAD_S = 0.12         # never extrapolate further than this past the last capture
    DUMP_S = 4.0
    SETTLE_S = 0.25            # after the last pan / zoom the camera counts as still (fine fits may correct the transform again)
    SMOOTH = 0.6               # weight of the newest frame-to-frame motion in the velocity estimate

    def __init__(self, make_grab, out_wh, mask=None, finder=None, log=lambda m: None):
        self.make_grab, self.out_wh, self.mask, self.log = make_grab, out_wh, mask, log
        self.finder = finder
        self.lock = threading.Lock()
        self.thread = None
        self.cur = None
        self.stop_ev = threading.Event()
        self.method = "hue"
        self.t_start = 0.0
        self.seen = False                            # found at least once since start()
        self.pend = None                             # (T, score, t): a registration waiting for a second look that agrees
        self.dump = None                             # dump(frame): called once when the map has not been found for DUMP_S (so a failure can be looked at)
        self._dumped = False
        self.frame_t = 0.0                           # when the screen was last sampled
        self.vel = None                              # (A, dt): the smoothed camera motion per dt seconds, screen px of the overlay

    @property
    def searching_s(self):
        """Seconds this watch has been looking for the map without having it (0 when it has it or is not running)."""
        if not self.active or self.cur is not None:
            return 0.0
        return time.time() - self.t_start

    @property
    def holding(self):
        """Is the full map on screen as far as we can tell: it is being followed, or it has not been found yet (M was just pressed). Once it was
        found and then lost (closed) this is False at once, without waiting for the watch to give up."""
        return self.active and not self.stop_ev.is_set() and (self.cur is not None or not self.seen)

    @property
    def active(self):
        return self.thread is not None and self.thread.is_alive()

    def start(self):
        if self.active:
            if not self.stop_ev.is_set():
                return False
            self.thread.join(1.0)                         # it was just stopped and is finishing its last look
            if self.active:
                return False
        self.stop_ev.clear()
        self.t_start = time.time()
        self.seen = self._dumped = False
        self.pend = None
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()
        return True

    def stop(self):
        self.stop_ev.set()
        with self.lock:
            self.cur = None
            self.vel = None

    def state(self):
        """M is moved on from the last capture by the camera's smoothed motion, to the moment the overlay is shown (it follows a zoom / pan
        instead of trailing behind it)."""
        with self.lock:
            cur, vel = self.cur, self.vel
        if cur is None:
            return None
        now = time.time()
        age = now - cur["t"]                                      # since the last fine fit
        M = cur["M"]
        if vel is not None and now - cur["tf"] < 0.3:
            A, dt = vel
            ahead = min(now - cur["tf"] + self.LEAD_S, self.MAX_AHEAD_S)
            M = compose(_power(A, ahead / dt), M)
        return dict(cur, M=M, age=age)

    def _run(self):
        try:
            self.grab = self.make_grab()
            if self.finder is None:
                self.finder = MapFinder()
            self._loop()
        except Exception as e:                       # a thread must not take the app down
            self.log("full map: %s" % e)
        finally:
            with self.lock:
                self.cur = None

    def _flow_frame(self, frame):
        g = cv2.cvtColor(frame, cv2.COLOR_BGRA2GRAY if frame.shape[2] == 4 else cv2.COLOR_BGR2GRAY)
        f = WORK_W / g.shape[1] if g.shape[1] > WORK_W else 1.0
        if f != 1.0:
            g = cv2.resize(g, (int(round(g.shape[1] * f)), int(round(g.shape[0] * f))), interpolation=cv2.INTER_AREA)
        m = self.mask() if self.mask else None
        if m is not None and m.shape != g.shape:
            m = cv2.resize(m, (g.shape[1], g.shape[0]), interpolation=cv2.INTER_NEAREST)
        if self.FLOW_REGION is not None:                  # the game's own panels (stats, legend, buttons) stand still while the map moves: they are all corners and would win the vote
            h, w = g.shape
            x0, y0, x1, y1 = self.FLOW_REGION
            reg = np.zeros(g.shape, np.uint8)
            reg[int(h * y0):int(h * y1), int(w * x0):int(w * x1)] = 255
            m = reg if m is None else cv2.bitwise_and(m, reg)
        return g, f, m

    def _publish(self, T, score, t, W, H):
        o = np.array([[self.out_wh[0] / W, 0.0, 0.0], [0.0, self.out_wh[1] / H, 0.0]])
        with self.lock:
            self.cur = dict(M=compose(o, T), score=score, t=t, tf=self.frame_t)

    def _motion(self, D, f, dt, W, H):
        """Fold one frame-to-frame motion D (small frame px) into the smoothed velocity; None / a still frame decays it."""
        with self.lock:
            if D is None or dt <= 0:
                self.vel = None if D is None else self.vel
                return
            o = np.array([[self.out_wh[0] / W, 0.0, 0.0], [0.0, self.out_wh[1] / H, 0.0]])
            S = np.array([[1 / f, 0, 0], [0, 1 / f, 0]])
            A = compose(o, S, D, np.array([[f, 0, 0], [0, f, 0]]), cv2.invertAffineTransform(o))         # the same motion in overlay px
            if self.vel is None:
                self.vel = (A, dt)
                return
            A0, dt0 = self.vel
            k = self.SMOOTH
            P = _power(A0, dt / dt0)                                                   # the old estimate over this frame's dt
            self.vel = ((1 - k) * P + k * A, dt)

    def _confirmed(self, reg, now, H):
        """A map found from scratch is believed at once when the fit is strong and the zoom ordinary; a weaker fit or a big zoom (a small
        patch of coast can look like many places) must be found again, in the same place, on a later frame."""
        T, score, _ = reg
        if score >= self.SURE and scale_of(T) <= self.SURE_ZOOM * H / 1125.0:
            self.pend = None
            return True
        p = self.pend
        if p is not None and now - p[2] <= self.CONFIRM_S and abs(np.log(scale_of(T) / scale_of(p[0]))) < 0.04                 and float(np.hypot(*(T[:, 2] - p[0][:, 2]))) < 12.0 * H / 1125.0:
            self.pend = None
            return True
        self.pend = (T, score, now)
        return False

    def _loop(self):
        t0 = last_ok = last_look = time.time()
        T = prev = prev_m = None         # T: z4 px -> frame px
        found, bad_since, prev_t, moving_until = False, 0.0, 0.0, 0.0
        while not self.stop_ev.is_set():
            frame = self.grab()
            now = self.frame_t = time.time()                    # the moment the screen was sampled
            if frame.shape[2] == 4:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
            H, W = frame.shape[:2]
            g, f, m = self._flow_frame(frame)
            m = grow(m)
            moved = False
            if T is None:                                     # not locked: look for the map from scratch
                if not found and not self._dumped and now - t0 > self.DUMP_S and self.dump is not None:
                    self._dumped = True
                    try:
                        self.dump(frame)
                    except Exception as e:
                        self.log("full map: dump failed (%s)" % e)
                reg = self.finder.register(frame)
                if reg is not None and not self._confirmed(reg, now, H):
                    reg = None
                if reg is not None:
                    T, score, self.method = reg
                    self.seen = True
                    found, last_ok, last_look, bad_since = True, now, now, 0.0
                    self.log("full map found (fit %.2f, zoom %.3f)" % (score, scale_of(T)))
                    self._publish(T, score, now, W, H)
                elif now - last_ok > self.GIVE_UP_S and (found or now - t0 > self.NEVER_S):
                    self.log("full map: %s" % ("closed" if found else "not found in %.0f s" % (now - t0)))
                    break
            else:
                D = track(prev, g, prev_m, m) if prev is not None else None
                self._motion(D, f, now - prev_t, W, H)
                if D is not None:                              # follow the pan / zoom (flow works in the small frame)
                    S = np.array([[1 / f, 0, 0], [0, 1 / f, 0]])
                    T = compose(S, D, np.array([[f, 0, 0], [0, f, 0]]), T)
                    moved = True
                    if abs(scale_of(D) - 1.0) > 0.0015 or float(np.hypot(D[0, 2], D[1, 2])) > 0.6:
                        moving_until = now + self.SETTLE_S
                if now - last_look >= self.LOOK_EVERY:         # fine fit: removes drift, and says whether this is still the map
                    last_look = now
                    r = self.finder.refine(frame, scale_of(T), T[0, 2], T[1, 2], ds=0.03, m=14, steps=5, method=self.method)
                    rec = None
                    if r is not None and r[0] < KEEP:          # it does not fit: look wider before believing the map is gone (a quick zoom outruns the flow)
                        rec = self.finder.refine(frame, scale_of(T), T[0, 2], T[1, 2], ds=0.15, m=40, steps=13, method=self.method)
                        if rec is not None and rec[0] >= self.RECOVER:
                            T, r = matrix(*rec[1:]), rec
                        else:
                            rec = None
                    if r is not None and r[0] >= KEEP:
                        if rec is None and now >= moving_until:  # still: the outline fit removes the flow's drift
                            T = matrix(*r[1:])
                        last_ok, bad_since = now, 0.0          # moving: the flow is more exact than a fit on a half-zoomed frame; the fit only checks it is still the map
                        self._publish(T, r[0], now, W, H)
                    elif r is not None:
                        bad_since = bad_since or now
                        if now - bad_since > self.LOST_S:
                            self.log("full map: lost it (fit %.2f)" % r[0])
                            T = None
                            with self.lock:
                                self.cur = None
                    elif moved:                                # too flat to fit (all land): trust the flow
                        last_ok = now
                        self._publish(T, 0.0, now, W, H)
                    elif now - last_ok > self.LOST_FLAT_S:
                        self.log("full map: lost it (nothing to follow)")
                        T = None
                        with self.lock:
                            self.cur = None
                elif moved:
                    c = self.cur
                    if c is not None:                          # between fine fits: the same fit, moved by the flow
                        self._publish(T, c["score"], c["t"], W, H)
            if T is None:
                with self.lock:
                    self.vel = None
            prev, prev_m, prev_t = g, m, now
            time.sleep(0.002)
