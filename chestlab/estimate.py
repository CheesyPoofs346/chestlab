"""Estimators: where am I on the island (silhouette fit), which way am I facing (arrow), smoothing, horizon pitch."""
import math
from typing import NamedTuple

import cv2
import numpy as np

from .geometry import wrap180
from .render import draw_arrow


# Under the storm the minimap is  seen = STORM_GAIN * original + tint  (colour model fitted to real storm frames; the violet
# tint changes with the storm's phase, so it is read from each frame). SEA_BGR: deep-sea colour of a clear minimap, the
# median over 11 real frames; the sea dominates the storm pixels, so their median minus the gained sea colour is the tint.
STORM_GAIN = 0.4
SEA_BGR = np.array([151, 114, 25], np.float32)         # ... median colour of the whole minimap
SEA_MODE_BGR = np.array([140, 93, 17], np.float32)     # ... dominant (most common) colour of the whole minimap

ECC_CROP = True

# Minimap scale measured on 10 real frames: per-frame optimum wobbles 0.0680-0.0705 (the island is too small to pin it
# down from one frame), median 0.0698. The game's minimap zoom is constant, so snap to it.
REAL_ZOOM = 0.0698


def _sig(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def classify(bgr):
    """Soft land/water map: ~+1 land, ~-1 water, ~0 unknown (storm tint, white overlays, pins). Sigmoid thresholds on
    hue/saturation/value instead of hard cut-offs: hard ones jump at the island edge and cap sub-pixel accuracy.
    Returns (obs, known)."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    H, S, V = (hsv[..., i].astype(np.float32) for i in range(3))
    land = _sig((H - 5) / 2) * _sig((80 - H) / 2) * _sig((S - 35) / 8) * _sig((V - 70) / 8)
    water = _sig((H - 82) / 2) * _sig((112 - H) / 2) * _sig((S - 35) / 8) * _sig((V - 60) / 8)
    return (land - water).astype(np.float32), (land + water) > 0.5


_E7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
_E5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))


def storm_mask(bgr):
    """Pixels under the storm: violet/magenta/pink. Thin lines (the blue route line) are opened away."""
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    m = ((hsv[..., 0] >= 118) & (hsv[..., 0] <= 178) & (hsv[..., 1] >= 55)).astype(np.uint8)
    return cv2.morphologyEx(m, cv2.MORPH_OPEN, _E5) > 0


def dominant_color(px, q=6):
    """Most common colour of an (n, 3) pixel set (histogram peak, refined by the median of its neighbours): unlike the
    median it is not pulled towards the island or the shallows."""
    px = px.reshape(-1, 3)
    k = (px // q).astype(np.int32)
    peak = np.bincount(k[:, 0] * 2500 + k[:, 1] * 50 + k[:, 2]).argmax()
    centre = np.array([peak // 2500, (peak // 50) % 50, peak % 50], np.float32) * q + q / 2
    return np.median(px[np.abs(px - centre).max(axis=1) <= 1.5 * q], axis=0).astype(np.float32)


def destorm(bgr, storm, robust=True):
    """Undo the storm tint inside `storm`, recovering the colours the minimap would show without it, so the ordinary
    classifier (and its sub-pixel edges) works there too. Outside the storm the image is untouched. The tint is the storm
    sea's colour minus the gained clear-sea colour. robust: use the dominant colour (near exact, but it can lock onto
    the wrong water when the storm only covers e.g. the shallows) instead of the median (always roughly right)."""
    sea = dominant_color(bgr[storm]) - STORM_GAIN * SEA_MODE_BGR if robust else np.median(bgr[storm], axis=0) - STORM_GAIN * SEA_BGR
    est = np.clip((bgr.astype(np.float32) - sea) / STORM_GAIN, 0, 255)
    return np.where(storm[..., None], est, bgr).astype(np.uint8)


def classify_mixed(bgr, robust=True):
    """classify() for a minimap that is partly or wholly under the storm: de-storm first. A band along the storm's edge
    (a blend of both) is left unknown. Returns (obs, known)."""
    storm = storm_mask(bgr)
    if storm.mean() < 0.02:
        return classify(bgr)
    obs, known = classify(destorm(bgr, storm, robust))
    s8 = storm.astype(np.uint8)
    edge = (cv2.dilate(s8, _E7) - cv2.erode(s8, _E7)) > 0
    return obs, known & ~edge


class Fit(NamedTuple):
    u: float          # player position on the island map (pixels)
    v: float
    score: float      # coarse land/water agreement, 1.0 = perfect
    cc: float         # ECC correlation after sub-pixel refinement (nan if refinement failed)
    zoom: float
    tx: float         # template origin inside the minimap crop (pixels)
    ty: float
    adaptive: bool = False   # fitted with the storm-proof classifier (less precise)


class SilhouetteFitter:
    """Registers the island's land/water silhouette against a player-centred minimap crop: coarse masked
    cross-correlation, then ECC translation for sub-pixel position. The player is the crop centre, so the fitted
    template origin gives the player's map position directly."""

    def __init__(self, world, size=231, min_score=0.80):
        self.world, self.size, self.c, self.min_score = world, size, (size - 1) / 2, min_score
        self.full, _ = classify(world.img)   # full-resolution map, used for the land/beach mask
        self.zoom = REAL_ZOOM   # constant in the game; only re-scanned after repeated failures
        self.fails = 0
        self.adaptive = False   # which classifier the last good fit used
        self.last_score = 0.0   # best coarse match of the most recent attempt (for diagnostics)
        self._tpl = {}

    def template(self, zoom):
        key = round(zoom, 5)
        if key not in self._tpl:
            if len(self._tpl) > 64:
                self._tpl.clear()
            # classify the *resized colour image*, exactly as the observation is processed: classifying first and
            # shrinking after gives a different edge position and a ~1.4 m error floor
            small, zx, zy = self.world.small(zoom)
            t, _ = classify(small)
            self._tpl[key] = (cv2.GaussianBlur(t, (0, 0), 0.8), zx, zy)
        return self._tpl[key]

    def _origin_from_uv(self, uv, zoom):
        _, zx, zy = self.template(zoom)
        return self.c - ((uv[0] + 0.5) * zx - 0.5), self.c - ((uv[1] + 0.5) * zy - 0.5)

    def _uv_from_origin(self, tx, ty, zoom):
        _, zx, zy = self.template(zoom)
        return (self.c - tx + 0.5) / zx - 0.5, (self.c - ty + 0.5) / zy - 0.5

    def _coarse(self, obs, known, zoom, around=None, radius=10):
        tpl, _, _ = self.template(zoom)
        th, tw = tpl.shape
        op = cv2.copyMakeBorder(obs, th, th, tw, tw, cv2.BORDER_CONSTANT, value=0)
        kp = cv2.copyMakeBorder(known.astype(np.float32), th, th, tw, tw, cv2.BORDER_CONSTANT, value=0)
        S = cv2.matchTemplate(op, tpl, cv2.TM_CCORR)
        N = cv2.matchTemplate(kp, np.ones_like(tpl), cv2.TM_CCORR)
        score = S / np.maximum(N, 1.0)
        score[N < 0.6 * tw * th] = -1.0
        if around is not None:
            iy, ix = np.mgrid[0:score.shape[0], 0:score.shape[1]]
            score[(ix - tw - around[0]) ** 2 + (iy - th - around[1]) ** 2 > radius ** 2] = -1.0
        i = np.unravel_index(np.argmax(score), score.shape)
        return float(score[i]), int(i[1]) - tw, int(i[0]) - th

    def _refine(self, obs, known, zoom, tx0, ty0):
        """ECC translation refinement of the integer coarse origin -> (tx, ty, cc)."""
        tpl, _, _ = self.template(zoom)
        th, tw = tpl.shape
        s = self.size
        canvas, win = np.full((s, s), -1.0, np.float32), np.zeros((s, s), np.uint8)
        xs0, ys0, xs1, ys1 = max(tx0, 0), max(ty0, 0), min(tx0 + tw, s), min(ty0 + th, s)
        if xs1 - xs0 < 8 or ys1 - ys0 < 8:
            return float(tx0), float(ty0), float("nan")
        canvas[ys0:ys1, xs0:xs1] = tpl[ys0 - ty0:ys1 - ty0, xs0 - tx0:xs1 - tx0]
        win[ys0:ys1, xs0:xs1] = 1
        mask = (known & (win > 0)).astype(np.uint8)
        warp = np.eye(2, 3, dtype=np.float32)
        # ECC only needs the region the template covers (+ a margin): ~9x fewer pixels than the whole crop
        m = 8 if ECC_CROP else s
        x0, y0, x1, y1 = max(xs0 - m, 0), max(ys0 - m, 0), min(xs1 + m, s), min(ys1 + m, s)
        canvas, mask, op = canvas[y0:y1, x0:x1], mask[y0:y1, x0:x1], cv2.GaussianBlur(obs, (0, 0), 1.0)[y0:y1, x0:x1]
        try:
            cc, warp = cv2.findTransformECC(canvas, op, warp, cv2.MOTION_TRANSLATION,
                                            (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 80, 1e-5), mask, 3)
        except cv2.error:
            return float(tx0), float(ty0), float("nan")
        if abs(warp[0, 2]) > 6 or abs(warp[1, 2]) > 6:  # a refinement should only ever nudge
            return float(tx0), float(ty0), float("nan")
        return tx0 + float(warp[0, 2]), ty0 + float(warp[1, 2]), float(cc)

    # Storm mode ("adaptive"): classify_mixed. A little rougher than the normal mode, so it must pass stricter checks.
    ADAPTIVE_MIN_SCORE, ADAPTIVE_MIN_CC = 0.80, 0.90

    # Classifier variants, in the order they are tried: plain (exact when enough of the island is clear), then the
    # storm-aware one with the dominant-colour tint, then with the median tint. (storm?, robust tint?)
    MODES = ((False, True), (True, True), (True, False))

    def _observe(self, minimap, mode):
        adaptive, robust = mode
        return classify_mixed(minimap, robust) if adaptive else classify(minimap)

    def acquire(self, minimap):
        """First look: scan zooms (unknown minimap scale), keep the best. Returns Fit or None."""
        self.last_score = 0.0
        for mode in self.MODES:
            adaptive = mode[0]
            obs, known = self._observe(minimap, mode)
            lo, hi = REAL_ZOOM * 0.96, REAL_ZOOM * 1.04   # the zoom is constant in-game; a far-off 'best' zoom is a false match
            best = max(((*self._coarse(obs, known, z), z) for z in np.geomspace(lo, hi, 28)), key=lambda r: r[0])
            z0 = best[3]
            best = max(((*self._coarse(obs, known, z), z) for z in z0 * np.linspace(0.96, 1.04, 9)), key=lambda r: r[0])
            self.last_score = max(self.last_score, best[0])
            if best[0] < (self.ADAPTIVE_MIN_SCORE if adaptive else self.min_score):
                continue
            old = self.zoom
            z = best[3]
            self.zoom, self.adaptive = REAL_ZOOM, adaptive   # within +-4% of it by construction: snap
            fit = self._fit_mode(minimap, mode, None)
            if fit is not None:
                self.fails = 0
                return fit
            self.zoom = old
        return None

    def _fit_mode(self, minimap, mode, guess_uv):
        adaptive = mode[0]
        obs, known = self._observe(minimap, mode)
        if adaptive and (storm_mask(minimap).mean() < 0.02 or known.mean() < 0.4):
            return None   # mixed mode only makes sense with a visible storm and enough classified pixels
        around = self._origin_from_uv(guess_uv, self.zoom) if guess_uv is not None else None
        score, tx0, ty0 = self._coarse(obs, known, self.zoom, around)
        self.last_score = max(self.last_score, score)
        if score < (self.ADAPTIVE_MIN_SCORE if adaptive else self.min_score):
            return None
        tx, ty, cc = self._refine(obs, known, self.zoom, tx0, ty0)
        if adaptive and (cc < self.ADAPTIVE_MIN_CC if cc == cc else score < 0.85):
            return None   # storm mode: ECC must agree, or (if ECC could not converge) the coarse match must be very good
        u, v = self._uv_from_origin(tx, ty, self.zoom)
        return Fit(u, v, score, cc, self.zoom, tx, ty, adaptive)

    def fit(self, minimap, guess_uv=None):
        if self.zoom is None:
            return self.acquire(minimap)
        self.last_score = 0.0
        for mode in self.MODES:
            fit = self._fit_mode(minimap, mode, guess_uv)
            if fit is not None:
                self.adaptive, self.fails = fit.adaptive, 0
                return fit
        self.fails += 1
        if self.fails % 8 == 0:   # stuck: the zoom may be wrong (or the minimap size changed) -> scan again
            self.zoom = REAL_ZOOM
            return self.acquire(minimap)
        return None


class ArrowHeading:
    """Facing direction from the player arrow at the minimap centre: rotating signed-template correlation (white
    fill +1, dark outline -1), parabola-refined to sub-degree.

    Each template is made by *drawing* the arrow and running it through the very same extraction (colour
    thresholds + opening) that the observation goes through. Building templates from the ideal polygon instead
    leaves them ~1-2 px fatter than what the extractor sees and caps the match score near 0.45."""

    def __init__(self, size=17.0, notch=0.35, patch=41, step=1):
        self.patch, self.step = patch, step
        self.k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        self.angles = np.arange(0, 360, step, dtype=float)
        c = (patch - 1) / 2
        T = []
        for a in self.angles:
            canvas = np.full((patch, patch, 3), (190, 160, 30), np.uint8)   # teal water behind the marker
            draw_arrow(canvas, c, c, a, size, notch)
            _, v = self._extract(canvas)
            T.append(v / (np.linalg.norm(v) + 1e-9))
        self.T = np.array(T)

    def _extract(self, patch_bgr):
        hsv = cv2.cvtColor(patch_bgr, cv2.COLOR_BGR2HSV)
        white = cv2.morphologyEx(((hsv[..., 1] < 60) & (hsv[..., 2] > 185)).astype(np.uint8), cv2.MORPH_OPEN, self.k)
        dark = (hsv[..., 2] < 90).astype(np.float32)
        return white, (white.astype(np.float32) - dark).ravel()

    def estimate(self, minimap):
        s = minimap.shape[0]
        c, h = (s - 1) // 2, self.patch // 2
        white, o = self._extract(minimap[c - h:c + h + 1, c - h:c + h + 1])
        n = np.linalg.norm(o)
        if white.sum() < 8 or n < 1e-6:
            return None, 0.0
        sc = self.T @ (o / n)
        i = int(np.argmax(sc))
        a, b, d = sc[i - 1], sc[i], sc[(i + 1) % len(sc)]
        frac = 0.5 * (a - d) / (a - 2 * b + d) if (a - 2 * b + d) < -1e-9 else 0.0
        return float((self.angles[i] + frac * self.step) % 360), float(b)


class PoseFilter:
    """Constant-velocity Kalman filter on map position + alpha-beta filter on heading (with angle wrap)."""
    CALM_S = 0.5     # seconds of standing still before the filter locks down

    def __init__(self, q_acc=60.0, r_pos=8.0, alpha=0.35, beta=0.08):   # r_pos: map px (1 px = 0.085 m); clear-sky fit noise ~ 8, storm ~ 22
        self.q, self.r, self.alpha, self.beta = q_acc, r_pos, alpha, beta
        self.x = None
        self.th = self.om = None
        self.calm = 0.0      # seconds the player has been standing still (fixes agree with the estimate, speed ~ 0)

    def reset(self):
        self.x = None
        self.th = None
        self.calm = 0.0

    def predict(self, dt):
        if self.x is None:
            return
        F = np.eye(4); F[0, 2] = F[1, 3] = dt
        G = np.array([[dt * dt / 2, 0], [0, dt * dt / 2], [dt, 0], [0, dt]])
        q = self.q * (0.08 if self.calm > self.CALM_S else 1.0)    # standing still: trust the estimate, don't wander with the noise
        if self.calm > self.CALM_S:
            self.x[2:] *= 0.5 ** (dt / 0.15)               # and let any leftover drift velocity die away
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + G @ G.T * q ** 2
        if self.th is not None:
            self.th = (self.th + self.om * dt) % 360

    def update(self, u, v, heading, dt, r=None):
        r = self.r if r is None else r
        if self.x is None:
            self.x, self.P = np.array([u, v, 0.0, 0.0]), np.diag([r * r, r * r, 40.0 ** 2, 40.0 ** 2])
        else:
            H = np.eye(2, 4)
            K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + np.eye(2) * r * r)
            inn = np.array([u, v]) - H @ self.x
            if float(np.hypot(*inn)) < 1.3 * r and float(np.hypot(*self.x[2:])) < 8.0:
                self.calm += dt
            else:
                self.calm = 0.0
            self.x = self.x + K @ inn
            self.P = (np.eye(4) - K @ H) @ self.P
        if heading is not None:
            if self.th is None:
                self.th, self.om = heading, 0.0
            else:
                e = float(wrap180(heading - self.th))
                self.th = (self.th + self.alpha * e) % 360
                self.om += self.beta * e / dt

    @property
    def pos(self):
        return None if self.x is None else (float(self.x[0]), float(self.x[1]))


class Tracker:
    """Whole pipeline: minimap crop in, (position, heading, on_island) out."""

    def __init__(self, world, min_score=0.80, beach_margin=25, fitter=None):
        self.world = world
        self.fitter = fitter or SilhouetteFitter(world, min_score=min_score)   # fitter: another map source (worldmap.WorldFitter)
        self.arrow = ArrowHeading()
        self.filt = PoseFilter()
        self.misses = 0
        self.miss_s = 0.0
        self.vclock, self.valt = 0.0, None
        self.cand = None     # (u, v, count): a fix far from the estimate, waiting for the next ones to agree
        self.land = None
        if hasattr(self.fitter, "full"):
            land = (self.fitter.full > 0).astype(np.uint8)
            k = 2 * beach_margin + 1
            self.land = cv2.dilate(land, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))) > 0

    def on_land(self, u, v):
        """Is this map position on the island (land plus a beach margin)?"""
        i, j = int(round(v)), int(round(u))
        return 0 <= i < self.land.shape[0] and 0 <= j < self.land.shape[1] and bool(self.land[i, j])

    COAST_FRAMES, RESET_FRAMES = 20, 45      # missed fits: keep predicting for ~0.7 s, forget after ~1.5 s
    COAST_S, RESET_S = 2.5, 5.0              # ... but when the sensor runs slowly (a full-map search per frame) it is time that counts, not frames
    COAST_STORM_S, RESET_STORM_S = 4.0, 9.0  # a storm over the minimap hides the map for a while: keep predicting longer (the fixes come back when it clears or the view changes)
    STORM_FRAC = 0.15
    VERIFY_S = 1.5                           # with a lock held, look at the whole map this often (fitters that have verify()); two looks that agree move the lock
    GATE_PX, CONFIRM = 50.0, 3               # a fix > 50 px (~4 m) from the estimate must be repeated 3x to count

    def step(self, minimap, dt=1 / 30, heading=None):
        """heading: compass bearing from the HUD if known; otherwise estimated from the minimap arrow.
        -> dict(locked, on_island, raw, pos, heading, fit, coasting). `raw` is set only for a fix that was used."""
        self.filt.predict(dt)
        guess = self.filt.pos if self.misses < 5 else None
        fit = self.fitter.fit(minimap, guess_uv=guess)
        if fit is None and guess is not None:
            fit = self.fitter.fit(minimap)  # lost the local lock: full search
        if heading is None:
            heading, _ = self.arrow.estimate(minimap)
        pos = self.filt.pos
        if fit is not None and pos is not None:
            gate = max(self.GATE_PX, 6.0 * float(np.sqrt(self.filt.P[0, 0])) + 4.0 * self.filt.r)
            if np.hypot(fit.u - pos[0], fit.v - pos[1]) > gate:
                c = self.cand
                if c is not None and np.hypot(fit.u - c[0], fit.v - c[1]) < 25.0:
                    self.cand = (fit.u, fit.v, c[2] + 1)
                else:
                    self.cand = (fit.u, fit.v, 1)
                if self.cand[2] >= self.CONFIRM:          # it really did move (respawn, teleport, vehicle): take it
                    self.filt.reset()
                    self.cand = None
                else:
                    fit = None                            # lone outlier: ignore this fix
            else:
                self.cand = None
        if fit is not None and hasattr(self.fitter, "verify"):
            self.vclock += dt
            if self.vclock >= self.VERIFY_S:
                self.vclock = 0.0
                alt = self.fitter.verify(minimap, (fit.u, fit.v), fit.score)
                if alt is None:
                    self.valt = None
                elif self.valt is not None and np.hypot(alt.u - self.valt[0], alt.v - self.valt[1]) < 40.0:
                    self.filt.reset()                       # two looks agree: the lock was on the wrong spot
                    self.valt, fit, self.cand = None, alt, None
                else:
                    self.valt = (alt.u, alt.v)
        if fit is None:
            self.misses += 1
            self.miss_s += dt
            stormy = float(storm_mask(minimap).mean()) >= self.STORM_FRAC
            if self.misses > self.RESET_FRAMES and self.miss_s > (self.RESET_STORM_S if stormy else self.RESET_S):
                self.filt.reset()
            coast = (self.misses <= self.COAST_FRAMES or self.miss_s <= (self.COAST_STORM_S if stormy else self.COAST_S)) and self.filt.pos is not None
            return {"locked": coast, "coasting": coast, "on_island": coast and self.on_land(*self.filt.pos), "raw": None,
                    "pos": self.filt.pos if coast else None, "heading": self.filt.th if coast else None}
        self.misses = 0
        self.miss_s = 0.0
        self.filt.update(fit.u, fit.v, heading, dt, r=self.filt.r * (1.6 if fit.adaptive else 1.0))   # storm fixes: a little noisier (p90 ~1.9 m)
        return {"locked": True, "coasting": False, "on_island": self.on_land(*self.filt.pos), "raw": (fit.u, fit.v),
                "pos": self.filt.pos, "heading": self.filt.th, "raw_heading": heading, "fit": fit}


class HorizonPitch:
    """Pitch from the strongest horizontal sky/ground edge (works on the simulator's clean sky; real game frames
    with clouds, trees and rocks are untested)."""

    def estimate(self, view, cam):
        g = cv2.cvtColor(view, cv2.COLOR_BGR2GRAY).astype(np.float32).mean(axis=1)
        g = np.convolve(g, np.ones(5) / 5, mode="same")
        grad = np.abs(np.diff(g))
        grad[:3] = grad[-3:] = 0
        row = float(np.argmax(grad)) + 0.5
        return math.degrees(math.atan((row - cam.h / 2) / cam.f))
