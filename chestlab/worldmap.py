"""The whole battle-royale map: localise the player on it from the minimap, and every chest on it (fortnite.gg data).

Coordinates stay in the island's map-pixel units (uv; 1 px = 0.0875 m, see World), so every existing piece (Kalman filter,
chests, HUD, the island's own map) keeps its meaning: fortnite.gg map unit g=(x right, y down) <-> uv = 100*g + (-19281, -7376)
(fitted on the island's 19 buried-chest icons; island.png registers onto the fortnite.gg tiles to 1 px at that transform).

Localising: the minimap is a small colour render of the same terrain as the fortnite.gg tiles (a different grade), so it is
matched in Lab colour space (zero-mean normalised cross-correlation per channel: immune to brightness / contrast / the storm's
gain+offset), with HUD overlays (route line, pins, arrows, icons) and the minority of a half-closed storm masked out.
First look = global search on a coarse level (z2), verified on z3; after that a small window around the last position.
On the island the precise silhouette fitter takes over when it agrees."""
import functools
import json
import os

import cv2
import numpy as np

from .paths import BASE
from .estimate import REAL_ZOOM, Fit, SilhouetteFitter, Tracker, storm_mask

ASSETS = os.path.join(BASE, "assets")
GG = os.path.join(ASSETS, "gg")
ORIGIN = np.array([-19281.0, -7376.0])         # uv of the fortnite.gg map corner
UV_PER_UNIT = 100.0                           # uv px per fortnite.gg map unit
KINDS = ("buried", "chest_sure", "rare", "sprite", "vault", "chest")     # world_chests.json order of draw priority
Z_REAL = 1.0 / (REAL_ZOOM * UV_PER_UNIT)                       # map units per minimap px  (x 2**level = tile px per minimap px)


def uv_to_px(uv, level):
    """uv -> pixels of the tile level `level` (zoom: 2**level px per map unit)."""
    return (np.asarray(uv, float) - ORIGIN) / UV_PER_UNIT * 2 ** level


def px_to_uv(px, level):
    return np.asarray(px, float) / 2 ** level * UV_PER_UNIT + ORIGIN


def lab8(bgr, sigma):
    """Blurred Lab, 8-bit per channel: the map descriptor."""
    lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2Lab)
    if sigma:
        lab = cv2.GaussianBlur(lab, (0, 0), sigma)
    return [np.ascontiguousarray(lab[..., i]) for i in range(3)]


class WorldMap:
    """The tiles (assets/gg/full_z4.png, stitched by tools/build_world.py), their descriptors at two coarse levels, and the
    chest list."""
    LEVELS = (2, 3)

    def __init__(self, path=None):
        img = cv2.imread(path or os.path.join(GG, "full_z4.png"), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError("assets/gg/full_z4.png missing: run tools/build_world.py")
        self.valid4 = img.max(axis=2) > 6                                    # tiles beyond the playable area are black
        self.desc = {}
        self.valid = {}
        for lv in self.LEVELS:
            f = 2.0 ** (lv - 4)
            small = cv2.resize(img, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
            self.desc[lv] = lab8(small, 0.75 * 2 ** (lv - 3))
            self.valid[lv] = cv2.resize(self.valid4.astype(np.uint8), (small.shape[1], small.shape[0]), interpolation=cv2.INTER_AREA) > 0
        self.shape4 = img.shape[:2]
        self.chests, self.kind = self._load_chests()

    @staticmethod
    def _load_chests():
        with open(os.path.join(ASSETS, "world_chests.json"), encoding="utf-8") as f:
            data = json.load(f)
        pts, kind = [], []
        for k in KINDS:
            for x, y in data.get(k, []):
                pts.append(px_to_uv([x, y], 4))
                kind.append(k)
        return np.array(pts, float), np.array(kind)

    def on_map(self, u, v, margin_m=0.0):
        """Is uv inside the map's playable (tiled) area?"""
        x, y = uv_to_px([u, v], 4)
        i, j = int(round(y)), int(round(x))
        return 0 <= i < self.shape4[0] and 0 <= j < self.shape4[1] and bool(self.valid4[i, j])


@functools.lru_cache(maxsize=1)
def world_map():
    """The one WorldMap of this process (loading + descriptors take ~0.7 s and ~60 MB)."""
    return WorldMap()


def available():
    return os.path.exists(os.path.join(GG, "full_z4.png")) and os.path.exists(os.path.join(ASSETS, "world_chests.json"))


class WorldFitter:
    """Drop-in for SilhouetteFitter (fit / acquire / zoom / last_score / adaptive) over the whole map."""
    TRACK_LEVEL, ACQ_LEVEL = 3, 2
    ROWS, COLS = (0.22, 0.93), (0.07, 0.93)     # the part of the minimap crop that is map (the rest: HUD icons, frame)
    MIN_SCORE, MIN_ACQ, MARGIN = 0.50, 0.54, 0.04
    LOCAL_TRIM_RAW, LOCAL_TRIM = 0.36, 0.66          # the same, while tracking (a lock is already held)
    TRIM_RAW, TRIM_ACQ = 0.48, 0.74                 # a match this weak is still taken when, with the worst-fitting fifth of the pixels left out, it is this good
    BIAS_PX = (-2.4, -6.9)                       # z4 px: this matcher minus the silhouette fitter over 12 clear real frames (the minimap draws
                                                 # terrain shadows, so its appearance sits ~3.7 m down-right of the silhouette); sd 0.5 m
    GUIDED_MIN, GUIDED_PX = 0.40, 4.0            # while tracking: a weaker match still counts when its peak sits within this many tile px of where the tracker predicted (place names, pings, pins cover the map)
    TEXT_PX, TEXT_PX_WIDE = 1, 9                 # white map text / icons: normal frames keep a thin mask (accuracy), a frame that found nothing and is full of text is tried again with a wide one                                  # dilation (minimap px) of the white-text mask: the outline of a place name sits within ~3 px of it
    ZSCALES = (0.95, 1.0, 0.90)                  # the minimap is not always drawn at exactly the same size (resized screenshots, other HUD scales)
    STORM_MODES = (True, False)                  # mask the storm (violet) or not: pink / purple trees look like storm to the colour test
    VARIANT_GAIN = 0.04                          # ...it only replaces the remembered variant when it scores this much higher
    GOOD_ACQ = 0.60                              # the remembered (zoom, storm) variant scoring this high needs no other look
    ISLAND_ZONE_PX = 600.0                       # uv distance from island.png within which the silhouette fitter is tried

    def __init__(self, wm, island=None):
        self.wm, self.island = wm, island
        self.zoom = REAL_ZOOM
        self.fails = 0
        self.adaptive = False
        self.last_score = 0.0
        self.src = ""                            # which matcher gave the last fix ("island" / "world")
        self.text_px = self.TEXT_PX
        self.zs, self.sx = 0.95, True             # the minimap size factor and storm mode that matched last

    # ---- observation: the minimap as a masked Lab template at a tile level
    def _template(self, minimap, level, zs=None, sx=None):
        zs = self.zs if zs is None else zs
        sx = self.sx if sx is None else sx
        s = Z_REAL * zs * 2 ** level
        h = minimap.shape[0]
        interp = cv2.INTER_AREA if s < 1 else cv2.INTER_LINEAR
        storm = storm_mask(minimap)
        fr = float(storm.mean())
        up = cv2.resize(minimap, None, fx=s, fy=s, interpolation=interp)
        H, W = up.shape[:2]
        hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
        text = ((hsv[..., 1] < 45) & (hsv[..., 2] > 190)).astype(np.uint8)              # white map text / icons: they come with a thick dark outline, mask that too
        text = cv2.dilate(text, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.text_px, self.text_px)))
        bad = (text > 0) | ((hsv[..., 0] >= 158) & (hsv[..., 1] > 110) & (hsv[..., 2] > 150))
        bad |= (hsv[..., 0] >= 104) & (hsv[..., 0] <= 125) & (hsv[..., 1] >= 170) & (hsv[..., 2] >= 170)       # the blue pins and arrows drawn on the minimap (your own marker, pings): water is teal, not this blue
        k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        if sx and fr >= 0.02:
            edge = (cv2.dilate(storm.astype(np.uint8), k) - cv2.erode(storm.astype(np.uint8), k)) > 0
            bad |= edge | (~storm if fr >= 0.5 else storm)             # each side of the storm edge has its own gain / offset
        bad = cv2.dilate(bad.astype(np.uint8), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
        r0, r1, c0, c1 = int(H * self.ROWS[0]), int(H * self.ROWS[1]), int(W * self.COLS[0]), int(W * self.COLS[1])
        chans = lab8(up, 0.5 * max(s, 1.0))
        mask = cv2.resize(1 - bad, (W, H), interpolation=cv2.INTER_NEAREST)[r0:r1, c0:c1].astype(np.float32)
        c = ((h - 1) / 2 + 0.5) * s - 0.5                               # player (crop centre) in the resized crop
        return [ch[r0:r1, c0:c1] for ch in chans], mask, (c - c0, c - r0), fr

    @staticmethod
    def _score(window, tpl, mask):
        acc = 0
        for w, t in zip(window, tpl):
            acc = acc + np.nan_to_num(cv2.matchTemplate(w, t, cv2.TM_CCOEFF_NORMED, mask=mask), nan=-1.0, posinf=-1.0, neginf=-1.0)   # flat patches (open sea) have no variance
        if float(mask.mean()) < 0.08:                              # (almost) everything masked out: the correlation of nothing says nothing, however high it comes out
            return np.full_like(acc, -1.0)
        return acc / len(tpl)

    def _trimmed(self, obs, level, px, py, drop=0.2):
        """The match at (px, py) with the `drop` worst-fitting pixels left out: icons drawn on the minimap (the bus, pings, key hints) are not on
        the map, so they should not decide whether this is the place. -> normalised correlation over the rest."""
        tpl, mask, (ox, oy), _ = obs
        th, tw = tpl[0].shape
        x, y = int(round(px - ox)), int(round(py - oy))
        D = self.wm.desc[level]
        if x < 0 or y < 0 or y + th > D[0].shape[0] or x + tw > D[0].shape[1]:
            return -1.0
        keep = mask > 0
        if keep.sum() < 2000:
            return -1.0
        z = []
        for w, t in zip((d[y:y + th, x:x + tw] for d in D), tpl):
            a, b = w.astype(np.float64), t.astype(np.float64)
            a = (a - a[keep].mean()) / (a[keep].std() + 1e-6)
            b = (b - b[keep].mean()) / (b[keep].std() + 1e-6)
            z.append((a, b))
        err = sum((a - b) ** 2 for a, b in z)
        keep &= err <= np.quantile(err[keep], 1.0 - drop)
        return float(np.mean([(a[keep] * b[keep]).mean() for a, b in z]))

    @staticmethod
    def _peak(score):
        _, mx, _, loc = cv2.minMaxLoc(score)
        x, y = loc
        sx = sy = 0.0
        if 0 < x < score.shape[1] - 1:
            a, b, c = score[y, x - 1], score[y, x], score[y, x + 1]
            sx = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c != 0 else 0.0
        if 0 < y < score.shape[0] - 1:
            a, b, c = score[y - 1, x], score[y, x], score[y + 1, x]
            sy = 0.5 * (a - c) / (a - 2 * b + c) if a - 2 * b + c != 0 else 0.0
        return float(mx), x + float(np.clip(sx, -1, 1)), y + float(np.clip(sy, -1, 1))

    def _local(self, obs, level, centre_px, radius):
        """Best match with the player within `radius` tile px of `centre_px` -> (score, player_x, player_y) in tile px."""
        tpl, mask, (ox, oy), _ = obs
        th, tw = tpl[0].shape
        cx, cy = int(round(centre_px[0] - ox)), int(round(centre_px[1] - oy))      # template origin guess
        D = self.wm.desc[level]
        mh, mw = D[0].shape
        x0, y0 = max(cx - radius, 0), max(cy - radius, 0)
        x1, y1 = min(cx + radius + tw, mw), min(cy + radius + th, mh)
        if x1 - x0 < tw + 2 or y1 - y0 < th + 2:
            return -1.0, 0.0, 0.0
        sc = self._score([d[y0:y1, x0:x1] for d in D], tpl, mask)
        s, px, py = self._peak(sc)
        return s, x0 + px + ox, y0 + py + oy

    # ---- first look
    def _variants(self, minimap):
        """(zoom, storm-mode) pairs to try, the remembered one first."""
        has_storm = float(storm_mask(minimap).mean()) >= 0.02
        out = [(self.zs, self.sx if has_storm else True)]
        for zs in self.ZSCALES:
            for sx in self.STORM_MODES if has_storm else (True,):
                if (zs, sx) not in out:
                    out.append((zs, sx))
        return out

    def _cands(self, minimap, zs, sx):
        lv = self.ACQ_LEVEL
        obs = self._template(minimap, lv, zs, sx)
        tpl, mask, (ox, oy), _ = obs
        sc = self._score(self.wm.desc[lv], tpl, mask)
        cands = []
        for _ in range(3):                                    # the best peaks, each at least ~120 m apart
            s, px, py = self._peak(sc)
            if s < 0.3:
                break
            cands.append((s, px + ox, py + oy))
            cv2.circle(sc, (int(round(px)), int(round(py))), 40, -1.0, -1)
        return cands

    def acquire(self, minimap, variants=None):
        fit = self._acquire(minimap, variants)
        if fit is None and self.TEXT_PX_WIDE != self.TEXT_PX:
            hsv = cv2.cvtColor(minimap, cv2.COLOR_BGR2HSV)
            if float(((hsv[..., 1] < 45) & (hsv[..., 2] > 190)).mean()) >= 0.03:      # a place name across the map: look again with its outline masked too
                self.text_px = self.TEXT_PX_WIDE
                try:
                    fit = self._acquire(minimap, variants)
                finally:
                    self.text_px = self.TEXT_PX
        return fit

    def _acquire(self, minimap, variants=None):
        self.last_score = 0.0
        best_v, cands = None, []
        for v in (variants or self._variants(minimap)):
            c = self._cands(minimap, *v)
            if c and (best_v is None or c[0][0] > cands[0][0] + self.VARIANT_GAIN):      # a different zoom / storm mode must clearly beat the first
                best_v, cands = v, c
            if best_v is not None and cands[0][0] >= self.GOOD_ACQ:
                break
        if best_v is None:
            return None
        zs, sx = best_v
        best = None
        obs3 = self._template(minimap, self.TRACK_LEVEL, zs, sx)
        for s2, x2, y2 in cands:
            s, x, y = self._local(obs3, self.TRACK_LEVEL, (x2 * 2, y2 * 2), 10)
            if best is None or s > best[0]:
                best = (s, x, y)
        zs, best = self._tune(minimap, best, zs, sx)
        obs3 = self._template(minimap, self.TRACK_LEVEL, zs, sx)
        self.last_score = best[0]
        if self.TRIM_RAW <= best[0] < self.MIN_ACQ and self._trimmed(obs3, self.TRACK_LEVEL, best[1], best[2]) >= self.TRIM_ACQ:
            best = (self.MIN_ACQ, best[1], best[2])                    # occluded but right
        second = max((c[0] for c in cands if abs(c[1] * 2 - best[1]) + abs(c[2] * 2 - best[2]) > 40), default=0.0)
        if best[0] < self.MIN_ACQ or (cands and best[0] < second + self.MARGIN and len(cands) > 1 and second > 0.7 * best[0]):
            return None
        self.zs, self.sx = zs, sx
        return self._fit_from(best, minimap, obs3[3])

    VERIFY_DIST_PX, VERIFY_GAIN = 150.0, 0.08
    ZFINE = (0.90, 0.925, 0.95, 0.975, 1.0, 1.025, 1.05)

    def _tune(self, minimap, best, zs, sx):
        """The minimap's size on screen depends on the HUD scale / resolution: at the found spot, try a few sizes and keep the one that fits best
        (a template of the wrong size is a little off in position too). -> (zs, (score, x, y))."""
        top = (zs, best)
        for z in self.ZFINE:
            if abs(z - zs) < 1e-6:
                continue
            obs = self._template(minimap, self.TRACK_LEVEL, z, sx)
            r = self._local(obs, self.TRACK_LEVEL, (best[1], best[2]), 5)
            if r[0] > top[1][0]:
                top = (z, r)
        return top

    def verify(self, minimap, uv, local_score):
        """A look at the whole map while a lock is held: where the minimap fits best, if that is somewhere else (another place that fits clearly better
        than the tracked one). A lock that sits on the wrong spot would otherwise be 'confirmed' by its own local search for ever."""
        lv = self.TRACK_LEVEL
        c = uv_to_px(uv, lv)
        s0, px, py = self._local(self._template(minimap, lv), lv, c, 6)
        z, (s1, _, _) = self._tune(minimap, (s0, px, py), self.zs, self.sx)       # the minimap's size can change (HUD scale): keep it tuned, a wrong one is metres off
        if s1 > s0 + 0.01:
            self.zs = z
        keep = self.last_score
        alt = self.acquire(minimap, [(self.zs, self.sx)])           # the remembered zoom / storm mode only: one look, ~0.15 s
        sc = self.last_score
        self.last_score = keep
        if alt is None or sc < local_score + self.VERIFY_GAIN:
            return None
        return alt if np.hypot(alt.u - uv[0], alt.v - uv[1]) > self.VERIFY_DIST_PX else None

    def _fit_from(self, best, minimap, storm=0.0):
        s, x, y = best
        k = 0.0 if storm >= 0.5 else 1.0                                  # the bias is a clear-weather shadow effect
        u, v = px_to_uv([x - k * self.BIAS_PX[0] / 2, y - k * self.BIAS_PX[1] / 2], self.TRACK_LEVEL)
        fit = Fit(float(u), float(v), float(s), float("nan"), self.zoom, 0.0, 0.0, True)
        self.src = "world"
        return self._done(self._island(minimap, fit))

    def _island(self, minimap, fit):
        """On the island the silhouette fitter is ~10x more precise: use its fix when it is near this one."""
        isl = self.island
        if isl is None:
            return fit
        w, h = isl.world.w, isl.world.h
        dx = max(-fit.u, 0.0, fit.u - w)
        dy = max(-fit.v, 0.0, fit.v - h)
        if np.hypot(dx, dy) > self.ISLAND_ZONE_PX:
            return fit
        f2 = isl.fit(minimap, guess_uv=(fit.u, fit.v))
        if f2 is not None and np.hypot(f2.u - fit.u, f2.v - fit.v) < 120.0:       # 120 px ~ 10 m
            self.src = "island"
            return f2
        return fit

    def _alts(self):
        out = [(self.zs, not self.sx)]
        for d in (0.025, -0.025, 0.05, -0.05):
            z = round(self.zs + d, 4)
            if 0.86 <= z <= 1.08:
                out.append((z, self.sx))
        return out

    def fit(self, minimap, guess_uv=None):
        self.last_score = 0.0
        if guess_uv is not None:
            lv = self.TRACK_LEVEL
            obs = self._template(minimap, lv)
            c = uv_to_px(guess_uv, lv)
            s, x, y = self._local(obs, lv, c, 14)
            self.last_score = max(self.last_score, s)
            if self.LOCAL_TRIM_RAW <= s < self.MIN_SCORE and self._trimmed(obs, lv, x, y) >= self.LOCAL_TRIM:
                s = self.MIN_SCORE                                         # icons on the minimap (pings, the bus, route lines) pulled the score down; the rest fits
            if self.GUIDED_MIN <= s < self.MIN_SCORE and np.hypot(x - c[0], y - c[1]) <= self.GUIDED_PX:
                s = self.MIN_SCORE                                         # covered, but exactly where we expected it
            if s >= self.MIN_SCORE:
                self.fails = 0
                return self._fit_from((s, x, y), minimap, obs[3])
            best = None
            for z, sx in self._alts():                                     # the remembered size / storm mode stopped fitting: try the neighbours before giving up
                o = self._template(minimap, lv, z, sx)
                s2, x2, y2 = self._local(o, lv, c, 14)
                self.last_score = max(self.last_score, s2)
                if s2 < self.MIN_SCORE and not (self.LOCAL_TRIM_RAW <= s2 and self._trimmed(o, lv, x2, y2) >= self.LOCAL_TRIM):
                    continue
                if best is None or s2 > best[0]:
                    best = (s2, x2, y2, z, sx, o[3])
            if best is not None:
                s2, x2, y2, z, sx, storm = best
                self.zs, self.sx, self.fails = z, sx, 0
                return self._fit_from((max(s2, self.MIN_SCORE), x2, y2), minimap, storm)
            self.fails += 1
            return None
        fit = self.acquire(minimap)
        self.fails = 0 if fit is not None else self.fails + 1
        return fit

    def _done(self, fit):
        self.adaptive = bool(fit.adaptive)
        return fit


class WorldTracker(Tracker):
    """Tracker over the whole map: the same gate / coast / Kalman logic, position fixes from WorldFitter."""

    def __init__(self, island_world, wm=None):
        wm = wm or world_map()
        super().__init__(island_world, fitter=WorldFitter(wm, SilhouetteFitter(island_world)))
        self.wm = wm

    def on_land(self, u, v):
        return self.wm.on_map(u, v)
