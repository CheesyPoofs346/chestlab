"""Which of the 19 buried-chest spawns exist in THIS match, read off the in-game full map (open it with M, press F11).

assets/island.png is itself a screenshot of that map with 11 chest icons on it, so it serves twice: the picture the frame is
registered to (SIFT + RANSAC similarity, so any resolution / map zoom / pan works), and the source of the icon sprite. The
frame is warped into island.png's coordinates and each spawn position is checked for the icon (masked template match over a
few scales, since the icon may not scale with the map zoom).  scan(frame) -> ScanResult."""
import json
import os
from dataclasses import dataclass

import cv2
import numpy as np

from .paths import BASE as ROOT
ISLAND = os.path.join(ROOT, "assets", "island.png")
SPAWNS = os.path.join(ROOT, "assets", "spawns.json")
HALF = 16                      # icon template half-size in island.png pixels (icon + its dark halo is ~30 px)
SCALES = (0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)
MIN_INLIERS, MIN_RATIO, MAX_ROT_DEG = 25, 0.20, 8.0
PRESENT = 0.68                 # template-match score at or above which a spawn counts as having its icon


@dataclass
class ScanResult:
    ok: bool
    message: str
    present: np.ndarray = None       # (n,) bool, per spawn
    scores: np.ndarray = None        # (n,) icon match score, for diagnosis
    inliers: int = 0
    scale: float = 0.0               # frame pixels per island.png pixel


def _sift():
    return cv2.SIFT_create(nfeatures=4000)


class MapScanner:
    def __init__(self, island=None, spawns=None):
        self.img = cv2.imread(ISLAND) if island is None else island
        if spawns is None:
            with open(SPAWNS, encoding="utf-8") as f:
                spawns = json.load(f)
        self.spawns = np.array([[c["map_x"], c["map_y"]] for c in spawns], float)
        self.listed = np.array([bool(c.get("on_map", True)) for c in spawns])      # the ones whose icon island.png shows
        self.sift = _sift()
        self.kp, self.des = self.sift.detectAndCompute(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY), None)
        self.tpl, self.mask = self._icon()

    def _icon(self):
        """Median of the icons already on island.png; the mask keeps the pixels that agree across them (the sprite, not the
        ground under it)."""
        crops = []
        for x, y in self.spawns[self.listed]:
            x, y = int(round(x)), int(round(y))
            if HALF <= x < self.img.shape[1] - HALF and HALF <= y < self.img.shape[0] - HALF:
                crops.append(self.img[y - HALF:y + HALF + 1, x - HALF:x + HALF + 1].astype(np.float32))
        stack = np.stack(crops)
        tpl = np.median(stack, axis=0)
        sd = stack.std(axis=0).mean(axis=2)
        mask = (sd < 14).astype(np.float32)
        return tpl.astype(np.uint8), cv2.merge([mask] * 3)

    # ---- map registration
    def register(self, frame):
        """frame pixels -> island.png pixels: (2x3 similarity, inliers, ratio) or None."""
        g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp, des = self.sift.detectAndCompute(g, None)
        if des is None or len(kp) < 30:
            return None
        pairs = cv2.BFMatcher().knnMatch(des, self.des, k=2)
        good = [m for m, n in (p for p in pairs if len(p) == 2) if m.distance < 0.75 * n.distance]
        if len(good) < MIN_INLIERS:
            return None
        src = np.float32([kp[m.queryIdx].pt for m in good])
        dst = np.float32([self.kp[m.trainIdx].pt for m in good])
        M, inl = cv2.estimateAffinePartial2D(src, dst, method=cv2.RANSAC, ransacReprojThreshold=3.0, maxIters=4000, confidence=0.999)
        if M is None:
            return None
        n = int(inl.sum())
        rot = abs(np.degrees(np.arctan2(M[1, 0], M[0, 0])))
        if n < MIN_INLIERS or n / len(good) < MIN_RATIO or rot > MAX_ROT_DEG:
            return None
        return M, n, n / len(good)

    # ---- icon presence
    def icon_scores(self, aligned, search=7):
        """(len(SCALES), n) best masked template score near each spawn, per icon scale. aligned = frame warped into
        island.png space."""
        out = np.zeros((len(SCALES), len(self.spawns)))
        for k, s in enumerate(SCALES):
            sz = max(9, int(round(self.tpl.shape[0] * s)) | 1)
            t = cv2.resize(self.tpl, (sz, sz), interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC)
            m = cv2.resize(self.mask, (sz, sz), interpolation=cv2.INTER_NEAREST)
            h = sz // 2
            for i, (x, y) in enumerate(self.spawns):
                x, y = int(round(x)), int(round(y))
                x0, y0, x1, y1 = x - h - search, y - h - search, x + h + search + 1, y + h + search + 1
                if x0 < 0 or y0 < 0 or x1 > aligned.shape[1] or y1 > aligned.shape[0]:
                    continue
                out[k, i] = float(cv2.matchTemplate(aligned[y0:y1, x0:x1], t, cv2.TM_CCOEFF_NORMED, mask=m).max())
        return np.nan_to_num(out)

    @staticmethod
    def pick(scores):
        """All icons on the map are drawn at one size, so the scale that fits the best few spawns is THE scale; judging every
        spawn at that scale (and its neighbours) stops the best-of-all-scales from inflating the ones with no icon."""
        k = int(np.argmax(np.sort(scores, axis=1)[:, -5:].mean(axis=1)))
        return scores[max(k - 1, 0):k + 2].max(axis=0), SCALES[k]

    def scan(self, frame):
        reg = self.register(frame)
        if reg is None:
            return ScanResult(False, "map scan: can't see the island map - open the full map (M) and try again")
        M, n, ratio = reg
        h, w = self.img.shape[:2]
        aligned = cv2.warpAffine(frame, M, (w, h), flags=cv2.INTER_AREA if np.hypot(M[0, 0], M[1, 0]) < 1 else cv2.INTER_LINEAR,
                                 borderMode=cv2.BORDER_REPLICATE)
        sc, icon_scale = self.pick(self.icon_scores(aligned))
        present = sc >= PRESENT
        s = float(np.hypot(M[0, 0], M[1, 0]))
        if not present.any():
            return ScanResult(False, "map scan: found the map but no chest icons on it (scores %s)" % np.round(sc.max(), 2), present, sc, n, 1.0 / s)
        return ScanResult(True, "map scan: %d of %d spawns have a chest this match" % (int(present.sum()), len(present)), present, sc, n, 1.0 / s)
