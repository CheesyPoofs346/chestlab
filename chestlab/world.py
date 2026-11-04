import json
import os

import cv2
import numpy as np

from .paths import BASE

ASSETS = os.path.join(BASE, "assets")


class World:
    """Island map + chests. Map space = pixels of island.png (u right, v down). ENU space = metres.
    `mpp` (metres per map pixel) is an assumption until calibrated against a real in-game distance."""

    def __init__(self, mpp=0.21):
        self.mpp = mpp
        self.img = cv2.imread(os.path.join(ASSETS, "island.png"), cv2.IMREAD_COLOR)
        self.h, self.w = self.img.shape[:2]
        with open(os.path.join(ASSETS, "chests.json"), encoding="utf-8") as f:
            self.chests = np.array([[c["map_x"], c["map_y"]] for c in json.load(f)], float)
        corners = np.concatenate([self.img[:12, :12].reshape(-1, 3), self.img[-12:, :12].reshape(-1, 3)])
        self.ocean = tuple(int(x) for x in np.median(corners, axis=0))
        self._small = {}

    def small(self, zoom):
        """Island image shrunk by `zoom` -> (image, actual_zoom_x, actual_zoom_y). Cached."""
        key = round(zoom, 5)
        if key not in self._small:
            if len(self._small) > 64:
                self._small.clear()
            s = cv2.resize(self.img, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_AREA)
            self._small[key] = (s, s.shape[1] / self.w, s.shape[0] / self.h)
        return self._small[key]

    def uv_to_enu(self, uv, z=0.0):
        uv = np.atleast_2d(np.asarray(uv, float))
        return np.column_stack([uv[:, 0] * self.mpp, -uv[:, 1] * self.mpp, np.full(len(uv), z, float)])

    def enu_to_uv(self, enu):
        enu = np.atleast_2d(np.asarray(enu, float))
        return np.column_stack([enu[:, 0] / self.mpp, -enu[:, 1] / self.mpp])
