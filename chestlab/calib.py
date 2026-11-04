"""Calibration: correcting the chest map from where you actually stood, and solving FOV / yaw offset."""
import math

import numpy as np

from .geometry import wrap180


class ChestRecorder:
    """Press a key while standing at chest N: the median of the recent position estimates replaces that chest's
    map position. Because the same minimap fit produces both the chest and later player positions, systematic
    errors (map offset, wrong scale of the chest picture) cancel; only the random jitter remains."""

    def __init__(self, chests_uv):
        self.chests = np.array(chests_uv, float)
        self.n = np.zeros(len(chests_uv), int)

    def press(self, idx, recent_uv):
        est = np.median(np.asarray(recent_uv, float), axis=0)
        self.chests[idx] = est if self.n[idx] == 0 else (self.chests[idx] * self.n[idx] + est) / (self.n[idx] + 1)
        self.n[idx] += 1


def fit_fov_yaw(rel_deg, screen_x, width, fov0=80.0, iters=15):
    """Least squares (Gauss-Newton): screen_x = W/2 + f(fov) * tan(rel + yaw_offset). -> (fov_deg, yaw_offset_deg).
    rel_deg are bearings of marked chests relative to the compass heading; screen_x where you saw them."""
    rel, sx = np.asarray(rel_deg, float), np.asarray(screen_x, float)
    p = np.array([fov0, 0.0])

    def model(q):
        return width / 2 + (width / 2) / math.tan(math.radians(q[0] / 2)) * np.tan(np.radians(rel + q[1]))

    for _ in range(iters):
        r = sx - model(p)
        J = np.stack([(model(p + [1e-4, 0]) - model(p - [1e-4, 0])) / 2e-4,
                      (model(p + [0, 1e-4]) - model(p - [0, 1e-4])) / 2e-4], 1)
        step = np.linalg.lstsq(J, r, rcond=None)[0]
        p = p + step
        if np.abs(step).max() < 1e-7:
            break
    return float(p[0]), float(wrap180(p[1]))


def metres_per_pixel(hud_distance_m, uv_a, uv_b):
    """Scale from one in-game distance readout (e.g. the waypoint's '52m') between two known map points."""
    return hud_distance_m / float(np.hypot(*(np.asarray(uv_a, float) - np.asarray(uv_b, float))))
