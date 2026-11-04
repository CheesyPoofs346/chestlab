"""Pinhole camera math. World frame is ENU in metres: x east, y north, z up. Yaw is a compass heading in degrees
(clockwise from north), pitch is degrees above the horizon."""
import math

import numpy as np


def wrap180(a):
    return (np.asarray(a, dtype=float) + 180.0) % 360.0 - 180.0


def basis(yaw_deg, pitch_deg):
    """Unit vectors (forward, right, up) of a camera looking along yaw/pitch."""
    y, p = math.radians(yaw_deg), math.radians(pitch_deg)
    fwd = np.array([math.sin(y) * math.cos(p), math.cos(y) * math.cos(p), math.sin(p)])
    right = np.array([math.cos(y), -math.sin(y), 0.0])
    return fwd, right, np.cross(right, fwd)


class Camera:
    """Third-person orbit camera: it hangs `dist` metres behind a pivot `pivot_h` above the player, `shoulder`
    metres to the right (the over-the-shoulder offset). fov_h is the horizontal field of view in degrees."""

    def __init__(self, width=2560, height=1440, fov_h=80.0, near=0.2, dist=2.8, shoulder=0.5, pivot_h=1.5):
        self.w, self.h, self.fov_h, self.near = width, height, fov_h, near
        self.dist, self.shoulder, self.pivot_h = dist, shoulder, pivot_h
        self.f = (width / 2) / math.tan(math.radians(fov_h / 2))

    def with_size(self, width, height):
        return Camera(width, height, self.fov_h, self.near, self.dist, self.shoulder, self.pivot_h)

    def eye(self, player, yaw, pitch):
        fwd, right, _ = basis(yaw, pitch)
        return np.asarray(player, float) + [0, 0, self.pivot_h] - fwd * self.dist + right * self.shoulder

    def project(self, pts, eye, yaw, pitch):
        """-> (screen_xy (N,2), depth (N,), in_front (N,) bool)."""
        d = np.atleast_2d(np.asarray(pts, float)) - eye
        fwd, right, up = basis(yaw, pitch)
        zc, xc, yc = d @ fwd, d @ right, d @ up
        ok = zc > self.near
        z = np.where(ok, zc, 1.0)
        return np.stack([self.w / 2 + self.f * xc / z, self.h / 2 - self.f * yc / z], 1), zc, ok

    def ray(self, sx, sy, yaw, pitch):
        """Unnormalised world direction through screen pixel (sx, sy)."""
        fwd, right, up = basis(yaw, pitch)
        return fwd + right * (sx - self.w / 2) / self.f + up * (self.h / 2 - sy) / self.f

    def horizon_row(self, pitch):
        """Screen row where the horizon sits for a given pitch (looking up pushes it down)."""
        return self.h / 2 + self.f * math.tan(math.radians(pitch))
