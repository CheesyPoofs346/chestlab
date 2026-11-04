"""Game simulator: synthetic minimap and 3D view with known ground truth. The arrow sprite, storm tint, clutter
and compression here are deliberately *different* from what the estimators assume, so tests aren't circular."""
import math

import cv2
import numpy as np

from .geometry import basis


def arrow_polygon(cx, cy, heading, size=17.0, notch=0.35):
    """Arrow pointing along `heading` (compass degrees, clockwise from up). Image coords: x right, y down."""
    th = math.radians(heading)
    fwd, side, c = np.array([math.sin(th), -math.cos(th)]), np.array([math.cos(th), math.sin(th)]), np.array([cx, cy])
    return np.array([c + fwd * size,
                     c - fwd * size * 0.7 + side * size * 0.6,
                     c + fwd * size * (-0.7 + 0.7 * notch),
                     c - fwd * size * 0.7 - side * size * 0.6])


def _poly(img, pts, color, outline=None):
    q = np.round(pts * 16).astype(np.int32)
    cv2.fillPoly(img, [q], color, lineType=cv2.LINE_AA, shift=4)
    if outline is not None:
        cv2.polylines(img, [q], True, outline, 1, lineType=cv2.LINE_AA, shift=4)


def draw_arrow(img, cx, cy, heading, size=17.0, notch=0.35):
    """White arrow with a dark outline, as the game's player marker looks."""
    _poly(img, arrow_polygon(cx, cy, heading, size, notch), (255, 255, 255), (20, 20, 20))


def render_minimap(world, uv, heading, zoom, rng, size=231, storm=0.0, noise=0.0, jpeg=None, blur=0.0,
                   clutter=True, arrow=(17.0, 0.35), scene_seed=None):
    """Player-centred, north-up minimap. The player marker is always at the centre. scene_seed fixes the storm and
    clutter geometry so consecutive frames of a walk share them (only noise changes frame to frame)."""
    srng = np.random.default_rng(scene_seed) if scene_seed is not None else rng
    small, zx, zy = world.small(zoom)
    c = (size - 1) / 2
    tx, ty = c - ((uv[0] + 0.5) * zx - 0.5), c - ((uv[1] + 0.5) * zy - 0.5)
    out = cv2.warpAffine(small, np.float32([[1, 0, tx], [0, 1, ty]]), (size, size), flags=cv2.INTER_LINEAR,
                         borderMode=cv2.BORDER_CONSTANT, borderValue=world.ocean).astype(np.float32)
    if storm > 0:  # purple wall sweeping in from a random side, like the closing storm
        phi, off = srng.uniform(0, 2 * math.pi), srng.uniform(-10, 45)
        yy, xx = np.mgrid[0:size, 0:size]
        a = np.clip(((xx - c) * math.cos(phi) + (yy - c) * math.sin(phi) - off) / 8.0, 0, 1)[..., None] * storm
        out = out * (1 - a) + np.array([190, 50, 170], np.float32) * a
    out = np.clip(out, 0, 255).astype(np.uint8)
    if clutter:  # route line to the player + two waypoint pins
        ang = srng.uniform(0, 2 * math.pi)
        p0 = (int(c + 150 * math.cos(ang)), int(c + 150 * math.sin(ang)))
        cv2.line(out, p0, (int(c), int(c)), (255, 255, 255), 2, cv2.LINE_AA)
        for col in ((40, 190, 250), (150, 90, 240)):
            a2, r2 = srng.uniform(0, 2 * math.pi), srng.uniform(45, 100)
            p = (int(c + r2 * math.cos(a2)), int(c + r2 * math.sin(a2)))
            cv2.circle(out, p, 6, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(out, p, 6, (20, 20, 20), 1, cv2.LINE_AA)
            cv2.circle(out, p, 2, col, -1, cv2.LINE_AA)
    _poly(out, arrow_polygon(c, c, heading, *arrow), (255, 255, 255), (20, 20, 20))
    if blur > 0:
        out = cv2.GaussianBlur(out, (0, 0), blur)
    if noise > 0:
        out = np.clip(out.astype(np.float32) + rng.normal(0, noise, out.shape), 0, 255).astype(np.uint8)
    if jpeg:
        out = cv2.imdecode(cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(jpeg)])[1], cv2.IMREAD_COLOR)
    return out


def render_view(world, cam, eye, yaw, pitch):
    """Ground-plane view of the island texture under a sky gradient (cheap stand-in for the 3D game frame)."""
    W, H, f = cam.w, cam.h, cam.f
    fwd, right, up = basis(yaw, pitch)
    sx, sy = np.meshgrid(np.arange(W) + 0.5, np.arange(H) + 0.5)
    d = fwd + right * ((sx - W / 2) / f)[..., None] + up * ((H / 2 - sy) / f)[..., None]
    dz = d[..., 2]
    below = dz < -1e-4
    t = np.where(below, -eye[2] / np.where(below, dz, -1.0), 0.0)
    u = (eye[0] + d[..., 0] * t) / world.mpp
    v = -(eye[1] + d[..., 1] * t) / world.mpp
    ground = cv2.remap(world.img, np.where(below, u, -1).astype(np.float32), np.where(below, v, -1).astype(np.float32),
                       cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=world.ocean)
    k = np.clip(dz / 0.5, 0, 1)[..., None]
    sky = (np.array([245, 215, 150], np.float32) * (1 - k) + np.array([200, 120, 40], np.float32) * k).astype(np.uint8)
    return np.where(below[..., None], ground, sky)


def draw_markers(img, pts, color, r=7, labels=None):
    for i, (x, y) in enumerate(pts):
        if np.isfinite(x) and np.isfinite(y) and -50 < x < img.shape[1] + 50 and -50 < y < img.shape[0] + 50:
            cv2.circle(img, (int(x), int(y)), r, color, 2, cv2.LINE_AA)
            if labels is not None:
                cv2.putText(img, str(labels[i]), (int(x) + 9, int(y) - 9), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img
