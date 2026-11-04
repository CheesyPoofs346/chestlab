"""Chest-marker HUD renderer: pure numpy/OpenCV/PIL, anti-aliased, per-pixel alpha. It draws into a premultiplied BGRA
canvas (what a layered window wants), so it runs the same on a screenshot as on the live overlay.

Look: the game's own HUD language. A small gold chest, the distance in heavy italic white with a dark outline, and a
thin stem down to the spot. No panels, no glow, no colour-coding. A chest that may not exist in this match is a hollow
outline. Cost control: the chest + label sprites are cached, and only the rectangles drawn last frame are wiped."""
import functools
import math
import os
import time

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .paths import BASE

FONT = "C:/Windows/Fonts/seguibli.ttf"          # Segoe UI Black Italic (the game's own HUD voice, for the chest markers)
UI_FONT = "C:/Windows/Fonts/seguisb.ttf"        # Segoe UI Semibold (the radar: small, upright, easy to read)
# Where the app itself reads the game (2000-px-wide frame coordinates): left fully transparent so the overlay can
# stay visible to screenshots without ever hiding the minimap or the compass number from our own capture.
KEEP_CLEAR = ((1770, 6, 1996, 230), (930, 0, 1070, 44))
WHITE, INK = (255, 255, 255), (14, 12, 10)
GOLD, GOLD_LIT, GOLD_SHADE, IRON, LOCK = (40, 185, 242), (112, 224, 255), (18, 128, 196), (32, 52, 84), (225, 238, 245)
# chest colours (BGR: body, lid, glint) per kind of chest: buried treasure is gold, the rest of the map's chests are told apart by hue
PALETTE = {"buried": (GOLD_SHADE, GOLD, GOLD_LIT), "chest": ((24, 92, 214), (42, 142, 250), (112, 192, 255)),
           "rare": ((140, 48, 104), (196, 88, 156), (238, 168, 218)), "sprite": ((56, 136, 38), (88, 198, 78), (170, 240, 150))}
PALETTE["chest_sure"] = PALETTE["chest"]
PALETTE["vault"] = ((150, 100, 40), (210, 150, 70), (250, 210, 150))      # behind a vault door: key or bomb
UP_COL, DOWN_COL = (120, 226, 255), (255, 196, 140)   # BGR: warm for a chest above you, cool for one below
LEVEL_M = 1.5                                         # height difference from which a marker says 'up' / 'down'
RING_M = 25.0                                         # chests closer than this get a ring on the ground (where to dig)
ICON_DIR = os.path.join(BASE, "assets", "icons")
# fortnite.gg map icon ids (https://fortnite.gg/icons/<id>.png): 1327 chest with the gold outline = 100% spawn, 2 plain chest = may not spawn,
# 3 buried chest, 1326 rare chest, 1630 sprite chest, 996 the padlock the map uses for its vaults (key or bomb)
GG_ICON = {"rare": 1326, "sprite": 1630, "vault": 996}
RADAR_LEVELS = (10.0, 20.0, 40.0, 80.0)               # auto range: the smallest of these that holds the target, so the last metres get big
RADAR_M = 60.0                                        # radar: metres from the centre to the rim (chests up to 1.5x that sit on the rim, dimmed)
RADAR_ICONS = 28                                      # most chest icons the radar will draw (the nearest ones): enough to be useful, not a cloud
ROUTE_LEG = 4                                         # route mode: the line runs through this many chests ahead
RADAR_R = 0.078                                       # radar outer radius, fraction of the screen height
RADAR_AT = (0.905, 0.60)                              # radar centre, fraction of screen width / height (default: right side, clear of minimap, compass, hotbar)
RADAR_SPOTS = {"default": RADAR_AT,                   # config radar_at: one of these names, or [x, y] fractions of the screen
               "left": (0.095, 0.60),
               "top-left": (0.075, 0.20), "top-right": (0.915, 0.30),
               "bottom-left": (0.175, 0.775), "bottom-right": (0.915, 0.74)}


def radar_spot(name):
    """Radar centre (fractions of the screen) for a config value: a spot name or [x, y]; anything else is the default."""
    if isinstance(name, (list, tuple)) and len(name) == 2:
        try:
            return (min(max(float(name[0]), 0.04), 0.96), min(max(float(name[1]), 0.08), 0.92))
        except (TypeError, ValueError):
            pass
    return RADAR_SPOTS.get(name, RADAR_AT) if isinstance(name, str) else RADAR_AT


def fade(d):
    return 1.0 if d <= 40 else max(0.6, 1.0 - (d - 40) / 220)


def hex_bgr(h):
    return tuple(int(h[i:i + 2], 16) for i in (5, 3, 1))


def P(a):
    return np.round(np.asarray(a, np.float64) * 16).astype(np.int32)


@functools.lru_cache(maxsize=64)
def _font(px, face=None):
    return ImageFont.truetype(face or FONT, int(px))


@functools.lru_cache(maxsize=512)
def text_mask(s, px, face=None):
    f = _font(px, face)
    l, t, r, b = f.getbbox(s)
    im = Image.new("L", (r - l + 2, b - t + 2), 0)             # italic glyphs lean past the origin: l can be negative
    ImageDraw.Draw(im).text((-l, -t), s, font=f, fill=255)
    return np.asarray(im)


def over(dst, mask, color, op=1.0):
    """Composite a coverage mask in `color` onto the float premultiplied BGRA array `dst`."""
    a = (mask.astype(np.float32) * (op / 255.0))[..., None]
    dst *= 1.0 - a
    dst += a * np.array((color[0], color[1], color[2], 255), np.float32)


def blit(cv, sp, x, y, op=1.0, dirty=None):
    """Composite a premultiplied BGRA sprite at (x, y), scaled by opacity `op`."""
    h, w = sp.shape[:2]
    x, y = int(round(x)), int(round(y))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, cv.shape[1]), min(y + h, cv.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    s = sp[y0 - y:y1 - y, x0 - x:x1 - x]
    if op < 0.995:
        s = cv2.convertScaleAbs(s, alpha=op)
    inv = cv2.bitwise_not(s[..., 3])
    cv[y0:y1, x0:x1] = cv2.add(cv2.multiply(cv[y0:y1, x0:x1], cv2.merge([inv, inv, inv, inv]), scale=1 / 255.0), s)
    if dirty is not None:
        dirty.append((x0, y0, x1, y1))


def _u8(f):
    return np.clip(f + 0.5, 0, 255).astype(np.uint8)


# ---- sprites (each cached: they only depend on a size, a string, a state)
@functools.lru_cache(maxsize=512)
def text_sprite(s, px, col=WHITE, face=None):
    """Heavy italic text with a dark outline and a faint shadow, as a premultiplied BGRA sprite."""
    m = text_mask(s, px, face)
    r = max(2, int(round(px * 0.11)))
    pad = r + 3
    base = np.zeros((m.shape[0] + 2 * pad, m.shape[1] + 2 * pad), np.uint8)
    base[pad:pad + m.shape[0], pad:pad + m.shape[1]] = m
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    edge = cv2.dilate(base, k)
    shade = cv2.GaussianBlur(np.roll(edge, max(1, px // 14), axis=0), (0, 0), max(1.0, px * 0.09))
    out = np.zeros(base.shape + (4,), np.float32)
    over(out, shade, (0, 0, 0), 0.45)
    over(out, edge, INK, 0.96)
    over(out, base, col)
    return _u8(out)


@functools.lru_cache(maxsize=64)
def chest_sprite(s, hollow=False, kind="buried"):
    """The golden chest from the game, side view: lit lid, shaded body, iron band and lock. hollow = outline only (a spawn
    that may not exist in this match)."""
    pad = max(2, s // 12)
    n = s + 2 * pad
    def draw(fn):
        m = np.zeros((n, n), np.uint8)
        fn(m, lambda a: P(np.asarray(a, float) * s + pad))
        return m
    def rect(x0, y0, x1, y1):
        return lambda m, q: cv2.fillPoly(m, [q([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])], 255, cv2.LINE_AA, 4)
    def dome(m, q):
        cv2.ellipse(m, tuple(int(v) for v in q([0.5, 0.5])), tuple(int(v) for v in P([0.42 * s, 0.34 * s])), 0, 180, 360, 255, -1, cv2.LINE_AA, 4)
    body, lid = draw(rect(.08, .50, .92, .92)), draw(dome)
    sil = cv2.bitwise_or(body, lid)
    r = max(1, int(round(s / 13)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = np.zeros((n, n, 4), np.float32)
    if hollow:
        over(out, sil, INK, 0.30)
        over(out, cv2.dilate(sil, k) - cv2.erode(sil, k), WHITE, 0.95)
        over(out, draw(lambda m, q: cv2.polylines(m, [q([[.08, .50], [.92, .50]])], False, 255, max(1, s // 16), cv2.LINE_AA, 4)), WHITE, 0.8)   # seam
        return _u8(out)
    over(out, cv2.dilate(sil, k), INK, 0.97)
    shade, top, lit = PALETTE.get(kind, PALETTE["buried"])
    over(out, body, shade)
    over(out, lid, top)
    over(out, cv2.bitwise_and(sil, draw(rect(.42, .10, .58, .92))), IRON)                       # iron band
    over(out, draw(lambda m, q: cv2.polylines(m, [q([[.08, .50], [.92, .50]])], False, 255, max(1, s // 16), cv2.LINE_AA, 4)), IRON, 0.9)   # seam
    over(out, draw(rect(.44, .52, .56, .68)), LOCK)                                             # lock
    hi = draw(lambda m, q: cv2.ellipse(m, tuple(int(v) for v in q([0.5, 0.5])), tuple(int(v) for v in P([0.34 * s, 0.26 * s])), 0, 200, 250, 255, max(1, s // 12), cv2.LINE_AA, 4))
    over(out, cv2.bitwise_and(hi, lid), lit, 0.9)                                          # glint on the lid
    return _u8(out)


@functools.lru_cache(maxsize=64)
def arrow_sprite(s, kind, col=WHITE):
    """kind: 'up' / 'down' (a filled arrow head) or 'left' / 'right' (a chevron), with a dark outline."""
    pad = max(2, s // 6)
    n = s + 2 * pad
    m = np.zeros((n, n), np.uint8)
    q = lambda a: P(np.asarray(a, float) * s + pad)
    if kind in ("up", "down"):
        a = [[.5, .08], [.94, .88], [.06, .88]] if kind == "up" else [[.5, .92], [.94, .12], [.06, .12]]
        cv2.fillPoly(m, [q(a)], 255, cv2.LINE_AA, 4)
    else:
        a = [[.30, .08], [.78, .5], [.30, .92]] if kind == "right" else [[.70, .08], [.22, .5], [.70, .92]]
        cv2.polylines(m, [q(a)], False, 255, max(2, s // 4), cv2.LINE_AA, 4)
    r = max(1, int(round(s / 9)))
    edge = cv2.dilate(m, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
    out = np.zeros((n, n, 4), np.float32)
    over(out, edge, INK, 0.95)
    over(out, m, col)
    return _u8(out)


@functools.lru_cache(maxsize=128)
def badge_sprite(n, s, hot=False):
    """The chest's place in the route: a number in a disc (gold when it is the one to open next)."""
    pad = 3
    d = s + 2 * pad
    m = np.zeros((d, d), np.uint8)
    cv2.circle(m, (d * 8, d * 8), int(s * 8), 255, -1, cv2.LINE_AA, 4)
    out = np.zeros((d, d, 4), np.float32)
    over(out, cv2.dilate(m, np.ones((3, 3), np.uint8)), INK if not hot else WHITE, 0.95)
    over(out, m, GOLD if hot else (46, 42, 38))
    t = text_mask(str(n), max(8, int(s * (0.92 if n < 10 else 0.74))))
    th, tw = t.shape
    y0, x0 = (d - th) // 2, (d - tw) // 2
    over(out[y0:y0 + th, x0:x0 + tw], t, INK if hot else WHITE)
    return _u8(out)


def dot_sprite(s, col):
    n = s + 6
    m = np.zeros((n, n), np.uint8)
    cv2.circle(m, (n * 8, n * 8), int(s * 8), 255, -1, cv2.LINE_AA, 4)
    out = np.zeros((n, n, 4), np.float32)
    over(out, cv2.dilate(m, np.ones((3, 3), np.uint8)), INK, 0.9)
    over(out, m, col)
    return _u8(out)


def _disc(n, c, r, thick=-1):
    m = np.zeros((n, n), np.uint8)
    cv2.circle(m, (int(round(c * 16)), int(round(c * 16))), int(round(r * 16)), 255, thick, cv2.LINE_AA, 4)
    return m


def _dir(deg, r, c):
    """Point at compass bearing `deg` (0 = north = up) and distance r from the centre (c, c)."""
    a = math.radians(deg)
    return c + r * math.sin(a), c - r * math.cos(a)


def _circ(m, org, x, y, r, thick=-1):
    cv2.circle(m, (int(round((x - org[0]) * 16)), int(round((y - org[1]) * 16))), int(round(r * 16)), 255, thick, cv2.LINE_AA, 4)


def _poly(m, org, pts, grow=0):
    cv2.fillPoly(m, [P(np.asarray(pts, float) - org)], 255, cv2.LINE_AA, 4)
    if grow:
        m[:] = cv2.dilate(m, np.ones((grow + 1, grow + 1), np.uint8))


def _stamp(lay, cx, cy, half, *parts):
    """Composite small shapes onto the float layer inside a window around (cx, cy): parts = (draw(mask, window origin), colour,
    opacity), in order. Cheap next to a full-layer composite, and a radar has a few dozen of these."""
    x0, y0 = max(int(cx - half), 0), max(int(cy - half), 0)
    x1, y1 = min(int(cx + half) + 2, lay.shape[1]), min(int(cy + half) + 2, lay.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    win, org = lay[y0:y1, x0:x1], np.array([x0, y0], float)
    for draw, col, op in parts:
        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
        draw(m, org)
        over(win, m, col, op)


def _put(lay, sp, cx, cy, op=1.0):
    """Composite a premultiplied BGRA sprite centred on (cx, cy) onto the float layer."""
    h, w = sp.shape[:2]
    x, y = int(round(cx - w / 2)), int(round(cy - h / 2))
    x0, y0, x1, y1 = max(x, 0), max(y, 0), min(x + w, lay.shape[1]), min(y + h, lay.shape[0])
    if x1 <= x0 or y1 <= y0:
        return
    src = sp[y0 - y:y1 - y, x0 - x:x1 - x].astype(np.float32) * op
    win = lay[y0:y1, x0:x1]
    win *= 1.0 - src[..., 3:4] / 255.0
    win += src


@functools.lru_cache(maxsize=32)
def vault_sprite(s, hollow=False):
    """A vault door: steel-blue plate, round door with a wheel. hollow = a vault you may have no key for."""
    pad = max(2, s // 12)
    n = s + 2 * pad
    c = n / 2.0
    plate = np.zeros((n, n), np.uint8)
    cv2.fillPoly(plate, [P([[pad + .08 * s, pad + .08 * s], [pad + .92 * s, pad + .08 * s], [pad + .92 * s, pad + .92 * s], [pad + .08 * s, pad + .92 * s]])], 255, cv2.LINE_AA, 4)
    r = max(1, int(round(s / 13)))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    out = np.zeros((n, n, 4), np.float32)
    if hollow:
        over(out, plate, INK, 0.30)
        over(out, cv2.dilate(plate, k) - cv2.erode(plate, k), WHITE, 0.95)
        over(out, _disc(n, c, 0.30 * s, max(1, s // 14)), WHITE, 0.8)
        return _u8(out)
    shade, top, lit = PALETTE["vault"]
    over(out, cv2.dilate(plate, k), INK, 0.97)
    over(out, plate, shade)
    over(out, _disc(n, c, 0.31 * s), top)
    over(out, _disc(n, c, 0.31 * s, max(1, s // 14)), lit)
    spokes = np.zeros((n, n), np.uint8)
    for deg in (0, 90):
        a, b = _dir(deg, 0.22 * s, c), _dir(deg + 180, 0.22 * s, c)
        cv2.line(spokes, tuple(int(round(v * 16)) for v in a), tuple(int(round(v * 16)) for v in b), 255, max(1, s // 10), cv2.LINE_AA, 4)
    over(out, spokes, lit)
    over(out, _disc(n, c, 0.08 * s), LOCK)
    return _u8(out)


def gg_icon(icon_id, size):
    """fortnite.gg's map icon `icon_id`, trimmed and scaled so its longer side is `size` px, as a premultiplied BGRA sprite;
    None when the file is not there (the drawn chest sprites take over)."""
    return _gg_icon(int(icon_id), int(size))


@functools.lru_cache(maxsize=96)
def _gg_icon(icon_id, size):
    im = cv2.imread(os.path.join(ICON_DIR, "%d.png" % icon_id), cv2.IMREAD_UNCHANGED)
    if im is None or im.ndim != 3 or im.shape[2] != 4:
        return None
    x, y, w, h = cv2.boundingRect((im[..., 3] > 8).astype(np.uint8))
    im = im[y:y + h, x:x + w]
    f = size / max(w, h)
    im = cv2.resize(im, (max(1, int(round(w * f))), max(1, int(round(h * f)))), interpolation=cv2.INTER_AREA if f < 1 else cv2.INTER_CUBIC)
    a = im[..., 3:].astype(np.float32) / 255.0
    return _u8(np.concatenate([im[..., :3].astype(np.float32) * a, im[..., 3:].astype(np.float32)], axis=2))


def radar_icon(kind, sure, size):
    """The sprite for one chest on the radar: fortnite.gg's icon for its kind, else the drawn one."""
    # the gold outline is fortnite.gg's "100% spawn" mark: it follows the chest's KIND, not `sure` (route mode marks every route chest sure)
    iid = GG_ICON.get(kind) or (1327 if kind == "chest_sure" else 3 if kind == "buried" and sure else 2)
    return gg_icon(iid, size)


@functools.lru_cache(maxsize=4)
def _radial(n, c, inner, outer):
    """Opacity ramp across the layer: 1 at the centre fading to `inner`/`outer` (used for the wedge and the glass)."""
    y, x = np.mgrid[0:n, 0:n].astype(np.float32)
    return np.clip(np.hypot(x - c, y - c) / max(outer, 1e-3), 0, 1)


@functools.lru_cache(maxsize=4)
def radar_glass(R):
    """The part of the radar that never moves: a soft shadow and dark glass with a faint sheen and a hairline edge. No bezel, no
    colour: the chests, your fan and the north marker are what should catch the eye."""
    c = R + max(6, int(R * 0.22))
    n = 2 * c + 1
    out = np.zeros((n, n, 4), np.float32)
    sh = cv2.GaussianBlur(_disc(n, c + R * 0.05, R), (0, 0), R * 0.06)
    over(out, sh, (0, 0, 0), 0.38)
    r = _radial(n, c, 0.0, R)
    face = _disc(n, c, R).astype(np.float32) / 255.0
    over(out, ((0.50 + 0.22 * r) * face * 255).astype(np.uint8), (36, 31, 27), 1.0)
    spec = cv2.GaussianBlur(_disc(n, c - R * 0.36, R * 0.44), (0, 0), R * 0.13)
    over(out, (spec.astype(np.float32) * face).astype(np.uint8), WHITE, 0.055)
    over(out, _disc(n, c, R - 0.5, 1), WHITE, 0.30)                                  # hairline edge
    over(out, _disc(n, c, R + 0.8, 1), INK, 0.35)
    return _u8(out)


@functools.lru_cache(maxsize=24)
def radar_rings(R, rm10, px):
    """The two range rings and their labels for a range of rm10 / 10 metres (cached: it only changes while the range is zooming)."""
    rm = rm10 / 10.0
    c = R + max(6, int(R * 0.22))
    n = 2 * c + 1
    Rin = 0.80 * R
    out = np.zeros((n, n, 4), np.float32)
    for f, lab in ((0.5, rm * 0.5), (1.0, rm)):
        over(out, _disc(n, c, Rin * f, 1), WHITE, 0.16 if f < 1 else 0.28)
        sp = text_sprite(("%g" % round(lab, 1)) + (" m" if f == 1.0 else ""), px, WHITE, UI_FONT)
        x, y = _dir(143, Rin * f, c)
        _put(out, sp, x + sp.shape[1] * 0.42, y + sp.shape[0] * 0.15, 0.62)
    return _u8(out)


def row(parts, gap):
    """Sprites side by side, centred vertically, into one sprite."""
    w = sum(p.shape[1] for p in parts) + gap * (len(parts) - 1)
    h = max(p.shape[0] for p in parts)
    out = np.zeros((h, w, 4), np.uint8)
    x = 0
    for p in parts:
        blit(out, p, x, (h - p.shape[0]) / 2)
        x += p.shape[1] + gap
    return out


def dist_text(d):
    return "%.1f m" % d if d < 10 else "%d m" % round(d)


class Hud:
    def __init__(self, W, H):
        self.W, self.H, self.u = W, H, H / 1440.0
        self.cache = {}       # finished labels, keyed by what they say
        self.dirty = []       # rectangles drawn into the canvas last frame (wiped at the start of the next)
        self.changed = (0, 0, W, H)    # bounding box of everything that differs from the previous frame: all the window has to be told about

    # ---- labels
    def label(self, d, dz, size, sure, edge=None, side=None, kind="buried", step=0, hot=False):
        """Chest icon + distance (+ how far above / below you). edge 'left' / 'right': a turn-this-way chevron on the outer
        side. side 'up' / 'down': the arrow is shown even for a small height difference. -> cached sprite."""
        level = abs(dz) >= LEVEL_M
        key = (dist_text(d), round(dz) if level else 0, size, sure, edge, side, kind, step, hot)
        sp = self.cache.get(key)
        if sp is None:
            if len(self.cache) > 400:
                self.cache.clear()
            px = max(12, int(round(size * 0.66)))
            parts = [chest_sprite(size, not sure, kind), text_sprite(dist_text(d), px)]
            if step:
                parts.insert(0, badge_sprite(step, max(14, int(size * 0.62)), hot))
            if level or side:
                up = dz > 0 if level else side == "up"
                col = UP_COL if up else DOWN_COL
                parts.append(arrow_sprite(max(10, int(size * 0.42)), "up" if up else "down", col))
                if level:
                    parts.append(text_sprite("%d m" % round(abs(dz)), max(10, int(px * 0.78)), col))
            gap = max(4, int(size * 0.16))
            if edge:
                chev = arrow_sprite(max(10, int(size * 0.5)), edge)
                parts = parts + [chev] if edge == "right" else [chev] + parts
            sp = self.cache[key] = row(parts, gap)
        return sp

    def place(self, cv, sp, cx, cy, op=1.0, anchor="c"):
        """(cx, cy) = the sprite's centre, or its left / right edge for anchor 'l' / 'r'."""
        h, w = sp.shape[:2]
        x = cx if anchor == "l" else cx - w if anchor == "r" else cx - w / 2
        blit(cv, sp, x, cy - h / 2, op, self.dirty)

    # ---- one chest
    def marker(self, cv, m, t, nearest=False):
        u, x, y, d = self.u, m["x"], m["y"], m["dist"]
        tx, ty = m["top"]
        sure, op = m.get("sure", True), fade(m["dist"])
        size = int(round(float(np.clip(58 - 0.40 * d, 30, 58)) * u * (1.18 if nearest else 1.0)))
        lw = max(1, int(round(2 * u)))
        if sure:
            ring = m.get("ring")
            if ring and d < RING_M:                                        # where to dig
                pts = [P(ring)]
                cv2.polylines(cv, pts, True, (0, 0, 0, int(255 * 0.5 * op)), lw + 2, cv2.LINE_AA, 4)
                cv2.polylines(cv, pts, True, (*[int(235 * op)] * 3, int(235 * op)), lw, cv2.LINE_AA, 4)
                self.dirty.append((int(min(p[0] for p in ring)) - 6, int(min(p[1] for p in ring)) - 6, int(max(p[0] for p in ring)) + 7, int(max(p[1] for p in ring)) + 7))
            op = op * (1.0 if d < 35 else 0.55)                              # far stems only hint at the direction
            a, b = (int(round(tx)), int(round(ty))), (int(round(x)), int(round(y)))        # thin stem from the label to the spot
            cv2.line(cv, a, b, (0, 0, 0, int(255 * 0.45 * op)), lw + 2, cv2.LINE_AA)
            cv2.line(cv, a, b, (*[int(235 * op)] * 3, int(235 * op)), lw, cv2.LINE_AA)
            r = max(2, int(round(3 * u)))
            cv2.circle(cv, b, r + 1, (0, 0, 0, int(255 * 0.5 * op)), -1, cv2.LINE_AA)
            cv2.circle(cv, b, r, (*[int(240 * op)] * 3, int(240 * op)), -1, cv2.LINE_AA)
            self.dirty.append((min(a[0], b[0]) - lw - 4, min(a[1], b[1]) - lw - 4, max(a[0], b[0]) + lw + 5, max(a[1], b[1]) + lw + 5))
        op = fade(d)
        sp = self.label(d, m.get("dz", 0.0), size, sure, kind=m.get("kind", "buried"), step=m.get("step", 0), hot=bool(m.get("next")))
        self.place(cv, sp, tx, ty - sp.shape[0] / 2 - 4 * u, op if sure else op * 0.85)

    # ---- radar
    def _range(self, hd, st, t):
        """Metres from the centre to the rim. Fixed when the config says so; otherwise the smallest level that holds the target
        (with hysteresis, and eased over ~0.15 s), so walking up to a chest zooms the radar in until a pixel is about 5 cm."""
        fixed = float(st.get("radar_m", 0.0))
        if fixed > 0:
            self.lvl = self.rng = fixed
            return fixed
        L = getattr(self, "lvl", None)
        if L is None or L not in RADAR_LEVELS:
            L = next((v for v in RADAR_LEVELS if hd <= 0.92 * v), RADAR_LEVELS[-1])
            self.rng = None
        else:
            i = RADAR_LEVELS.index(L)
            while i < len(RADAR_LEVELS) - 1 and hd > 0.97 * RADAR_LEVELS[i]:
                i += 1
            while i > 0 and hd < 0.42 * RADAR_LEVELS[i]:
                i -= 1
            L = RADAR_LEVELS[i]
        self.lvl = L
        last = getattr(self, "rng_t", None)
        dt = 1.0 if last is None or getattr(self, "rng", None) is None else min(max(t - last, 0.0), 0.1)
        self.rng_t = t
        cur = getattr(self, "rng", None)
        self.rng = L if cur is None else cur + (L - cur) * min(1.0, dt * 9.0)
        return float(self.rng)

    def radar(self, cv, st, t=0.0):
        """Heading-up radar (config radar_up False = north up): what is ahead of you is up. The nearest chests are fortnite.gg's own
        icons at their exact bearing and flat distance, the fan is what your screen shows, the rim turns so the gold marker is always
        north. The next route chest, or else the nearest sure one, is larger with a thin ring, its distance under the dial, and sets the
        range; ring labels say how far the rings are. Route mode adds a gold line through the next few chests."""
        u = self.u
        R = max(40, int(round(RADAR_R * self.H)))
        c = R + max(6, int(R * 0.22))
        n = 2 * c + 1
        Rin = 0.80 * R                                                                    # the working area; the rim ring sits outside it
        yaw = st["yaw"]
        rot = yaw if st.get("radar_up", True) else 0.0                                    # the bearing that points up
        lay = np.zeros((n, n, 4), np.float32)
        live = [m for m in st["markers"] if not m["dug"] and "hd" in m]
        sure = [m for m in live if m.get("sure", True)]
        hot = next((m for m in live if m.get("next")), None) or min(sure, key=lambda m: m["hd"], default=None)
        rm = self._range(RADAR_LEVELS[2] if hot is None else hot["hd"], st, t)           # the range holds the target: close in, the radar is zoomed in
        k = Rin / rm
        ahead = {id(m) for m in live if hot is not None and hot.get("step") and hot["step"] <= m.get("step", 0) <= hot["step"] + ROUTE_LEG}      # route chests to come
        live = [m for m in live if m["hd"] <= rm * 1.5 or id(m) in ahead]             # (those out of range sit on the rim, in their direction)
        sure = [m for m in sure if m["hd"] <= rm * 1.5]
        # fan: what the screen shows
        half = st.get("fov", 80.0) / 2
        wedge = [(c, c)] + [_dir(yaw - rot + a, Rin, c) for a in np.linspace(-half, half, 13)]
        m = np.zeros((n, n), np.uint8)
        cv2.fillPoly(m, [P(wedge)], 255, cv2.LINE_AA, 4)
        fan = m.astype(np.float32) * (0.22 * (1.0 - 0.85 * _radial(n, c, 0.0, Rin)))
        x, y, w, h = cv2.boundingRect(m)
        over(lay[y:y + h, x:x + w], fan[y:y + h, x:x + w].astype(np.uint8), WHITE, 1.0)
        # rim: ticks turn with you, a small gold pointer marks north
        rim = np.zeros((n, n, 4), np.uint8)
        for deg in range(0, 360, 10):
            major = deg % 90 == 0
            r0 = R * (0.905 if major else 0.935 if deg % 30 == 0 else 0.955)
            a, b = _dir(deg - rot, r0, c), _dir(deg - rot, R * 0.985, c)
            cv2.line(rim, tuple(int(round(v * 16)) for v in a), tuple(int(round(v * 16)) for v in b), (115, 115, 115, 115), 2 if major else 1, cv2.LINE_AA, 4)
        nx, ny = _dir(0 - rot, R * 0.93, c)
        tri = [_dir(0 - rot, R * 0.985, c), _dir(0 - rot + 3.6, R * 0.895, c), _dir(0 - rot - 3.6, R * 0.895, c)]
        _stamp(lay, nx, ny, R * 0.12, (lambda q, o: _poly(q, o, tri, 1), INK, 0.7), (lambda q, o: _poly(q, o, tri), GOLD_LIT, 1.0))
        # chests
        inside = sorted(live, key=lambda m: m["hd"] + (0 if m.get("sure", True) else 25.0))        # a maybe-chest counts as farther than a sure one
        shown = [m for m in inside if m["hd"] <= rm][:RADAR_ICONS] + [m for m in sorted(sure, key=lambda m: m["hd"]) if m["hd"] > rm][:3]
        for m in live:
            if (m is hot or id(m) in ahead) and m not in shown:
                shown.append(m)
        pos = {id(m): _dir(m["brg"] - rot, Rin * 0.97 if m["hd"] > rm else m["hd"] * k, c) for m in shown}
        route = sorted((m for m in shown if m.get("step")), key=lambda m: m["step"])
        if hot is not None and hot.get("step"):                                           # route mode: the way through the next few chests
            path = [(c, c)] + [pos[id(m)] for m in route if m["step"] >= hot["step"]][:ROUTE_LEG]
            if len(path) > 1:
                m = np.zeros((n, n), np.uint8)
                cv2.polylines(m, [P(path)], False, 255, max(3, int(round(3.4 * u))), cv2.LINE_AA, 4)
                x, y, w, h = cv2.boundingRect(m)
                over(lay[y:y + h, x:x + w], m[y:y + h, x:x + w], INK, 0.5)
                m[:] = 0
                cv2.polylines(m, [P(path)], False, 255, max(1, int(round(1.6 * u))), cv2.LINE_AA, 4)
                over(lay[y:y + h, x:x + w], m[y:y + h, x:x + w], GOLD_LIT, 0.9)
        isz = max(12, int(round(22 * u)))
        for m in sorted(shown, key=lambda m: -m["hd"]):                                   # far first, near ones on top
            far = m["hd"] > rm
            x, y = pos[id(m)]
            sz = int(round(isz * (1.28 if m is hot else 0.8 if far else 1.0)))
            kind, ok = m.get("kind", "buried"), m.get("sure", True)
            ic = radar_icon(kind, ok, sz)
            if ic is None:                                                                # no icon files: the drawn sprites
                ic = vault_sprite(sz, not ok) if kind == "vault" else chest_sprite(sz, not ok, kind)
            _put(lay, ic, x, y, (0.5 if far else 1.0) * (1.0 if ok else 0.62))
            if abs(m.get("dz", 0.0)) >= LEVEL_M and not far:                              # on another floor: a small arrow beside the icon
                s3, up = 3.2 * u, m["dz"] > 0
                ax, ay = x + ic.shape[1] / 2 + 2.5 * u, y - ic.shape[0] * 0.18
                tri = [[ax, ay - s3], [ax + s3, ay + s3], [ax - s3, ay + s3]] if up else [[ax, ay + s3], [ax + s3, ay - s3], [ax - s3, ay - s3]]
                _stamp(lay, ax, ay, 8 * u, (lambda q, o: _poly(q, o, tri, 2), INK, 0.9), (lambda q, o: _poly(q, o, tri), UP_COL if up else DOWN_COL, 1.0))
        if hot is not None and id(hot) in pos:
            x, y = pos[id(hot)]
            rr = isz * 1.28 * 0.72
            _stamp(lay, x, y, rr + 4 * u, (lambda q, o: _circ(q, o, x, y, rr, max(1, int(round(1.3 * u)))), WHITE, 0.85))
            hx, hy = x, y                                                                  # a fine cross on the exact spot
            _stamp(lay, hx, hy, 6 * u, (lambda q, o: _circ(q, o, hx, hy, 1.4 * u), WHITE, 1.0))
        # how precise the fix is: you are somewhere inside this disc
        sg = st.get("sigma_m")
        if sg:
            rs = max(2.5 * u, sg * k)
            _stamp(lay, c, c, rs + 3, (lambda q, o: _circ(q, o, c, c, rs), WHITE, 0.10), (lambda q, o: _circ(q, o, c, c, rs, 1), WHITE, 0.32))
        # you: a small arrow, always up in heading-up mode
        ar = yaw - rot
        arrow = [_dir(ar, 0.14 * R, c), _dir(ar + 148, 0.10 * R, c), _dir(ar + 180, 0.045 * R, c), _dir(ar - 148, 0.10 * R, c)]
        _stamp(lay, c, c, 0.2 * R, (lambda q, o: _poly(q, o, arrow, 2), INK, 0.9), (lambda q, o: _poly(q, o, arrow), WHITE, 1.0))
        at = radar_spot(st.get("radar_at"))
        cx, cy = at[0] * self.W, at[1] * self.H
        self.place(cv, radar_glass(R), cx, cy)
        self.place(cv, radar_rings(R, int(round(rm * 10)), max(9, int(round(10.5 * u)))), cx, cy)
        self.place(cv, rim, cx, cy)
        self.place(cv, _u8(lay), cx, cy)
        if hot is not None:
            px = max(12, int(round(21 * u)))
            key = ("r", dist_text(hot["dist"]), px)
            sp = self.cache.get(key)
            if sp is None:
                sp = self.cache[key] = text_sprite(dist_text(hot["dist"]), px, WHITE, UI_FONT)
            self.place(cv, sp, cx, cy + R + px * 1.05)

    # ---- the in-game full map (M)
    MAP_PRIO = {"vault": 0, "rare": 1, "sprite": 2, "chest_sure": 3, "buried": 4, "chest": 5}
    MAP_ICONS, MAP_DOTS = 420, 1500

    def fullmap(self, cv, st, t):
        """Chests drawn ON the open full map. st["fullmap"] = dict(M = fortnite.gg z4 map px -> screen px (2x3), items = Pipeline.map_items()).
        Zoomed out the icons would pile up, so they are thinned (the better kind wins a crowded spot, the rest become dots) and grow
        with the zoom. Route mode: every chest is still drawn; the route's are numbered and joined by a gold line, with a ring on the next one."""
        fm, u, W, H = st["fullmap"], self.u, self.W, self.H
        M, it = fm["M"], fm["items"]
        sc = float(np.hypot(M[0, 0], M[1, 0]))
        xy = it["xy"] @ M[:, :2].T + M[:, 2]
        pad = 30
        vis = (xy[:, 0] > -pad) & (xy[:, 0] < W + pad) & (xy[:, 1] > -pad) & (xy[:, 1] < H + pad) & ~it["dug"]
        route = bool(it["step"].any())
        isz = int(round(min(max(9 + 20 * sc, 12), 32) * u))
        cell = max(4, int(isz * 0.95))
        order = np.nonzero(vis)[0]
        prio = np.array([self.MAP_PRIO.get(str(k), 5) + (0 if s else 0.5) for k, s in zip(it["kind"][order], it["sure"][order])])
        prio -= 10.0 * (it["step"][order] > 0)                                         # route chests first: they are never thinned away
        order = order[np.argsort(prio, kind="stable")]
        taken, icons, dots, rest = set(), 0, 0, []
        if route:                                                                    # the way, under the icons
            steps = np.argsort(it["step"])
            steps = steps[it["step"][steps] > 0]
            pts = xy[steps]
            if len(pts) > 1:
                x0, y0 = max(int(pts[:, 0].min()) - 8, 0), max(int(pts[:, 1].min()) - 8, 0)
                x1, y1 = min(int(pts[:, 0].max()) + 9, W), min(int(pts[:, 1].max()) + 9, H)
                if x1 > x0 and y1 > y0:
                    q = P(pts - [x0, y0])
                    lay = np.zeros((y1 - y0, x1 - x0, 4), np.float32)
                    for width, col, op in ((max(4, int(round(5 * u))), INK, 0.55), (max(2, int(round(2.4 * u))), GOLD_LIT, 0.95)):
                        m = np.zeros((y1 - y0, x1 - x0), np.uint8)
                        cv2.polylines(m, [q], False, 255, width, cv2.LINE_AA, 4)
                        over(lay, m, col, op)
                    blit(cv, _u8(lay), x0, y0, 1.0, self.dirty)
        for i in order:
            x, y = xy[i]
            kind, ok, stp = str(it["kind"][i]), bool(it["sure"][i]), int(it["step"][i])
            key = (int(x // cell), int(y // cell))
            if not stp and (key in taken or icons >= self.MAP_ICONS):
                rest.append(i)
                continue
            taken.add(key)
            sz = int(round(isz * 1.3)) if stp else isz                                    # the route stands out from the rest
            ic = radar_icon(kind, ok or bool(stp), sz)
            if ic is None:
                ic = vault_sprite(sz, not ok) if kind == "vault" else chest_sprite(sz, not ok, kind)
            self.place(cv, ic, x, y, 1.0 if ok or stp else 0.7)
            icons += 1
            if stp:
                hot = int(i) == int(it["next"])
                bs = max(10, int(round((19 if hot else 16) * u)))
                self.place(cv, badge_sprite(stp, bs, hot), x + sz * 0.45, y - sz * 0.45)
        dcol = {k: PALETTE[k][1] for k in PALETTE}
        for i in rest:                                                                # what the icons had no room for
            if dots >= self.MAP_DOTS:
                break
            x, y = xy[i]
            kind = str(it["kind"][i])
            r = max(2, int(round(2.2 * u + sc)))
            key = (int(x // (r * 3)), int(y // (r * 3)), 1)
            if key in taken:
                continue
            taken.add(key)
            self.place(cv, dot_sprite(r, dcol.get(kind, GOLD)), x, y, 0.9)
            dots += 1
        nxt = int(it["next"])
        if nxt >= 0 and vis[nxt]:                                                     # where to go next: a ring that breathes
            x, y = xy[nxt]
            rr = isz * (0.95 + 0.12 * math.sin(t * 5.0))
            m = np.zeros((int(rr * 2 + 10),) * 2, np.uint8)
            c0 = m.shape[0] / 2
            cv2.circle(m, (int(c0 * 16), int(c0 * 16)), int(rr * 16), 255, max(1, int(round(2 * u))), cv2.LINE_AA, 4)
            lay = np.zeros(m.shape + (4,), np.float32)
            over(lay, m, WHITE, 0.9)
            self.place(cv, _u8(lay), x, y)

    def _bbox(self, rects):
        if not rects:
            return (0, 0, 0, 0)
        a = np.asarray(rects)
        return (max(int(a[:, 0].min()), 0), max(int(a[:, 1].min()), 0), min(int(a[:, 2].max()), self.W), min(int(a[:, 3].max()), self.H))

    def _lines(self, text, px, maxw):
        """`text` broken at spaces into lines no wider than maxw px (a single word wider than that is left alone)."""
        lines, cur = [], ""
        for w in text.split():
            trial = (cur + " " + w).strip()
            if cur and text_mask(trial, px, UI_FONT).shape[1] > maxw:
                lines.append(cur)
                cur = w
            else:
                cur = trial
        return lines + [cur] if cur else lines

    def _chrome(self, cv, status, dot, toast):
        """The status line (bottom left, shrunk to fit) and the toast (bottom centre, wrapped to at most 70% of the screen width)."""
        u, W, H = self.u, self.W, self.H
        if status:
            col = tuple(int(v) for v in dot)
            key = ("s", status, col, W)
            sp = self.cache.get(key)
            if sp is None:
                px, maxw = max(12, int(15 * u)), int(0.40 * W)
                while px > 10 and text_mask(status, px, UI_FONT).shape[1] > maxw:
                    px -= 1                                                    # a long status gets smaller rather than running off the screen
                text = status
                while len(text) > 4 and text_mask(text, px, UI_FONT).shape[1] > maxw:
                    text = text[:-2].rstrip() + "…" if not text.endswith("…") else text[:-2] + "…"       # still too long at the smallest size: cut it
                sp = self.cache[key] = row([dot_sprite(max(6, int(px * 0.5)), col), text_sprite(text, px, WHITE, UI_FONT)], max(5, int(px * 0.5)))
            self.place(cv, sp, 24 * u, H - 30 * u, 0.9, "l")
        if toast:
            key = ("t", toast, W)
            sps = self.cache.get(key)
            if sps is None:
                px = max(14, int(22 * u))
                sps = self.cache[key] = [text_sprite(l, px, WHITE, UI_FONT) for l in self._lines(toast, px, int(0.70 * W))]
            y = H - 190 * u - 0.5 * (len(sps) - 1) * sps[0].shape[0] * 1.05 if sps else 0
            for sp in sps:
                self.place(cv, sp, W / 2, y)
                y += sp.shape[0] * 1.05

    # ---- everything
    def render(self, cv, st, status, dot, toast="", t=None):
        t = time.time() if t is None else t
        u, W, H = self.u, self.W, self.H
        prev = self.dirty
        for x0, y0, x1, y1 in prev:                                        # wipe only what the last frame touched
            cv[max(y0, 0):max(y1, 0), max(x0, 0):max(x1, 0)] = 0
        self.dirty = []
        if st.get("fullmap"):                                              # the game's full map is open: chests on it, nothing else
            self.fullmap(cv, st, t)
            self.dirty.append((0, 0, W, H))
            self._chrome(cv, status, dot, toast)
            self.changed = self._bbox(prev + self.dirty)
            return
        edge, vert, on_screen = [], [], []
        for m in (st["markers"] if st.get("markers_3d", True) else ()):        # radar-only mode: nothing is drawn over the world
            if m["dug"] or not m.get("show", True):
                continue
            x_ok = m["front"] and (-80 <= m["x"] <= W + 80 or -80 <= m["top"][0] <= W + 80)
            lo, hi = min(m["y"], m["top"][1]), max(m["y"], m["top"][1])
            if x_ok and hi >= -40 and lo <= H + 80:
                on_screen.append(m)
            elif x_ok:
                vert.append(m)         # ahead of you but above / below the screen: another floor, or you look steeply up / down
            else:
                edge.append(m)
        near = min((m["dist"] for m in on_screen if m.get("sure", True)), default=None)
        for m in sorted(on_screen, key=lambda m: -m["dist"]):              # far first so near ones sit on top
            self.marker(cv, m, t, nearest=bool(m.get("next")) or (near is not None and m["dist"] == near and m.get("sure", True)))
        if st.get("yaw") is not None:
            n0 = len(self.dirty)
            self.radar(cv, st, t)
            if st.get("stale"):                                            # no fix for a moment: the last radar stays, dimmed, instead of blinking out
                if len(self.dirty) > n0:                                   # (the radar's rects overlap: dim their union once)
                    x0, y0, x1, y1 = self._bbox(self.dirty[n0:])
                    reg = cv[y0:y1, x0:x1]
                    reg[:] = (reg.astype(np.float32) * 0.5).astype(np.uint8)
        s = int(round(36 * u))
        for i, m in enumerate(sorted(edge, key=lambda m: m["dist"])[:3]):  # off-screen: which way to turn
            right = m["rel"] > 0
            sp = self.label(m["dist"], m.get("dz", 0.0), s, m.get("sure", True), "right" if right else "left", kind=m.get("kind", "buried"), step=m.get("step", 0), hot=bool(m.get("next")))
            self.place(cv, sp, (W - 22 * u) if right else 22 * u, H * 0.36 + i * 66 * u, max(fade(m["dist"]), 0.85), "r" if right else "l")
        for side, y in (("up", 0.17 * H), ("down", 0.80 * H)):             # off the top / bottom: another floor
            ms = sorted((m for m in vert if (m["y"] < 0) == (side == "up")), key=lambda m: m["dist"])[:3]
            last = -1e9
            for m in sorted(ms, key=lambda m: m["x"]):
                sp = self.label(m["dist"], m.get("dz", 0.0), s, m.get("sure", True), None, side, m.get("kind", "buried"), m.get("step", 0), bool(m.get("next")))
                x = max(min(max(m["x"], 0.2 * W), 0.8 * W), last + sp.shape[1] + 24 * u)
                last = x
                self.place(cv, sp, x, y, max(fade(m["dist"]), 0.85))
        self._chrome(cv, status, dot, toast)
        for x0, y0, x1, y1 in KEEP_CLEAR:                                  # the app's own read areas stay transparent
            k = W / 2000.0
            cv[int(y0 * k):int(y1 * k), int(x0 * k):int(x1 * k)] = 0
        self.changed = self._bbox(prev + self.dirty)
