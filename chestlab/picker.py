"""The start-up route picker: a window with one card per route (a map thumbnail of its path, chests, seconds, chests/min), grouped in tabs.

The whole picture is drawn by `render()` (PIL, 2x supersampled, pure: no window needed, so it is tested and previewed as a PNG); `Picker`
is the state machine (hover, click, keys); `pick()` is the thin Tk shell that shows the picture in a window and feeds it mouse / keys.
pick(routes_info, last) -> "0" (no route) / "<menu number>" / None (window closed)."""
import functools
import os
import re
import sys

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .paths import BASE as ROOT
REF = os.path.join(ROOT, "assets", "gg", "full_z4.png")
W, H = 1120, 800
PAD, GAP, CARD_W, CARD_H = 28, 16, 532, 128
THUMB = (150, 108)
S = 2                                                         # supersampling: everything is drawn at 2x and shrunk
BG, CARD, CARD_HOT, CARD_SEL = (15, 17, 21), (28, 32, 39), (36, 41, 50), (40, 37, 27)           # RGB
GOLD, TEXT, MUTED, DIM, TRACK, GREEN = (242, 185, 40), (236, 239, 244), (139, 146, 158), (86, 92, 102), (44, 49, 58), (61, 255, 138)
FONTS = {"r": "C:/Windows/Fonts/segoeui.ttf", "s": "C:/Windows/Fonts/seguisb.ttf", "b": "C:/Windows/Fonts/segoeuib.ttf"}
TABS = ("Saved", "Solo", "2 players", "Quiet", "100% chests", "All chests")
SOLO_TEXT = "Quiet spots away from the main POIs"
GROUP_DESC = {0: "A route you brought yourself", 1: SOLO_TEXT, 2: "Two players, opposite directions", 3: "Long loop through a quiet corner", 4: "Only chests that always spawn", 5: "Every chest on the loop"}


@functools.lru_cache(maxsize=64)
def font(kind, px):
    return ImageFont.truetype(FONTS[kind], int(px * S))


def tab_of(r):
    return {"saved": 0, "solo": 1, "pair": 2, "quiet": 3}.get(r["group"], 4 if r["name"].startswith("100") else 5)


def title_of(r):
    """A readable title from the route's name ('Solo 60 s, quiet' -> '60 second run')."""
    n, g = r["name"], r["group"]
    if g == "saved":
        return n
    if g == "solo":
        return "%s second run" % re.search(r"(\d+)", n).group(1)
    if g == "pair":
        m = re.search(r"(\d+) s #(\d+): player (\d+)", n)
        return "Option %s · Player %s" % (m.group(2), m.group(3)) if m else n
    if g == "quiet":
        return "Near " + re.sub(r"'S\b", "'s", n.split("near ", 1)[-1])
    return "Loop " + n.split("#")[-1]


def describe(pipe):
    """Pipeline -> what the picker needs per route: name, group, chests, seconds, per_min and its chests' positions in z4 map px."""
    xy = pipe.map_items()["xy"]
    return [dict(name=r["name"], group=r["group"], n=len(r["idx"]), seconds=r["seconds"], per_min=r["per_min"], pts=xy[r["idx"]]) for r in pipe.routes]


# ---- thumbnails
@functools.lru_cache(maxsize=1)
def _ref_half():
    im = cv2.imread(REF)
    return None if im is None else cv2.resize(im, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)


def thumb(pts, w=THUMB[0] * S, h=THUMB[1] * S, ref=None):
    """The map around a route with its path drawn on it: RGB (h, w, 3)."""
    ref = _ref_half() if ref is None else ref
    pts = np.asarray(pts, float) / 2.0                               # z4 px -> the half-size map's px
    lo, hi = pts.min(axis=0), pts.max(axis=0)
    c, span = (lo + hi) / 2, np.maximum(hi - lo, 1.0) * 1.35
    bw = max(span[0], span[1] * w / h, 300.0)                         # never closer than ~330 m across
    bh = bw * h / w
    bw, bh = min(bw, ref.shape[1]), min(bh, ref.shape[0])
    x0 = int(np.clip(c[0] - bw / 2, 0, ref.shape[1] - bw))
    y0 = int(np.clip(c[1] - bh / 2, 0, ref.shape[0] - bh))
    im = cv2.resize(ref[y0:y0 + int(bh), x0:x0 + int(bw)], (w, h), interpolation=cv2.INTER_AREA)
    im = (im.astype(np.float32) * 0.8).astype(np.uint8)
    q = (pts - [x0, y0]) * (w / bw)
    qi = np.round(q * 8).astype(np.int32)
    cv2.polylines(im, [qi], False, (12, 12, 12), 8, cv2.LINE_AA, 3)
    cv2.polylines(im, [qi], False, (40, 185, 242), 4, cv2.LINE_AA, 3)
    for p in qi:
        cv2.circle(im, (int(p[0]), int(p[1])), int(3.5 * S * 8), (245, 245, 245), -1, cv2.LINE_AA, 3)
    c0 = (int(qi[0][0]), int(qi[0][1]))
    cv2.circle(im, c0, int(7 * S * 8), (138, 255, 61), -1, cv2.LINE_AA, 3)
    cv2.circle(im, c0, int(7 * S * 8), (14, 12, 10), 2, cv2.LINE_AA, 3)
    return cv2.cvtColor(im, cv2.COLOR_BGR2RGB)


# ---- layout and state
def layout(n_cards):
    tabs, x = [], PAD
    for lab in TABS:
        w = int(font("s", 15).getlength(lab) / S) + 40
        tabs.append((x, 96, x + w, 136))
        x += w + 10
    cards = [(PAD + (i % 2) * (CARD_W + GAP), 160 + (i // 2) * (CARD_H + GAP)) for i in range(n_cards)]
    cards = [(x, y, x + CARD_W, y + CARD_H) for x, y in cards]
    return dict(tabs=tabs, cards=cards, none=(W - PAD - 360, 738, W - PAD - 170, 778), start=(W - PAD - 150, 738, W - PAD, 778))


class Picker:
    """Which tab, which card, what is under the mouse. Pure state: the window just calls these."""

    def __init__(self, routes, last=None):
        self.routes = routes
        self.tab_ids = [[i for i, r in enumerate(routes) if tab_of(r) == t] for t in range(len(TABS))]
        self.last = last if last is not None and 0 <= last < len(routes) else None
        self.tab = tab_of(routes[self.last]) if self.last is not None else next((t for t, ids in enumerate(self.tab_ids) if ids), 0)
        self.sel = self.last if self.last is not None else (self.tab_ids[self.tab][0] if self.tab_ids[self.tab] else None)
        self.hover = None                    # ("tab", i) | ("card", route index) | ("none",) | ("start",)
        self.result = "unset"

    @property
    def ids(self):
        return self.tab_ids[self.tab]

    def hit(self, x, y):
        lay = layout(len(self.ids))
        for name in ("start", "none"):
            x0, y0, x1, y1 = lay[name]
            if x0 <= x < x1 and y0 <= y < y1:
                return (name,)
        for i, (x0, y0, x1, y1) in enumerate(lay["tabs"]):
            if x0 <= x < x1 and y0 <= y < y1:
                return ("tab", i)
        for k, (x0, y0, x1, y1) in enumerate(lay["cards"]):
            if x0 <= x < x1 and y0 <= y < y1:
                return ("card", self.ids[k])
        return None

    def set_tab(self, t):
        self.tab = t % len(TABS)
        if self.sel not in self.ids:
            self.sel = self.ids[0] if self.ids else None

    def move(self, x, y):
        old, self.hover = self.hover, self.hit(x, y)
        return old != self.hover

    def click(self, x, y, double=False):
        h = self.hit(x, y)
        if h is None:
            return
        if h[0] == "tab":
            self.set_tab(h[1])
        elif h[0] == "card":
            self.sel = h[1]
            if double:
                self.finish()
        elif h[0] == "none":
            self.result = "0"
        elif h[0] == "start":
            self.finish()

    def finish(self):
        self.result = str(self.sel + 1) if self.sel is not None else "0"

    def key(self, name):
        """Arrow keys move between cards (2 columns), Tab / Shift-Tab between groups, 1-5 jump to a group, Enter starts, Esc = no route."""
        ids = self.ids
        if name in ("Return", "KP_Enter"):
            self.finish()
        elif name == "Escape":
            self.result = "0"
        elif name in ("Tab", "ISO_Left_Tab"):
            self.set_tab(self.tab + (-1 if name == "ISO_Left_Tab" else 1))
        elif len(name) == 1 and name in "123456":
            self.set_tab(int(name) - 1)
        elif name in ("Left", "Right", "Up", "Down") and ids:
            k = ids.index(self.sel) if self.sel in ids else 0
            k += {"Left": -1, "Right": 1, "Up": -2, "Down": 2}[name]
            self.sel = ids[min(max(k, 0), len(ids) - 1)]


# ---- drawing
def render(p, thumbs):
    """Picker state -> BGR image (H, W, 3). `thumbs` = {route index: RGB array from thumb()}."""
    im = Image.new("RGB", (W * S, H * S), BG)
    d = ImageDraw.Draw(im)
    s = lambda *v: tuple(int(round(a * S)) for a in v)
    best = max((r["per_min"] for r in p.routes), default=1.0)

    def text(x, y, t, kind, px, col, anchor="ls"):
        d.text((x * S, y * S), t, font=font(kind, px), fill=col, anchor=anchor)

    text(PAD, 52, "Choose your route", "b", 30, TEXT)
    text(PAD, 78, "Pick one, press Enter. You can change it any time in the ChestLab settings window.", "r", 14, MUTED)
    if p.last is not None:
        text(W - PAD, 52, "last time: " + title_of(p.routes[p.last]), "r", 13, DIM, "rs")
    lay = layout(len(p.ids))
    for i, (x0, y0, x1, y1) in enumerate(lay["tabs"]):
        on, hot = i == p.tab, p.hover == ("tab", i)
        d.rounded_rectangle(s(x0, y0, x1, y1), radius=20 * S, fill=GOLD if on else CARD_HOT if hot else CARD)
        text((x0 + x1) / 2 - 0, (y0 + y1) / 2 + 5.5, TABS[i], "s", 15, (24, 20, 10) if on else TEXT if hot else MUTED, "ms")
        n = len(p.tab_ids[i])
        if n and not on:
            d.ellipse(s(x1 - 14, y0 + 6, x1 - 4, y0 + 16), fill=TRACK)
            text(x1 - 9, y0 + 14.5, str(n), "s", 10, MUTED, "ms")
    for k, (x0, y0, x1, y1) in enumerate(lay["cards"]):
        ri = p.ids[k]
        r = p.routes[ri]
        sel, hot = ri == p.sel, p.hover == ("card", ri)
        d.rounded_rectangle(s(x0, y0, x1, y1), radius=14 * S, fill=CARD_SEL if sel else CARD_HOT if hot else CARD)
        if sel:
            d.rounded_rectangle(s(x0, y0, x1, y1), radius=14 * S, outline=GOLD, width=2 * S)
        th = thumbs.get(ri)
        if th is not None:
            m = Image.new("L", (THUMB[0] * S, THUMB[1] * S), 0)
            ImageDraw.Draw(m).rounded_rectangle((0, 0, THUMB[0] * S - 1, THUMB[1] * S - 1), radius=9 * S, fill=255)
            im.paste(Image.fromarray(th), s(x0 + 10, y0 + 10), m)
        tx = x0 + 178
        text(tx, y0 + 34, title_of(r), "s", 20, TEXT)
        text(tx, y0 + 55, GROUP_DESC[tab_of(r)], "r", 13, MUTED)
        text(tx, y0 + 90, "%d" % r["n"], "s", 19, TEXT)
        w1 = font("s", 19).getlength("%d" % r["n"]) / S
        text(tx + w1 + 4, y0 + 90, "chests", "r", 12, MUTED)
        x2 = tx + w1 + 4 + font("r", 12).getlength("chests") / S + 18
        text(x2, y0 + 90, "%d" % round(r["seconds"]), "s", 19, TEXT)
        w2 = font("s", 19).getlength("%d" % round(r["seconds"])) / S
        text(x2 + w2 + 4, y0 + 90, "sec", "r", 12, MUTED)
        text(x1 - 20, y0 + 56, "%.1f" % r["per_min"], "b", 30, GOLD, "rs")
        text(x1 - 20, y0 + 76, "chests / min", "r", 12, MUTED, "rs")
        bx0, bx1, by = tx, x1 - 20, y0 + 108
        d.rounded_rectangle(s(bx0, by, bx1, by + 4), radius=2 * S, fill=TRACK)
        d.rounded_rectangle(s(bx0, by, bx0 + (bx1 - bx0) * r["per_min"] / best, by + 4), radius=2 * S, fill=GOLD if sel else (150, 120, 40))
        if ri == p.last:
            text(x1 - 20, y0 + 28, "LAST USED", "s", 10, DIM, "rs")
    for name, label, primary in (("none", "No route · all chests", False), ("start", "Start", True)):
        x0, y0, x1, y1 = lay[name]
        hot = p.hover == (name,)
        d.rounded_rectangle(s(x0, y0, x1, y1), radius=20 * S, fill=(GOLD if not hot else (255, 204, 70)) if primary else CARD_HOT if hot else CARD)
        text((x0 + x1) / 2, (y0 + y1) / 2 + 5.5, label, "s", 15, (24, 20, 10) if primary else TEXT, "ms")
    text(PAD, 764, "← → ↑ ↓ choose      Tab / 1-6 switch group      Enter start      Esc no route", "r", 13, DIM)
    out = im.resize((W, H), Image.LANCZOS)
    return cv2.cvtColor(np.asarray(out), cv2.COLOR_RGB2BGR)


# ---- the window (Tk, thin: everything above is testable without one)
def pick(routes, last=None):
    import tkinter as tk
    try:
        import ctypes
        ctypes.windll.shcore.SetProcessDpiAwareness(2)        # real pixels, not a blurry scaled bitmap
    except Exception:
        pass
    p = Picker(routes, last)
    thumbs = {}
    for i, r in enumerate(routes):
        try:
            thumbs[i] = thumb(r["pts"])
        except Exception:
            pass
    root = tk.Tk()
    root.title("Choose your route")
    root.resizable(False, False)
    fit = min(1.0, (root.winfo_screenheight() - 90) / H, (root.winfo_screenwidth() - 30) / W)       # a small screen gets a smaller picker, never a cut-off one
    ww, wh = int(W * fit), int(H * fit)
    root.geometry("%dx%d+%d+%d" % (ww, wh, (root.winfo_screenwidth() - ww) // 2, max((root.winfo_screenheight() - wh) // 2 - 20, 0)))
    root.configure(bg="#0f1115")
    cv = tk.Canvas(root, width=ww, height=wh, highlightthickness=0, bd=0, bg="#0f1115")
    cv.pack()
    img = {}

    def redraw():
        im = render(p, thumbs)
        if fit < 1.0:
            im = cv2.resize(im, (ww, wh), interpolation=cv2.INTER_AREA)
        ok, buf = cv2.imencode(".png", im)
        img["i"] = tk.PhotoImage(data=bytes(buf))
        cv.delete("all")
        cv.create_image(0, 0, anchor="nw", image=img["i"])
        if p.result != "unset":
            root.destroy()

    cv.bind("<Motion>", lambda e: p.move(e.x / fit, e.y / fit) and redraw())
    cv.bind("<Button-1>", lambda e: (p.click(e.x / fit, e.y / fit), redraw()))
    cv.bind("<Double-Button-1>", lambda e: (p.click(e.x / fit, e.y / fit, True), redraw()))
    root.bind("<Key>", lambda e: (p.key(e.keysym), redraw()))
    root.protocol("WM_DELETE_WINDOW", lambda: (setattr(p, "result", "0"), root.destroy()))
    redraw()
    root.attributes("-topmost", True)
    root.after(200, lambda: (root.focus_force(), root.attributes("-topmost", False)))
    root.mainloop()
    return None if p.result == "unset" else p.result
