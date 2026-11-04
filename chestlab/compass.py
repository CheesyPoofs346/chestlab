"""Heading from the compass bar: the game prints your exact heading as a big white number at top centre.
Reads it with Windows' built-in OCR (via winocr); tries several preprocessings because the OCR engine is picky
about stroke thickness and size, and returns the first answer that is a plausible bearing."""
import asyncio
import re

import cv2
import numpy as np
from winocr import recognize_cv2

# number box, in a 2000-px-wide frame (scaled to the real frame width)
BOX = (965, 8, 1037, 34)
_NUM = re.compile(r"^\d{1,3}$")


REGION = (770, 4, 1230, 40)   # the part of the screen the compass sensor captures: big number + label row (2000-px scale)


def region_px(width):
    """Screen-pixel (left, top, width, height) of REGION for a frame `width` px wide."""
    s = width / 2000.0
    x0, y0, x1, y1 = (int(round(v * s)) for v in REGION)
    return x0, y0, x1 - x0, y1 - y0


def _crop(frame, k, s=None, ox=0, oy=0):
    """The number box scaled to `k` px per 2000-scale px. frame may be a whole frame or the REGION capture, whose
    top-left corner sits at screen pixel (ox, oy)."""
    s = frame.shape[1] / 2000.0 if s is None else s
    x0, y0, x1, y1 = (int(round(v * s)) for v in BOX)
    return cv2.resize(frame[y0 - oy:y1 - oy, x0 - ox:x1 - ox], None, fx=k / s, fy=k / s, interpolation=cv2.INTER_CUBIC)


def _white(c):
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    return ((hsv[..., 2] > 215) & (hsv[..., 1] < 55)).astype(np.uint8) * 255


def _pad(a):
    return cv2.cvtColor(cv2.copyMakeBorder(a, 40, 40, 40, 40, cv2.BORDER_CONSTANT, value=255), cv2.COLOR_GRAY2BGR)


def _gray(c):
    hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV)
    v = hsv[..., 2].astype(np.float32) - hsv[..., 1] * 0.6
    return _pad(255 - np.clip((v - 150) * 3, 0, 255).astype(np.uint8))


def _closed(c):
    return _pad(255 - cv2.morphologyEx(_white(c), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)))


def _thr(c, chan, t, close=False):
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) if chan is None else c[..., chan]
    b = ((g > t) * 255).astype(np.uint8)
    return _pad(255 - (cv2.morphologyEx(b, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8)) if close else b))


PREPS = (
    lambda f, **g: _gray(_crop(f, 2, **g)),
    lambda f, **g: _pad(255 - _white(_crop(f, 3, **g))),
    lambda f, **g: _gray(_crop(f, 4, **g)),
    lambda f, **g: _pad(255 - _white(_crop(f, 5, **g))),
    lambda f, **g: _closed(_crop(f, 4, **g)),
    lambda f, **g: _gray(_crop(f, 3, **g)),
    lambda f, **g: _thr(_crop(f, 2, **g), None, 200),          # plain brightness cut: the storm-tint shots
    lambda f, **g: _thr(_crop(f, 3, **g), 1, 230, True),
    lambda f, **g: _gray(_crop(f, 6, **g)),                   # Windows OCR is moody about glyph size: bigger copies read the shots the above miss
    lambda f, **g: _gray(_crop(f, 7, **g)),
    lambda f, **g: _gray(_crop(f, 8, **g)),
    lambda f, **g: _gray(_crop(f, 5, **g)),
)


class CompassReader:
    def __init__(self, lang="en"):
        self.lang = lang
        self.loop = asyncio.new_event_loop()
        self.best = 5        # the preprocessing that worked last time goes first: most frames then cost one ~5 ms read

    def _ocr(self, img):
        return self.loop.run_until_complete(recognize_cv2(img, self.lang)).text.replace(" ", "")

    @staticmethod
    def _with_zero(img):
        """Windows' OCR answers nothing for a lone one- or two-digit number (a heading under 100) but reads the same digits fine inside a longer
        string: put a rendered '0' in front of them (black on white, as many px tall as the digits)."""
        dark = (img[..., 0] < 128).astype(np.uint8)
        n, _, st, _ = cv2.connectedComponentsWithStats(dark)
        if n < 2:
            return None
        hmax = st[1:, cv2.CC_STAT_HEIGHT].max()
        keep = [i for i in range(1, n) if st[i, cv2.CC_STAT_HEIGHT] >= 0.6 * hmax]          # the digits, not the caret under them
        if not keep:
            return None
        x0 = min(st[i, cv2.CC_STAT_LEFT] for i in keep)
        x1 = max(st[i, cv2.CC_STAT_LEFT] + st[i, cv2.CC_STAT_WIDTH] for i in keep)
        y0 = min(st[i, cv2.CC_STAT_TOP] for i in keep)
        y1 = max(st[i, cv2.CC_STAT_TOP] + st[i, cv2.CC_STAT_HEIGHT] for i in keep)
        h = int(y1 - y0)
        if h < 8:
            return None
        digits = img[y0:y1, x0:x1]
        thick = max(2, int(round(h / 6)))
        scale = 0.5
        while cv2.getTextSize("0", cv2.FONT_HERSHEY_DUPLEX, scale, thick)[0][1] < h and scale < 20:
            scale += 0.1
        (tw, th), _ = cv2.getTextSize("0", cv2.FONT_HERSHEY_DUPLEX, scale, thick)
        gap, m = int(h * 0.5), int(h * 0.6)
        out = np.full((h + 2 * m, m + tw + gap + digits.shape[1] + m, 3), 255, np.uint8)
        cv2.putText(out, "0", (m, m + h - 1), cv2.FONT_HERSHEY_DUPLEX, scale, (0, 0, 0), thick, cv2.LINE_AA)
        out[m:m + h, m + tw + gap:m + tw + gap + digits.shape[1]] = digits
        return out

    def _number(self, img):
        t = self._ocr(img)
        if _NUM.match(t) and int(t) < 360:
            return int(t)
        if t == "":
            z = self._with_zero(img)
            if z is not None:
                t = self._ocr(z)
                if len(t) >= 2 and t[0] == "0" and t[1:].isdigit() and int(t[1:]) < 100:
                    return int(t[1:])
        return None

    def read(self, frame, tries=None, **geo):
        """-> heading in degrees (0..359) or None. `geo`: s, ox, oy when frame is the REGION capture (see _crop).
        `tries` caps how many preprocessings to attempt (each ~5 ms), so a bad frame can't stall a 60 Hz loop."""
        order = [self.best] + [i for i in range(len(PREPS)) if i != self.best]
        for i in order[:tries or len(order)]:
            n = self._number(PREPS[i](frame, **geo))
            if n is not None:
                self.best = i
                return n
        return None


# ---- second opinion: the label row of the compass bar (numbers every 15 degrees, letters every 45)
LAB = np.arange(0, 360, 15)
LABEL_BOX = (780, 21, 1220, 37)                     # 2000-scale
CAL_CX, CAL_PPD = 999.0, 2.60                       # caret x and px per degree, fitted on the real screenshots
_K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))


class BarComb:
    """Heading from where the translucent label glyphs sit on the compass bar. Noisier than the big number
    (about 1 degree against real readouts) but it works when the number can't be read (cloud behind it, storm tint)
    and it never confuses digits. Needs a prior within a few degrees to pick the right label spacing."""

    def __init__(self, span=6.0, step=0.1):
        self.hs = np.arange(-span, span + 1e-9, step)
        self.step = step

    @staticmethod
    def profile(frame, s=None, ox=0, oy=0):
        s = frame.shape[1] / 2000.0 if s is None else s
        x0, y0, x1, y1 = LABEL_BOX
        c = frame[int(round(y0 * s)) - oy:int(round(y1 * s)) - oy, int(round(x0 * s)) - ox:int(round(x1 * s)) - ox]
        c = cv2.resize(c, (x1 - x0, y1 - y0), interpolation=cv2.INTER_AREA)
        g = c.astype(np.float32).mean(2)
        w = np.clip(g - cv2.morphologyEx(g, cv2.MORPH_OPEN, _K) - 3, 0, 40)     # narrow bright strokes only
        return cv2.GaussianBlur(w.sum(0)[None, :], (0, 0), 2.5)[0]

    def estimate(self, frame, prior, **geo):
        """-> (heading deg, quality) or None. quality = peak over median of the score curve (>= ~1.15 is trustworthy)."""
        p = self.profile(frame, **geo)
        h = prior + self.hs
        d = (LAB[None, :] - h[:, None] + 180.0) % 360.0 - 180.0
        x = CAL_CX - LABEL_BOX[0] + CAL_PPD * d
        ok = (x > 4) & (x < len(p) - 4)
        sc = np.where(ok, np.interp(x, np.arange(len(p)), p), 0.0).sum(1)
        i = int(np.argmax(sc))
        fr = 0.0
        if 0 < i < len(sc) - 1:
            a, b, c = sc[i - 1], sc[i], sc[i + 1]
            if a - 2 * b + c < -1e-9:
                fr = 0.5 * (a - c) / (a - 2 * b + c)
        med = float(np.median(sc))
        return (prior + self.hs[i] + fr * self.step) % 360.0, float(sc[i] / med) if med > 1e-6 else 0.0
