"""Click-through, always-on-top layered window with per-pixel alpha. chestlab.hud draws into its pixel buffer and
UpdateLayeredWindow pushes it to the screen; no Tk painting, so edges are anti-aliased and screenshots capture it."""
import ctypes
import time
from ctypes import wintypes

import numpy as np

from .hud import Hud, hex_bgr

WS_POPUP = 0x80000000
WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_EX_TOOLWINDOW, WS_EX_LAYERED, WS_EX_NOACTIVATE = 0x8, 0x20, 0x80, 0x80000, 0x08000000
SW_HIDE, SW_SHOWNOACTIVATE = 0, 4
SWP_NOSIZE, SWP_NOMOVE, SWP_NOACTIVATE, HWND_TOPMOST = 0x1, 0x2, 0x10, -1
ULW_ALPHA, AC_SRC_OVER, AC_SRC_ALPHA = 2, 0, 1

u32, g32 = ctypes.windll.user32, ctypes.windll.gdi32
V = ctypes.c_void_p
u32.CreateWindowExW.argtypes = [wintypes.DWORD, wintypes.LPCWSTR, wintypes.LPCWSTR, wintypes.DWORD, ctypes.c_int, ctypes.c_int,
                                ctypes.c_int, ctypes.c_int, V, V, V, V]
u32.CreateWindowExW.restype = V
u32.GetDC.argtypes, u32.GetDC.restype = [V], V
u32.ReleaseDC.argtypes = [V, V]
u32.ShowWindow.argtypes = [V, ctypes.c_int]
u32.DestroyWindow.argtypes = [V]
u32.SetWindowPos.argtypes = [V, V, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, wintypes.UINT]
g32.CreateCompatibleDC.argtypes, g32.CreateCompatibleDC.restype = [V], V
g32.CreateDIBSection.argtypes = [V, V, wintypes.UINT, ctypes.POINTER(V), V, wintypes.DWORD]
g32.CreateDIBSection.restype = V
g32.SelectObject.argtypes, g32.SelectObject.restype = [V, V], V
g32.DeleteObject.argtypes = [V]
g32.DeleteDC.argtypes = [V]
u32.UpdateLayeredWindow.argtypes = [V, V, V, V, V, V, wintypes.DWORD, V, wintypes.DWORD]
u32.UpdateLayeredWindowIndirect.argtypes = [V, V]
u32.SetWindowDisplayAffinity.argtypes = [V, wintypes.DWORD]


class BITMAPINFOHEADER(ctypes.Structure):
    _fields_ = [("biSize", wintypes.DWORD), ("biWidth", wintypes.LONG), ("biHeight", wintypes.LONG), ("biPlanes", wintypes.WORD),
                ("biBitCount", wintypes.WORD), ("biCompression", wintypes.DWORD), ("biSizeImage", wintypes.DWORD),
                ("biXPelsPerMeter", wintypes.LONG), ("biYPelsPerMeter", wintypes.LONG), ("biClrUsed", wintypes.DWORD),
                ("biClrImportant", wintypes.DWORD)]


class ULWINFO(ctypes.Structure):
    """UPDATELAYEREDWINDOWINFO: like UpdateLayeredWindow but with a dirty rectangle, so only what changed is composited."""
    _fields_ = [("cbSize", wintypes.DWORD), ("hdcDst", V), ("pptDst", V), ("psize", V), ("hdcSrc", V), ("pptSrc", V),
                ("crKey", wintypes.DWORD), ("pblend", V), ("dwFlags", wintypes.DWORD), ("prcDirty", V)]


class RECT(ctypes.Structure):
    _fields_ = [("left", wintypes.LONG), ("top", wintypes.LONG), ("right", wintypes.LONG), ("bottom", wintypes.LONG)]


class BLENDFUNCTION(ctypes.Structure):
    _fields_ = [("BlendOp", ctypes.c_ubyte), ("BlendFlags", ctypes.c_ubyte), ("SourceConstantAlpha", ctypes.c_ubyte),
                ("AlphaFormat", ctypes.c_ubyte)]


class Overlay:
    def __init__(self, rect):
        self.visible = True        # the window is never hidden from capture: Snipping Tool / Print Screen see it
        self.hwnd = u32.CreateWindowExW(WS_EX_LAYERED | WS_EX_TRANSPARENT | WS_EX_TOOLWINDOW | WS_EX_NOACTIVATE | WS_EX_TOPMOST,
                                        "Static", "chest overlay", WS_POPUP, 0, 0, 16, 16, None, None, None, None)
        if not self.hwnd:
            raise OSError("could not create the overlay window")
        self.dc = g32.CreateCompatibleDC(None)
        self.bmp = None
        self.blend = BLENDFUNCTION(AC_SRC_OVER, 0, 255, AC_SRC_ALPHA)
        self.full, self.indirect = True, True        # full: the next update sends the whole buffer (first frame, after a resize)
        try:
            u32.SetWindowDisplayAffinity(self.hwnd, 0)   # never hidden from capture: Snipping Tool, Print Screen, OBS all see the overlay
        except Exception:
            pass
        self.move(rect)
        u32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE)

    def move(self, rect):
        """Follow the game window; rebuilds the pixel buffer only when the size changes."""
        self.left, self.top, w, h = rect
        if (w, h) != (getattr(self, "w", 0), getattr(self, "h", 0)):
            self.w, self.h = w, h
            if self.bmp:
                g32.SelectObject(self.dc, self.old)
                g32.DeleteObject(self.bmp)
            bmi = BITMAPINFOHEADER(ctypes.sizeof(BITMAPINFOHEADER), w, -h, 1, 32, 0, 0, 0, 0, 0, 0)
            bits = V()
            self.bmp = g32.CreateDIBSection(self.dc, ctypes.byref(bmi), 0, ctypes.byref(bits), None, 0)
            self.old = g32.SelectObject(self.dc, self.bmp)
            self.canvas = np.ctypeslib.as_array((ctypes.c_ubyte * (w * h * 4)).from_address(bits.value)).reshape(h, w, 4)
            self.hud = Hud(w, h)
            self.full = True
        u32.SetWindowPos(self.hwnd, HWND_TOPMOST, self.left, self.top, 0, 0, SWP_NOSIZE | SWP_NOACTIVATE)

    def raise_top(self):
        u32.SetWindowPos(self.hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE | SWP_NOACTIVATE)

    def toggle(self):
        self.visible = not self.visible
        u32.ShowWindow(self.hwnd, SW_SHOWNOACTIVATE if self.visible else SW_HIDE)

    def draw(self, st, status, toast, status_color="#ffffff"):
        self.hud.render(self.canvas, st, status, hex_bgr(status_color), toast)
        x0, y0, x1, y1 = self.hud.changed
        if not self.full and (x1 <= x0 or y1 <= y0):
            return                                              # nothing changed since the last frame: nothing to push
        screen = u32.GetDC(None)
        try:
            # always the whole buffer: UpdateLayeredWindowIndirect with a dirty rectangle returned success here but left the window showing the
            # first frame for ever (the start-up hint and "no minimap lock" stayed on screen while the radar ran underneath)
            pos, size, src = wintypes.POINT(self.left, self.top), wintypes.SIZE(self.w, self.h), wintypes.POINT(0, 0)
            if u32.UpdateLayeredWindow(self.hwnd, screen, ctypes.byref(pos), ctypes.byref(size), self.dc, ctypes.byref(src), 0,
                                       ctypes.byref(self.blend), ULW_ALPHA):
                self.full = False
        finally:
            u32.ReleaseDC(None, screen)

    def pump(self):
        """Dispatch pending window messages (a window whose thread never does that is flagged 'not responding')."""
        msg = wintypes.MSG()
        while u32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 1):
            u32.TranslateMessage(ctypes.byref(msg))
            u32.DispatchMessageW(ctypes.byref(msg))

    @staticmethod
    def vsync():
        """Block until the compositor's next frame, so the loop runs at the monitor's refresh rate and no faster."""
        try:
            ctypes.windll.dwmapi.DwmFlush()
        except OSError:
            pass

    def close(self):
        u32.DestroyWindow(self.hwnd)


class NullOverlay:
    """Same interface as Overlay but no window: draws into a plain array. For headless tests and previews."""

    def __init__(self, rect):
        self.visible = True
        self.move(rect)

    def move(self, rect):
        self.left, self.top, self.w, self.h = rect
        self.canvas = np.zeros((self.h, self.w, 4), np.uint8)
        self.hud = Hud(self.w, self.h)

    def raise_top(self):
        pass

    def toggle(self):
        self.visible = not self.visible

    def draw(self, st, status, toast, status_color="#ffffff"):
        self.hud.render(self.canvas, st, status, hex_bgr(status_color), toast)

    def pump(self):
        pass

    @staticmethod
    def vsync():
        time.sleep(1 / 144)

    def close(self):
        pass
