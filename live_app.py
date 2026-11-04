"""Chest overlay for the whole map (every chest of the fortnite.gg list). Run via run.bat while Fortnite is in borderless / windowed-fullscreen.
A radar on the screen (ahead is up, the nearest chests at their exact bearing and distance, the range zooms in as you reach one).
Pick a route in the window that opens at start-up (or `run.bat --route 3`, `--route "2 players"`, `--route none`): the radar then shows only
that route's chests, with a gold line through the next few.
There are no key binds to learn: every option (radar position and range, full-map chests on/off, refresh rate, route, skip a chest, new match,
quit, calibration) is in the ChestLab settings window, which starts minimised in the taskbar. The only keys it watches are the game's own: M and Esc
for the full map, and Shift to go back to the radar when M was pressed by accident. `--advanced` puts the match numbers on the status line.
The overlay is a normal layered window: Snipping Tool, Print Screen and OBS all capture it."""
import ctypes
import multiprocessing as mp
import os
import sys
import threading
import time
from ctypes import wintypes

try:
    ctypes.windll.shcore.SetProcessDpiAwareness(2)
except Exception:
    pass

import cv2
import keyboard
import mss
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from chestlab.compass import region_px
from chestlab.engine import MssGrabber, Rate
from chestlab.procs import ProcessEngine
from chestlab.live import Pipeline, crop_minimap, load_config, save_config
from chestlab.overlay import Overlay
from chestlab import hud as hud_mod
from chestlab.paths import APP

MSS = getattr(mss, "MSS", None) or mss.mss
USER = os.path.join(APP, "user")
LOG = os.path.join(USER, "live.log")


def game_rect(title="Fortnite"):
    """Client area of the visible game window in screen pixels (any monitor, windowed or borderless), else None.
    Matches by title prefix and picks the largest visible window: a hidden 800x450 SDL helper is also titled exactly
    "Fortnite", while the real UnrealWindow is titled "Fortnite  " (trailing spaces)."""
    u = ctypes.windll.user32
    best = []

    @ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    def visit(h, _):
        n = u.GetWindowTextLengthW(h)
        if n and u.IsWindowVisible(h) and not u.IsIconic(h):
            buf = ctypes.create_unicode_buffer(n + 1)
            u.GetWindowTextW(h, buf, n + 1)
            if buf.value.strip() == title:
                r, pt = wintypes.RECT(), wintypes.POINT(0, 0)
                u.GetClientRect(h, ctypes.byref(r))
                u.ClientToScreen(h, ctypes.byref(pt))
                if r.right >= 640 and r.bottom >= 360:
                    best.append((r.right * r.bottom, dict(left=pt.x, top=pt.y, width=r.right, height=r.bottom)))
        return True

    u.EnumWindows(visit, 0)
    return max(best, key=lambda t: t[0])[1] if best else None


def log(msg):
    os.makedirs(USER, exist_ok=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(time.strftime("%H:%M:%S ") + msg + "\n")


def pick_route(pipe, cfg, args, ask, window=None):
    """Before the overlay starts: which route are we doing? `--route 3` / `--route "2 players"` / `--route none` on the command line, else
    the picker window (a card per route with a map preview; `window` = the function that shows it, default chestlab.picker.pick), else
    a numbered list in this console. Nothing is drawn over the game."""
    want = None
    if "--route" in args and args.index("--route") + 1 < len(args):
        want = args[args.index("--route") + 1]
    elif ask and pipe.routes:
        last = cfg.get("route", "")
        li = int(last) - 1 if str(last).isdigit() else next((i for i, r in enumerate(pipe.routes) if last and last.lower() in r["name"].lower()), None)
        try:
            if window is None:
                from chestlab import picker
                def on_change(key, value):                      # the picker's Settings page: apply and save at once
                    cfg[key] = pipe.cfg[key] = value
                    save_config(cfg)
                window = lambda: picker.pick(picker.describe(pipe), li, cfg, on_change)
            want = window()
        except Exception as e:
            log("route window failed (%s)" % e)
            if sys.stdin is not None and sys.stdin.isatty():
                print("\nWhich route are we doing?\n" + "\n".join(pipe.route_menu()))
                try:
                    want = input("number (Enter = %s): " % (last or "none")).strip() or last
                except EOFError:
                    want = last
            else:
                want = last
    elif ask:
        want = cfg.get("route", "")
    if want is None:
        return
    log(pipe.select_route(want))
    if want != cfg.get("route"):
        cfg["route"] = want if want not in ("0", "none", "off") else ""
        save_config(cfg)


def main(overlay=Overlay, grabber=MssGrabber(), hotkeys=True, rect=None, seconds=None, cfg=None, engine=ProcessEngine, ask_route=False, advanced=None):
    """rect/grabber/overlay/hotkeys are only overridden by the headless tests (no window, no screen, no key hooks)."""
    cfg = cfg or load_config()
    if advanced is None:
        advanced = ask_route and "--advanced" in sys.argv[1:]
    if ask_route:                                   # a real run: stay out of the game's way (the worker processes inherit this)
        try:
            ctypes.windll.winmm.timeBeginPeriod(1)                                                            # 1 ms sleeps, so 60 fps is 60 fps
            ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x4000)       # BELOW_NORMAL
        except Exception:
            pass
    if rect is None:
        with MSS() as s:
            primary = dict(s.monitors[1])
        mon = game_rect() or primary
    else:
        primary = mon = dict(rect)
    pipe = Pipeline(cfg)
    pick_route(pipe, cfg, sys.argv[1:] if ask_route else [], ask_route)
    eng = engine(pipe, mon, log, grabber)
    ov = overlay((mon["left"], mon["top"], mon["width"], mon["height"]))
    toast = {"t": "", "until": 0.0}
    run = {"on": True}

    def say(msg, secs=3.5):
        toast["t"], toast["until"] = msg, time.time() + secs
        log(msg)

    def act(fn):
        def go():
            with eng.lock:
                say(fn())
        return lambda: threading.Thread(target=go, daemon=True).start()

    mapwatch = []
    finders = []
    if ask_route and cfg.get("fullmap", True):      # load the map the full-map overlay matches against now, so M finds it at once
        def _preload():
            try:
                from chestlab.fullmap import MapFinder
                finders.append(MapFinder())
            except Exception as e:
                log("full map: could not load the map (%s)" % e)
        threading.Thread(target=_preload, daemon=True).start()

    def set_hold(on):
        h = getattr(eng, "hold", None)              # the minimap fit pauses while the full map is open (it is not on screen then)
        if hasattr(h, "value"):
            h.value = int(on)
        else:
            eng.hold = bool(on)

    def open_map():
        """M (the game opens its full map): find the map on screen and draw the chests on it until it closes."""
        if not cfg.get("fullmap", True):                 # switched off in the settings: the radar is all there is; M is just the game's
            return
        if not mapwatch:
            from chestlab.fullmap import MapWatch

            def make_grab():
                g = grabber()
                return lambda: g(dict(left=eng.mon["left"], top=eng.mon["top"], width=eng.mon["width"], height=eng.mon["height"]))
            free = lambda: np.where(ov.canvas[::2, ::2, 3] == 0, 255, 0).astype(np.uint8)        # pixels our overlay does not cover
            mapwatch.append(MapWatch(make_grab, (ov.w, ov.h), free, finder=finders[0] if finders else None, log=log))
        mapwatch[0].out_wh = (ov.w, ov.h)

        def dump(fr):
            small = cv2.resize(fr, (1280, int(fr.shape[0] * 1280 / fr.shape[1])), interpolation=cv2.INTER_AREA)
            cv2.imwrite(os.path.join(USER, "map_not_found.jpg"), small, [cv2.IMWRITE_JPEG_QUALITY, 88])
            log("full map: not found yet, saved user/map_not_found.jpg")
        mapwatch[0].dump = dump
        if mapwatch[0].active and not mapwatch[0].stop_ev.is_set():      # M again: the map is closing
            mapwatch[0].stop()
        elif mapwatch[0].start():
            say("looking for the full map...", 2)

    def close_map():
        if mapwatch and mapwatch[0].active:
            mapwatch[0].stop()

    def shift_back_to_radar():
        """Shift while we are only LOOKING for the map (M pressed by accident, or the map is not found): give up and show the radar again. With the
        map found and followed Shift is the game's own Reset and is left alone."""
        w = mapwatch[0] if mapwatch else None
        if w is not None and w.holding and w.cur is None:
            w.stop()
            say("back to the radar", 2)


    if hotkeys:                                                                 # only the game's own keys: everything else is in the settings window
        keyboard.add_hotkey("esc", close_map)                                   # Esc closes the game's map too
        keyboard.add_hotkey("m", open_map)                                      # the game's full map: chests drawn on it
        keyboard.add_hotkey("shift", shift_back_to_radar, suppress=False)       # M by accident: Shift goes back to the radar (the key still reaches the game)

    def set_value(key, value):
        """A setting changed in the settings window: apply it now and save it."""
        with eng.lock:
            cfg[key] = pipe.cfg[key] = value
            if key == "mpp":
                pipe.world.mpp = value
            save_config(cfg)
            if key == "fullmap" and not value and mapwatch and mapwatch[0].active:
                mapwatch[0].stop()
        log("setting %s = %s" % (key, value))
        return "Saved."

    def do(action):
        """A button in the settings window."""
        if action == "quit":
            run.update(on=False)
            return "Quitting..."
        if action == "hide":
            ov.toggle()
            return "Overlay shown." if ov.visible else "Overlay hidden."
        with eng.lock:
            if action.startswith("route:"):
                spec = action[6:]
                msg = pipe.select_route(spec)
                cfg["route"] = spec if spec not in ("0", "none", "off") else ""
                save_config(cfg)
            else:
                msg = {"skip": pipe.skip_route_chest, "new_match": pipe.reset_dug}[action]()
        say(msg)
        return msg

    win = None
    if ask_route and hotkeys:                                                   # a real run: the settings window (starts minimised: never pops up over the game)
        from chestlab import settings
        win = settings.SettingsWindow(cfg, [r["name"] for r in pipe.routes], 0 if pipe.route is None else pipe.route + 1, set_value, do)
        if not win.start():
            log("settings window failed: %r" % (win.error,))
            win = None

    seconds = seconds or float(os.environ.get("CHESTLAB_EXIT_AFTER", 0)) or None
    if seconds:
        threading.Timer(seconds, lambda: run.update(on=False)).start()

    say("Settings: open the ChestLab window from the taskbar" if win else "ChestLab running", 6)
    log("start; overlay visible to screenshots, monitor=%s" % {k: mon[k] for k in ("left", "top", "width", "height")})
    eng.start()

    fps, render_ms, work = Rate(), 0.0, []
    tally = dict(n=0, lock=0, storm=0, head=0, on=0)       # whole-session counts for the summary line in the log
    last_rect = last_state = lost_since = status_t = 0.0
    held_now = False
    last_good = []
    st = None                                                        # the last frame's state (None if the run ended before one)
    status, dot = "", "#ffb020"
    try:
        while run["on"]:
            t0 = time.perf_counter()
            ov.pump()
            if rect is None and t0 - last_rect > 2:                     # follow the game window if it moves / resizes
                last_rect = t0
                new = game_rect() or primary
                if new != eng.mon:
                    eng.mon = new
                    ov.move((new["left"], new["top"], new["width"], new["height"]))
                    log("capture area now %s" % new)
                ov.raise_top()
            st = eng.snapshot(t0, ov.w, ov.h, lead=cfg["lead_ms"] / 1000.0)
            fm = mapwatch[0].state() if mapwatch else None
            map_open = bool(mapwatch and mapwatch[0].holding)
            if map_open != held_now:
                held_now = map_open
                set_hold(map_open)
            if fm is not None and fm["age"] < 1.0:
                st["fullmap"] = dict(M=fm["M"], items=pipe.map_items())
            if st["markers"] and st.get("yaw") is not None:
                last_good[:] = [t0, {k: st[k] for k in ("markers", "yaw", "fov", "radar_m", "radar_up", "radar_at", "markers_3d", "sigma_m") if k in st}]
            elif last_good and t0 - last_good[0] < 1.5 and not map_open:       # a short loss of the fix: keep the radar (dimmed), do not blink
                st.update(last_good[1])
                st["stale"] = True
            tally["n"] += 1
            tally["lock"] += bool(st["locked"])
            tally["storm"] += bool(st["locked"] and st["storm"])
            tally["on"] += bool(st["locked"] and st["on_island"])
            tally["head"] += bool(st["locked"] and st["heading"] is not None and (st["hage"] or 9) <= 1.0)
            if not st["locked"]:
                lost_since = lost_since or t0
            else:
                lost_since = 0.0
            if ov.visible:
                if t0 - status_t >= 0.25 or eng.err:                    # status text changes 4x a second at most (each new text is a new sprite)
                    status_t = t0
                    if eng.err:
                        status, dot = "ERROR " + eng.err[:90], "#ff5a5a"
                    elif map_open:
                        status, dot = ("map: chests on" if st.get("fullmap") else "can't see the whole map: zoom out, or press Shift to go back to the radar" if mapwatch[0].searching_s > 2.5 else "looking for the map..."), "#3dff8a"
                    elif not st["locked"] and t0 - (lost_since or t0) < 1.0:
                        pass                                            # lost for less than a second: leave the status as it was (no flicker)
                    elif not st["locked"]:
                        status, dot = "no minimap lock" + (" (match %.2f, brightness %.0f)" % (st["score"], st["brightness"]) if advanced else ""), "#ff5a5a"
                    elif not st["on_island"]:
                        status, dot = "not on the map", "#ffb020"
                    elif st["heading"] is None or (st["hage"] or 9) > 1.0:
                        status, dot = "waiting for the compass number", "#ffb020"
                    else:
                        tags = (" | rough fix (storm or off the island)" if st["storm"] else "") + (" | floor %+d" % pipe.floor if pipe.floor else "")
                        status = pipe.route_text() if pipe.route is not None else ""
                        if advanced:
                            status = "heading %03.0f | %d fps | map %d Hz | scale %.4f | pitch %.1f | %s%s" % (
                                st["heading"], fps.v, eng.rate_p.v, pipe.cfg["mpp"], pipe.cfg["pitch"], status, tags)
                        elif st["storm"]:
                            status = (status + "  " if status else "") + "rough fix"
                        dot = "#ffb020" if st["storm"] else "#3dff8a"
                ov.draw(st, status, toast["t"] if time.time() < toast["until"] else "", dot)
            render_ms = (time.perf_counter() - t0) * 1000
            work.append(render_ms)
            del work[:-600]
            fps.tick()
            if time.time() - last_state > 5 and eng.pose is not None:
                last_state = time.time()
                near = min((m for m in st["markers"] if not m["dug"] and "hd" in m), key=lambda m: m["hd"], default=None)      # the radar's nearest chest: bearing / metres
                log("state locked=%s island=%s storm=%s pos=%s score=%.2f heading=%s src=%s ocr=%s markers=%d near=%s fps=%.0f render=%.1fms compass=%.0fHz map=%.0fHz" % (
                    st["locked"], st["on_island"], st["storm"], None if st["pos"] is None else tuple(round(v, 1) for v in st["pos"]), st["score"],
                    None if st["heading"] is None else round(st["heading"], 1), eng.heading_src,
                    st["compass"], len(st["markers"]), None if near is None else "#%d@%.0f/%.1fm" % (near["id"], near["brg"], near["hd"]),
                    fps.v, render_ms, eng.rate_c.v, eng.rate_p.v))
            time.sleep(max(1.0 / min(max(float(cfg.get("fps", 60)), 30.0), 144.0) - (time.perf_counter() - t0), 0.001))       # steady fps, whatever the monitor's refresh rate
    finally:
        if tally["n"]:
            n, w = tally["n"], np.asarray(work or [0.0])
            log("summary frames=%d locked=%.1f%% on_island=%.1f%% storm=%.1f%% heading_ok=%.1f%% fps=%.0f render_ms p50=%.1f p95=%.1f p99=%.1f" % (
                n, 100.0 * tally["lock"] / n, 100.0 * tally["on"] / n, 100.0 * tally["storm"] / n, 100.0 * tally["head"] / n, fps.v,
                *np.percentile(w, (50, 95, 99))))
        if mapwatch:
            mapwatch[0].stop()
        eng.stop = True
        if hasattr(eng, "close"):
            eng.close()
        if hotkeys:
            keyboard.unhook_all()
        if win is not None:
            win.close()
        ov.close()
    return dict(frames=fps.n, fps=fps.v, render_ms=render_ms, work=work, state=st, ov=ov, eng=eng, set_value=set_value, do=do)


def _child_ping(q):
    q.put(os.getpid())


def selftest():
    """`--selftest`: load everything the app needs (assets, fonts, the Windows OCR, a worker process) and draw one radar frame, without opening any window
    or touching the game. Used to check a build. Exit code 0 = fine."""
    import json
    from chestlab.compass import CompassReader
    from chestlab.hud import Hud, hex_bgr
    from chestlab.worldmap import world_map
    ok = True

    def step(name, fn):
        nonlocal ok
        try:
            print("%-28s %s" % (name, fn()))
        except Exception as e:
            ok = False
            print("%-28s FAILED %r" % (name, e))

    step("assets (base)", lambda: __import__("chestlab.paths", fromlist=["BASE"]).BASE)
    step("config + pipeline", lambda: "%d routes" % len(Pipeline(load_config()).routes))
    step("world map + chests", lambda: "%d chests" % len(world_map().chests))
    step("windows OCR", lambda: CompassReader().read(np.full((36, 460, 3), 40, np.uint8)) is None and "ok (blank -> nothing)")

    def draw():
        hud, cv = Hud(1280, 720), np.zeros((720, 1280, 4), np.uint8)
        hud.render(cv, dict(markers=[], yaw=None), "selftest", hex_bgr("#3dff8a"), "", t=0.1)
        return "ok"
    step("fonts + hud", draw)

    def settings_ui():
        import tkinter as tk
        from chestlab.settings import SettingsPanel
        r = tk.Tk()
        r.withdraw()                                   # built but never shown
        try:
            SettingsPanel(r, load_config(), ["a"], 0, lambda k, v: "", lambda a: "")
        finally:
            r.destroy()
        return "ok (not shown)"
    step("settings window", settings_ui)

    def child():
        ctx = mp.get_context("spawn")
        q = ctx.Queue()
        p = ctx.Process(target=_child_ping, args=(q,))
        p.start()
        pid = q.get(timeout=60)
        p.join(10)
        return "worker process %d" % pid
    step("worker process", child)
    print("SELFTEST", "OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    mp.freeze_support()
    if "--selftest" in sys.argv[1:]:
        sys.exit(selftest())                             # the built .exe starts its sensor processes by running itself again
    main(ask_route=True)
