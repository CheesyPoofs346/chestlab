"""Chest overlay for the whole map (every chest of the fortnite.gg list). Run via run.bat while Fortnite is in borderless / windowed-fullscreen.
A radar on the right of the screen (F1 moves it: left / corners) (ahead is up, the nearest chests at their exact bearing and distance, the range zooms in as you reach one).
Pick a route in the window that opens at start-up (or `run.bat --route 3`, `--route "2 players"`, `--route none`): the radar then shows only
that route's chests, with a gold line through the next few.
Keys: M map (chests drawn on the game's full map)   F12 next route / off   F2 skip a chest that is not there   F5 new match   F6 hide / show   F1 radar position   Alt+M full-map chests on/off   F10 quit.
`--advanced` adds the calibration, 3D-marker and spawn-learning keys (F3/F4 FOV, Alt+arrows tilt / turn / lead, Ctrl+PgUp/PgDn floor, Ctrl+Left/Right
reach, Ctrl+Up/Down distances, Ctrl+M 3D markers, Ctrl+R radar range, Ctrl+N north up, F7/F8 dug a chest / undo, F9 debug screenshot, F11 read the
spawns off the open map) and the numbers on the status line.
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
MAX_FPS = 60                    # the overlay never needs more: a higher cap only costs the game frames


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


def save_debug(mon, grabber=MssGrabber()):
    """What the capture sees right now: the screen, the minimap crop and the compass strip, into user/."""
    frame = cv2.cvtColor(grabber()(dict(left=mon["left"], top=mon["top"], width=mon["width"], height=mon["height"])), cv2.COLOR_BGRA2BGR)
    x0, y0, w, h = region_px(frame.shape[1])
    cv2.imwrite(os.path.join(USER, "debug_frame.png"), frame)
    cv2.imwrite(os.path.join(USER, "debug_minimap.png"), crop_minimap(frame))
    cv2.imwrite(os.path.join(USER, "debug_compass.png"), frame[y0:y0 + h, x0:x0 + w])
    log("saved user/debug_frame.png, debug_minimap.png, debug_compass.png")


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
                window = lambda: picker.pick(picker.describe(pipe), li)
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

    def nudge(key, delta, lo, hi, fmt):
        def f():
            cfg[key] = round(min(hi, max(lo, cfg[key] + delta)), 4)
            pipe.cfg[key] = cfg[key]
            save_config(cfg)
            return fmt % cfg[key]
        return f

    def scale(factor):
        def f():
            cfg["mpp"] = round(min(0.6, max(0.03, cfg["mpp"] * factor)), 4)
            pipe.cfg["mpp"] = pipe.world.mpp = cfg["mpp"]
            save_config(cfg)
            return "distance scale %.4f m per map px" % cfg["mpp"]
        return f

    def toggle_3d():
        cfg["markers_3d"] = pipe.cfg["markers_3d"] = not cfg.get("markers_3d", False)
        save_config(cfg)
        return "3D markers on" if cfg["markers_3d"] else "radar only"

    def cycle_radar():
        steps = (0.0, 20.0, 40.0, 80.0)
        cfg["radar_m"] = pipe.cfg["radar_m"] = steps[(steps.index(cfg["radar_m"]) + 1) % 4] if cfg.get("radar_m") in steps else 0.0
        save_config(cfg)
        return "radar range: auto (zooms in as you reach the chest)" if not cfg["radar_m"] else "radar range %.0f m" % cfg["radar_m"]

    def toggle_up():
        cfg["radar_up"] = pipe.cfg["radar_up"] = not cfg.get("radar_up", True)
        save_config(cfg)
        return "radar: ahead is up" if cfg["radar_up"] else "radar: north is up"

    def toggle_fullmap():
        cfg["fullmap"] = pipe.cfg["fullmap"] = not cfg.get("fullmap", True)
        save_config(cfg)
        if not cfg["fullmap"] and mapwatch and mapwatch[0].active:
            mapwatch[0].stop()
        return "chests on the full map: on" if cfg["fullmap"] else "chests on the full map: off (radar only)"

    def cycle_radar_spot():
        spots = list(hud_mod.RADAR_SPOTS)
        cur = cfg.get("radar_at")
        nxt = spots[(spots.index(cur) + 1) % len(spots)] if cur in spots else spots[0]
        cfg["radar_at"] = pipe.cfg["radar_at"] = nxt
        save_config(cfg)
        return "radar position: " + nxt

    scanner = []
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
        if not cfg.get("fullmap", True):                 # switched off (Alt+M): the radar is all there is; M is just the game's
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


    def scan_map():
        """F11 with the full map open: register the frame to the island map and read which spawns carry a chest icon."""
        say("reading the map...", 2)
        try:
            if not scanner:
                from chestlab.mapscan import MapScanner
                scanner.append(MapScanner())
            m = eng.mon
            frame = cv2.cvtColor(grabber()(dict(left=m["left"], top=m["top"], width=m["width"], height=m["height"])), cv2.COLOR_BGRA2BGR)
            res = scanner[0].scan(frame)
            with eng.lock:
                say(pipe.apply_scan(res), 9)
        except Exception as e:
            say("map scan failed: %s" % e, 6)

    if hotkeys:
        keyboard.add_hotkey("esc", close_map)                                   # Esc closes the game's map too
        keyboard.add_hotkey("m", open_map)                                      # the game's full map: chests drawn on it
        keyboard.add_hotkey("shift", shift_back_to_radar, suppress=False)       # M by accident: Shift goes back to the radar (the key still reaches the game)
        keyboard.add_hotkey("f12", act(pipe.cycle_route))                       # next route / off
        keyboard.add_hotkey("f2", act(pipe.skip_route_chest))                   # the next route chest is not there
        keyboard.add_hotkey("f5", act(pipe.reset_dug))                          # new match
        keyboard.add_hotkey("f6", ov.toggle)                                    # hide / show
        keyboard.add_hotkey("alt+m", act(toggle_fullmap))                       # chests on the game's full map: on / off
        keyboard.add_hotkey("f1", act(cycle_radar_spot))                        # radar position: default / each corner
        keyboard.add_hotkey("f10", lambda: run.update(on=False))                # quit
        if advanced:                                                            # calibration, 3D markers, learning spawns: not for normal use
            keyboard.add_hotkey("f7", act(pipe.dig))
            keyboard.add_hotkey("f8", act(pipe.undo_dig))
            keyboard.add_hotkey("f3", act(nudge("fov", -1.0, 50, 120, "FOV %.0f")))
            keyboard.add_hotkey("f4", act(nudge("fov", +1.0, 50, 120, "FOV %.0f")))
            keyboard.add_hotkey("alt+left", act(nudge("yaw_off", -0.5, -30, 30, "turn offset %.1f deg")))
            keyboard.add_hotkey("alt+right", act(nudge("yaw_off", +0.5, -30, 30, "turn offset %.1f deg")))
            keyboard.add_hotkey("alt+up", act(nudge("pitch", -0.5, -40, 25, "camera pitch %.1f deg (markers moved up)")))
            keyboard.add_hotkey("alt+down", act(nudge("pitch", +0.5, -40, 25, "camera pitch %.1f deg (markers moved down)")))
            keyboard.add_hotkey("alt+page up", act(nudge("lead_ms", +5.0, 0, 150, "turn lead %.0f ms")))
            keyboard.add_hotkey("alt+page down", act(nudge("lead_ms", -5.0, 0, 150, "turn lead %.0f ms")))
            keyboard.add_hotkey("ctrl+page up", act(lambda: pipe.change_floor(+1)))
            keyboard.add_hotkey("ctrl+page down", act(lambda: pipe.change_floor(-1)))
            keyboard.add_hotkey("ctrl+right", act(nudge("show_m", +10.0, 20, 500, "marking chests within %.0f m")))
            keyboard.add_hotkey("ctrl+left", act(nudge("show_m", -10.0, 20, 500, "marking chests within %.0f m")))
            keyboard.add_hotkey("ctrl+m", act(toggle_3d))
            keyboard.add_hotkey("ctrl+r", act(cycle_radar))
            keyboard.add_hotkey("ctrl+n", act(toggle_up))
            keyboard.add_hotkey("ctrl+up", act(scale(1.03)))
            keyboard.add_hotkey("ctrl+down", act(scale(1 / 1.03)))
            keyboard.add_hotkey("f9", lambda: threading.Thread(target=lambda: (save_debug(eng.mon, grabber), say("saved debug screenshot to the user folder")), daemon=True).start())
            keyboard.add_hotkey("f11", lambda: threading.Thread(target=scan_map, daemon=True).start())

    seconds = seconds or float(os.environ.get("CHESTLAB_EXIT_AFTER", 0)) or None
    if seconds:
        threading.Timer(seconds, lambda: run.update(on=False)).start()

    say("M map  ·  F12 route  ·  F2 skip chest  ·  F5 new match  ·  F6 hide  ·  F1 radar spot  ·  Alt+M map on/off  ·  F10 quit", 8)
    log("start; overlay visible to screenshots, monitor=%s" % {k: mon[k] for k in ("left", "top", "width", "height")})
    eng.start()

    fps, render_ms, work = Rate(), 0.0, []
    tally = dict(n=0, lock=0, storm=0, head=0, on=0)       # whole-session counts for the summary line in the log
    last_rect = last_state = lost_since = last_dump = status_t = 0.0
    held_now = False
    last_good = []
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
            if lost_since and t0 - lost_since > 4 and t0 - last_dump > 30:
                last_dump = t0
                threading.Thread(target=save_debug, args=(eng.mon, grabber), daemon=True).start()
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
            time.sleep(max(1.0 / MAX_FPS - (time.perf_counter() - t0), 0.001))       # steady MAX_FPS, whatever the monitor's refresh rate
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
        ov.close()
    return dict(frames=fps.n, fps=fps.v, render_ms=render_ms, work=work, state=st, ov=ov, eng=eng)


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
