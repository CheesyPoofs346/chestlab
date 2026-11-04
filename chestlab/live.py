"""Live pipeline: one game frame in -> player pose and chest markers (screen pixels) out. Pure function of the
frame plus a little state, so it runs the same on a screenshot as on the live capture."""
import collections
import json
import os
import time

import cv2
import numpy as np

from .calib import ChestRecorder
from .compass import CompassReader
from .estimate import Tracker
from .geometry import Camera, wrap180
from .world import World
from . import worldmap

from .paths import APP, BASE as ROOT
USER = os.path.join(APP, "user")
CONFIG, CHESTS = os.path.join(USER, "config.json"), os.path.join(USER, "chests.json")
LEARNED = os.path.join(USER, "learned.json")   # spawns you dug that the built-in map doesn't have
SPAWNS = os.path.join(ROOT, "assets", "spawns.json")   # every buried-chest spawn on the island (tools/build_spawns.py)
HEIGHTS = os.path.join(USER, "heights.json")   # per chest (built-in ones, then learned ones): metres above the island's ground level
SURE_KINDS = ('chest_sure', 'rare', 'sprite')   # world chests known to exist (100% spawn rate, or special); the others may not
ROUTE_LOOKAHEAD = 3                            # route mode marks this many chests ahead (the next one and the two after)
ROUTE_REACH_M = 2.5                            # a route chest counts as opened when you are this close
MERGE_PX = 30.0                                # a new dig within this many map px (~2.5 m) of a known spawn is the same spawn
MINIMAP_BOX = (1786, 22, 1980, 214)   # in a 2000-px-wide frame; scaled by frame width

DEFAULTS = dict(mpp=0.0875, fov=78.0, yaw_off=0.0, v_off=0.0, pitch=-3.5, chest_z=0.4, dig_radius_m=15.0,
                ocr_every_s=0.12, ring_m=1.2, pillar_m=8.0, lead_ms=25.0, floor_m=3.0, show_m=70.0, maybe_m=30.0,
                markers_3d=False, radar_m=0.0, radar_up=True, radar_at="default", fullmap=True, route="")      # radar only by default; Ctrl+M brings the 3D markers back


def load_config():
    try:
        with open(CONFIG, encoding="utf-8") as f:
            return {**DEFAULTS, **json.load(f)}
    except (OSError, ValueError):
        return dict(DEFAULTS)


def save_config(cfg):
    os.makedirs(USER, exist_ok=True)
    with open(CONFIG, "w", encoding="utf-8") as f:
        json.dump({k: cfg[k] for k in DEFAULTS}, f, indent=1)


def crop_minimap(frame, size=231, s=None, ox=0, oy=0):
    """The minimap, resized to size x size. frame: the whole frame, or a capture of just the minimap box whose
    top-left sits at screen pixel (ox, oy) with scale s (screen px per 2000-scale px)."""
    s = frame.shape[1] / 2000.0 if s is None else s
    x0, y0, x1, y1 = (int(round(v * s)) for v in MINIMAP_BOX)
    c = frame[y0 - oy:y1 - oy, x0 - ox:x1 - ox]
    return cv2.resize(c, (size, size), interpolation=cv2.INTER_AREA if c.shape[1] > size else cv2.INTER_CUBIC)


class Pipeline:
    def __init__(self, cfg=None, reader=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.world = World(mpp=self.cfg["mpp"])
        self.on_map = np.ones(len(self.world.chests), bool)
        try:                                         # all 19 spawns; only some exist in any one match
            with open(SPAWNS, encoding="utf-8") as f:
                sp = json.load(f)
            self.world.chests = np.array([[c["map_x"], c["map_y"]] for c in sp], float)
            self.on_map = np.array([bool(c.get("on_map", True)) for c in sp])
        except (OSError, ValueError, KeyError):
            pass
        self.n_island = len(self.world.chests)       # the island's own spawns come first (F11 map scan, your saved corrections)
        self.kind = np.array(["buried"] * self.n_island)
        self.use_world = bool(self.cfg.get("world", True)) and worldmap.available()
        if self.use_world:                           # every chest on the whole map (fortnite.gg); the island's are already in
            wm = worldmap.world_map()
            fresh = np.hypot(*(wm.chests[:, None, :] - self.world.chests[None]).transpose(2, 0, 1)).min(axis=1) > MERGE_PX
            self.world.chests = np.vstack([self.world.chests, wm.chests[fresh]])
            self.kind = np.append(self.kind, wm.kind[fresh])
            self.on_map = np.append(self.on_map, np.isin(wm.kind[fresh], SURE_KINDS))
        if os.path.exists(CHESTS):
            with open(CHESTS, encoding="utf-8") as f:
                saved = np.array(json.load(f), float)
            if saved.shape == self.world.chests.shape:
                self.world.chests = saved
            elif saved.shape == (self.n_island, 2):  # saved before the whole map existed
                self.world.chests[:self.n_island] = saved
        self.n_base = len(self.world.chests)
        try:
            with open(LEARNED, encoding="utf-8") as f:
                learned = np.array(json.load(f), float).reshape(-1, 2)
        except (OSError, ValueError):
            learned = np.zeros((0, 2))
        self.world.chests = np.vstack([self.world.chests, learned])
        self.on_map = np.append(self.on_map, np.ones(len(learned), bool))
        self.kind = np.append(self.kind, ["buried"] * len(learned))
        self.active = self.on_map.copy()             # spawns that exist in this match: your map's icons, until the game map says otherwise
        self.chest_h = self._load_heights(len(self.world.chests))
        self.floor = 0                               # which storey you are on (you set it: no game HUD shows altitude)
        self.rec = ChestRecorder(self.world.chests)
        self.dug = np.zeros(len(self.world.chests), bool)
        self.tracker = worldmap.WorldTracker(self.world) if self.use_world else Tracker(self.world)
        self.routes = self._load_routes()            # best chest-farming routes (tools/make_routes.py); F12 picks one
        self.route = None                            # index into self.routes, None = no route: every chest is marked
        self.route_i = 0                             # next chest of the route you still have to open
        self.reader = reader or CompassReader()
        self.compass, self.compass_t = None, -1e9
        self.recent = collections.deque(maxlen=30)   # last raw position fixes, for recording a chest
        self.live_pos = None                         # where the live engine says the player is right now (map px)
        self.state = dict(locked=False, on_island=False, pos=None, heading=None, markers=[])

    def _load_routes(self):
        """Every route we have, as [dict(name, idx=[chest index...], seconds, per_min, group)], in menu order: solo quiet routes
        (assets/fast_routes.json), two-player routes (pair_routes.json: one entry per player), quiet routes (lowkey_routes.json), then the
        100% / all-chests loops (routes_sure.json, routes_all.json). Chests are found by position; a route with a chest this map
        no longer has is left out."""
        def read(name):
            try:
                with open(os.path.join(ROOT, "assets", name + ".json"), encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                return []

        def route(name, group, uv, seconds, per_min):
            uv = np.asarray(uv, float)
            d = np.hypot(*(self.world.chests[None, :, :] - uv[:, None, :]).transpose(2, 0, 1))
            idx = d.argmin(axis=1)
            if (d.min(axis=1) < 40.0).all() and len(set(idx.tolist())) == len(idx):
                return dict(name=name, group=group, idx=[int(i) for i in idx], seconds=float(seconds), per_min=float(per_min))

        to_uv = lambda path: [np.asarray(q["unit"], float) * worldmap.UV_PER_UNIT + worldmap.ORIGIN for q in path]
        out = []
        for r in read("fast_routes"):
            out.append(route("Solo %d s, quiet" % r["budget_s"], "solo", to_uv(r["path"]), r["seconds"], r["chests_per_min_with_reset"]))
        for k, r in enumerate(read("pair_routes")):
            for n, pl in enumerate(r["players"]):
                out.append(route("2 players %d s #%d: player %d" % (r["budget_s"], k + 1, n + 1), "pair", to_uv(pl["path"]), pl["seconds"], r["chests_per_min_with_reset"] / 2))
        for r in read("lowkey_routes"):
            out.append(route("Quiet #{n} near %s" % r["nearest_place"], "quiet", to_uv(r["path"]), r["seconds"], r["chests_per_min"]))
        for mode, label in (("sure", "100% chests"), ("all", "All chests")):
            for k, r in enumerate(read("routes_" + mode)):
                out.append(route(label + " #{n}", "loop", r["path_uv"], r["seconds"], r["chests_per_min"]))
        for r in read("saved_routes"):                          # routes you brought yourself (assets/saved_routes.json): last, so the numbers above never move
            out.append(route(r["name"], "saved", to_uv(r["path"]), r["seconds"], r["chests_per_min_with_reset"]))
        out, count = [r for r in out if r is not None], {}
        for r in out:                                                                    # number the loops 1, 2, 3 ... among those that are left
            if "{n}" in r["name"]:
                key = r["name"].split("{n}")[0]
                count[key] = count.get(key, 0) + 1
                r["name"] = r["name"].replace("{n}", str(count[key]))
        return out

    def route_menu(self):
        """The numbered list the start-up picker prints: 0 = no route."""
        return ["  0  no route: mark every chest"] + ["%3d  %-34s %2d chests, %3.0f s, %.1f chests/min" % (i + 1, r["name"], len(r["idx"]), r["seconds"], r["per_min"]) for i, r in enumerate(self.routes)]

    def select_route(self, spec):
        """Pick a route by its menu number (0 / None = off) or by (part of) its name. -> a one-line message."""
        if spec in (None, "", 0, "0", "none", "off"):
            self.route = None
            return "route off: every chest is marked"
        if isinstance(spec, str) and not spec.isdigit():
            hits = [i for i, r in enumerate(self.routes) if spec.lower() in r["name"].lower()]
            if not hits:
                return "no route called %r" % spec
            i = hits[0]
        else:
            i = int(spec) - 1
            if not 0 <= i < len(self.routes):
                return "no route number %s" % spec
        self.route, self.route_i = i, 0
        r = self.routes[i]
        self.dug[r["idx"]] = False
        return "route %s: %d chests, about %.0f s a run (%.1f chests/min). Go to chest 1." % (r["name"], len(r["idx"]), r["seconds"], r["per_min"])

    def skip_route_chest(self):
        """F2: leave the next route chest (it is not there / you cannot reach it) and go on to the one after."""
        if self.route is None:
            return "no route is on"
        idx = self.routes[self.route]["idx"]
        if self.route_i >= len(idx):
            return "the route is finished"
        self.dug[idx[self.route_i]] = True
        self.route_i += 1
        return "skipped: on to chest %d of %d" % (min(self.route_i + 1, len(idx)), len(idx))

    @staticmethod
    def _load_heights(n):
        try:
            with open(HEIGHTS, encoding="utf-8") as f:
                h = np.asarray(json.load(f), float).ravel()
        except (OSError, ValueError):
            h = np.zeros(0)
        out = np.zeros(n)
        out[:min(n, len(h))] = h[:n]
        return out

    @property
    def player_z(self):
        """Your height above the island's ground level, metres."""
        return self.floor * self.cfg["floor_m"]

    # ---- one frame
    def process(self, frame, dt=1 / 15, force_ocr=False, full_size=None):
        """frame: the whole game frame, or just its top band (minimap + compass are all this needs) with
        full_size=(W, H) giving the real frame size for the 3D projection."""
        H, W = frame.shape[:2]
        if full_size is not None:
            W, H = full_size
        now = time.perf_counter()
        if force_ocr or now - self.compass_t >= self.cfg["ocr_every_s"]:
            c = self.reader.read(frame)
            if c is not None:
                self.compass = c
            self.compass_t = now
        r = self.step_minimap(crop_minimap(frame), dt, heading=self.compass)   # crop scales by frame width
        st = dict(locked=bool(r["locked"]), on_island=bool(r["on_island"]), pos=r["pos"], heading=r["heading"],
                  compass=self.compass, markers=[], score=self.tracker.fitter.last_score, zoom=self.tracker.fitter.zoom, brightness=float(frame[::16, ::16].mean()))
        if st["locked"] and st["on_island"] and r["heading"] is not None:
            st["markers"] = self.project(W, H, r["pos"], r["heading"])
            st.update(self.view(r["heading"], r["pos"], bool(r["raw"] is not None and r["fit"].adaptive)))
        self.state = st
        return st

    def step_minimap(self, minimap, dt, heading=None):
        """One minimap fit -> tracker result. Clear-weather raw fixes are kept for recording chests (storm-mode
        fixes are ~2 m rougher, so they are not)."""
        r = self.tracker.step(minimap, dt, heading=heading)
        if r["raw"] is not None and not r["fit"].adaptive:
            self.recent.append(r["raw"])
        return r

    SIGMA_ISLAND_M, SIGMA_WORLD_M, SIGMA_STORM_M = 0.3, 0.7, 1.2      # measured on real shots: world matcher sd 0.4 / 0.56 m (rms 0.68) vs the island fit; storm is rougher

    def accuracy_m(self, pos, storm=False):
        """How far off the position fix can be, roughly (1 sigma, metres): the radar draws it as a disc around you. The island
        has the precise silhouette fit; everywhere else the minimap is matched to the fortnite.gg tiles."""
        if pos is None:
            return self.SIGMA_WORLD_M
        w, h = self.world.w, self.world.h
        dx, dy = max(-pos[0], 0.0, pos[0] - w), max(-pos[1], 0.0, pos[1] - h)
        near_island = float(np.hypot(dx, dy)) <= 600.0
        return self.SIGMA_STORM_M if storm else self.SIGMA_ISLAND_M if near_island else self.SIGMA_WORLD_M

    def view(self, heading, pos=None, storm=False):
        """Which way the camera looks and how wide: what the radar's arrow and wedge show (the markers use the same numbers), and how
        precise the fix is."""
        return dict(sigma_m=self.accuracy_m(pos, storm), yaw=heading + self.cfg["yaw_off"], fov=self.cfg["fov"], markers_3d=bool(self.cfg["markers_3d"]), radar_m=float(self.cfg["radar_m"]), radar_up=bool(self.cfg["radar_up"]), radar_at=self.cfg["radar_at"])

    def map_items(self):
        """What the full-map overlay draws, in fortnite.gg's z4 map pixels (the full map's own coordinates): per chest its position, kind,
        whether it is a sure spawn, whether you dug it, and (route mode) its step number; plus the next route chest."""
        n = len(self.world.chests)
        pad = lambda a, fill: np.append(a, [fill] * max(0, n - len(a)))[:n]
        step = np.zeros(n, int)
        nxt = -1
        if self.route is not None:
            ridx = self.routes[self.route]["idx"]
            step[ridx] = np.arange(1, len(ridx) + 1)
            nxt = ridx[self.route_i] if self.route_i < len(ridx) else -1
        return dict(xy=worldmap.uv_to_px(self.world.chests, 4), kind=pad(self.kind, "buried"), sure=pad(self.active, True).astype(bool),
                    dug=pad(self.dug, False).astype(bool), step=step, next=nxt)

    def project(self, W, H, pos, heading):
        """Per chest: ground point, a ring on the ground (perspective-correct), the top of a beacon pillar, distance,
        and how far above (+) or below (-) you it is (dz, metres; `dist` is the straight-line distance incl. that).
        `show` is False for chests farther than cfg show_m: only the close ones are marked. `sure` is False for a spawn that
        is not known to exist in this match (it is only marked within maybe_m)."""
        c = self.cfg
        cam = Camera(W, H, c["fov"])
        yaw = heading + c["yaw_off"]
        zp = self.player_z
        pos = np.asarray(pos, float)
        d2 = np.hypot(*(self.world.chests - pos).T) * self.world.mpp
        reach = max(c["show_m"], c["maybe_m"]) * 1.3 + abs(zp) + 10.0
        if not c.get("markers_3d", True):
            reach = max(reach, 125.0)                                                    # radar-only: the radar's longest range is 80 m, chests out to 1.5x that sit on its rim
        idx = np.nonzero(d2 <= reach)[0]                                                 # the whole map has ~1000 chests: only the near ones
        step = {}
        nxt, upcoming = -1, ()
        if self.route is not None:                       # route mode: only the route's chests, numbered; the next one is always there
            ridx = self.routes[self.route]["idx"]
            self._route_progress(pos)
            step = {i: k + 1 for k, i in enumerate(ridx)}
            nxt = ridx[self.route_i] if self.route_i < len(ridx) else -1
            upcoming = range(self.route_i + 1, self.route_i + 1 + ROUTE_LOOKAHEAD)       # only the next few, numbered: a tight cluster would be a wall of markers
            idx = np.union1d(idx[np.isin(idx, ridx)], [nxt] if nxt >= 0 else []).astype(int)
        if not len(idx):
            return []
        chests = self.world.chests[idx]
        P = self.world.uv_to_enu(pos, zp)[0]
        eye = cam.eye(P, yaw, c["pitch"])
        base = self.world.uv_to_enu(chests, 0.0)
        n = len(base)
        h = np.zeros(len(self.world.chests))             # heights, tolerant of a chest list that is being extended right now
        k = min(len(h), len(self.chest_h))
        h[:k] = self.chest_h[:k]
        h = h[idx]
        base[:, 2] = h + c["chest_z"]
        dz = h - zp
        ang = np.linspace(0, 2 * np.pi, 13)[:-1]
        ring = np.concatenate([base + [c["ring_m"] * np.cos(a), c["ring_m"] * np.sin(a), 0] for a in ang])   # (12n, 3)
        d = np.hypot(d2[idx], dz)
        top = base + np.column_stack([np.zeros(n), np.zeros(n), np.clip(0.25 * d, 1.5, c["pillar_m"])])   # short when near
        pts = np.concatenate([base, top, ring])
        xy, zc, ok = cam.project(pts, eye, yaw, c["pitch"])
        xy[:, 1] += c["v_off"] * H / 1125.0
        act = np.ones(len(self.world.chests), bool)
        k2 = min(len(act), len(self.active))
        act[:k2] = self.active[:k2]
        act = act[idx]
        dug = np.zeros(len(self.world.chests), bool)
        k3 = min(len(dug), len(self.dug))
        dug[:k3] = self.dug[:k3]
        dug = dug[idx]
        kind = np.append(self.kind, ["buried"] * max(0, len(self.world.chests) - len(self.kind)))[idx]
        brg = np.degrees(np.arctan2(chests[:, 0] - pos[0], -(chests[:, 1] - pos[1]))) % 360
        rel = wrap180(brg - yaw)
        ring_xy = xy[2 * n:].reshape(len(ang), n, 2).transpose(1, 0, 2)
        ring_ok = ok[2 * n:].reshape(len(ang), n).T.all(axis=1)
        return [dict(id=int(idx[i]) + 1, x=float(xy[i, 0]), y=float(xy[i, 1]), top=(float(xy[n + i, 0]), float(xy[n + i, 1])),
                     ring=ring_xy[i].tolist() if ring_ok[i] else None, dist=float(d[i]), dz=float(dz[i]), rel=float(rel[i]),
                     brg=float(brg[i]), hd=float(d2[idx[i]]),           # compass bearing and flat distance (m): what the radar plots
                     front=bool(ok[i]), dug=bool(dug[i]), sure=bool(act[i]) or int(idx[i]) in step, kind=str(kind[i]),
                     show=(bool(d[i] <= (c["show_m"] if act[i] else c["maybe_m"])) if not step else step.get(int(idx[i]), 0) in upcoming),
                     step=step.get(int(idx[i]), 0), next=int(idx[i]) == nxt) for i in range(n)]

    def cycle_route(self):
        """F12: off -> route 1 -> route 2 ... -> off. A route marks only its own chests, numbered in the order to open them."""
        if not self.routes:
            return "no routes built yet: run tools/make_routes.py"
        self.route = 0 if self.route is None else (self.route + 1 if self.route + 1 < len(self.routes) else None)
        self.route_i = 0
        if self.route is None:
            return "route off: every chest is marked"
        r = self.routes[self.route]
        self.dug[r["idx"]] = False
        return "route %s: %d chests, about %.0f s a run (%.1f chests/min). Go to chest 1." % (r["name"], len(r["idx"]), r["seconds"], r["per_min"])

    def route_text(self):
        if self.route is None:
            return ""
        n = len(self.routes[self.route]["idx"])
        return "route %s %d/%d" % (self.routes[self.route]["name"], min(self.route_i, n), n) + (" done: leave and rejoin" if self.route_i >= n else "")

    def _route_progress(self, pos):
        """You reached the next chest of the route (within 2.5 m, or you marked it dug) -> on to the following one."""
        idx = self.routes[self.route]["idx"]
        while self.route_i < len(idx):
            i = idx[self.route_i]
            if not self.dug[i] and np.hypot(*(self.world.chests[i] - np.asarray(pos))) * self.world.mpp > ROUTE_REACH_M:
                break
            self.dug[i] = True
            self.route_i += 1

    def change_floor(self, delta):
        """Ctrl+PgUp/PgDn: you went up / down a storey. Moves every marker's height reference."""
        self.floor = int(min(8, max(-3, self.floor + delta)))
        return "you are on floor %+d (%.0f m %s ground)" % (self.floor, abs(self.player_z), "above" if self.floor > 0 else "below" if self.floor < 0 else "at") if self.floor else "you are on the ground floor"

    # ---- actions
    def _pos(self):
        return self.live_pos if self.live_pos is not None else self.tracker.filt.pos

    def nearest(self, max_m=None):
        pos = self._pos()
        if pos is None:
            return None, None
        d = np.hypot(*(self.world.chests - np.asarray(pos)).T) * self.world.mpp
        d[self.dug] = np.inf
        i = int(np.argmin(d))
        return (i, float(d[i])) if np.isfinite(d[i]) and (max_m is None or d[i] <= max_m) else (None, None)

    def dig(self):
        """F7: you're standing on a chest you just dug up. Hide it, and move it to where you really are."""
        i, d = self.nearest(self.cfg["dig_radius_m"])
        fixed = len(self.recent) >= 5
        if i is None:
            if not fixed:
                return "no chest within %.0f m (and storm: can't place a new one)" % self.cfg["dig_radius_m"]
            return self._learn_spawn()
        if fixed:
            self.rec.press(i, list(self.recent)[-15:])
            self.world.chests = self.rec.chests
        self.chest_h[i] = self.player_z              # you are standing on it: its height is yours
        self.active[i] = True
        self._save_chests()
        self.dug[i] = True
        return "chest #%d dug (was %.1f m away)%s%s" % (i + 1, d, "; position corrected" if fixed else "; storm: position left as is",
                                                        "; height %+.0f m" % self.player_z if self.player_z else "")

    def _save_chests(self):
        os.makedirs(USER, exist_ok=True)
        with open(CHESTS, "w", encoding="utf-8") as f:
            json.dump(self.rec.chests[:self.n_base].tolist(), f)
        with open(LEARNED, "w", encoding="utf-8") as f:
            json.dump(self.rec.chests[self.n_base:].tolist(), f)
        with open(HEIGHTS, "w", encoding="utf-8") as f:
            json.dump([round(float(v), 2) for v in self.chest_h], f)

    def _learn_spawn(self):
        """F7 with no known chest nearby: this island has a spawn the map lacks. Remember it for later matches."""
        est = np.median(np.asarray(list(self.recent)[-15:], float), axis=0)
        near = np.hypot(*(self.rec.chests - est).T) < MERGE_PX
        if near.any():                        # already known (and dug earlier this match): just average it in
            j = int(np.argmin(np.hypot(*(self.rec.chests - est).T)))
            self.rec.press(j, list(self.recent)[-15:])
            msg = "chest #%d re-recorded" % (j + 1)
        else:
            self.rec.chests = np.vstack([self.rec.chests, est])
            self.rec.n = np.append(self.rec.n, 1)
            self.dug = np.append(self.dug, False)
            self.chest_h = np.append(self.chest_h, self.player_z)
            self.on_map = np.append(self.on_map, True)
            self.active = np.append(self.active, True)
            self.kind = np.append(self.kind, "buried")
            j = len(self.rec.chests) - 1
            msg = "NEW spawn learned as #%d (%d known now)" % (j + 1, j + 1)
        msg += "; height %+.0f m" % self.player_z if self.player_z else ""
        self.world.chests = self.rec.chests
        self.dug[j] = True
        self.chest_h[j] = self.player_z
        self._save_chests()
        return msg

    def undo_dig(self):
        pos = self._pos()
        gone = np.nonzero(self.dug)[0]
        if not len(gone):
            return "nothing to undo"
        j = int(gone[np.argmin(np.hypot(*(self.world.chests[gone] - np.asarray(pos)).T))]) if pos else int(gone[-1])
        self.dug[j] = False
        return "chest #%d back" % (j + 1)

    def apply_scan(self, res):
        """Take a mapscan.ScanResult (the full in-game map, read with F11): which of the base spawns exist this match. Spawns you
        learned yourself keep their state, dug chests stay dug."""
        if not res.ok or res.present is None or len(res.present) != self.n_island:
            return res.message
        self.active[:self.n_island] = res.present
        ids = " ".join("#%d" % (i + 1) for i in np.nonzero(res.present)[0])
        return "%s: %s" % (res.message, ids)

    def reset_dug(self):
        self.dug[:] = False
        self.active = self.on_map.copy()
        self.floor = 0
        self.route_i = 0
        return "all chests back (new match)"
