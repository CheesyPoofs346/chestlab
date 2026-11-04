"""The settings window: every option and action in one place (no key binds to remember).

SettingsPanel builds the widgets on any Tk root and reports what you change through two callbacks:
    set_value(key, value) -> message      a config value (radar position, refresh rate, full map on/off ...)
    do(action)            -> message      a one-off action (skip chest, new match, hide overlay, quit ...)
SettingsWindow runs a panel in a window of its own on its own thread, so the overlay loop never waits for it. It starts minimised (it must not
pop up over the game) and the close button minimises it again; Quit is a button."""
import threading
import tkinter as tk
from tkinter import ttk

from .options import FPS, FULLMAP, RADAR_AT, RADAR_M, RADAR_UP, label_of, value_of

ACTIONS = [("skip", "Skip the next route chest (it is not there)"), ("new_match", "New match: forget which chests I opened"),
           ("hide", "Hide / show the overlay")]


class SettingsPanel:
    def __init__(self, root, cfg, routes, route_now, set_value, do):
        self.root, self.cfg, self.set_value, self.do = root, cfg, set_value, do
        self.routes = [("No route: mark every chest", "0")] + [(n, str(i + 1)) for i, n in enumerate(routes)]
        self.vars = {}
        root.title("ChestLab settings")
        root.minsize(440, 0)
        pad = dict(padx=12, pady=(10, 0))
        f = ttk.Frame(root, padding=(0, 0, 0, 12))
        f.pack(fill="both", expand=True)
        self.status = tk.StringVar(value="Changes apply at once and are saved.")

        radar = ttk.LabelFrame(f, text="Radar")
        radar.pack(fill="x", **pad)
        self._combo(radar, "Position", "radar_at", RADAR_AT, cfg.get("radar_at", "default"))
        self._combo(radar, "Range", "radar_m", RADAR_M, float(cfg.get("radar_m", 0.0)))
        self._combo(radar, "Orientation", "radar_up", RADAR_UP, bool(cfg.get("radar_up", True)))

        game = ttk.LabelFrame(f, text="Game map and overlay")
        game.pack(fill="x", **pad)
        self._check(game, "Draw chests on the game's full map (M)", "fullmap", bool(cfg.get("fullmap", True)))
        self._combo(game, "Overlay refresh rate (fps)", "fps", FPS, int(cfg.get("fps", 60)))

        rt = ttk.LabelFrame(f, text="Route")
        rt.pack(fill="x", **pad)
        self.route_var = tk.StringVar(value=self._route_label(route_now))
        box = ttk.Combobox(rt, textvariable=self.route_var, values=[t for t, _ in self.routes], state="readonly", width=48)
        box.grid(row=0, column=0, padx=8, pady=8, sticky="we")
        box.bind("<<ComboboxSelected>>", lambda e: self.on_route())
        rt.columnconfigure(0, weight=1)
        self.route_box = box

        act = ttk.LabelFrame(f, text="Actions")
        act.pack(fill="x", **pad)
        for i, (name, text) in enumerate(ACTIONS):
            ttk.Button(act, text=text, command=lambda n=name: self.on_action(n)).grid(row=i, column=0, padx=8, pady=3, sticky="we")
        act.columnconfigure(0, weight=1)

        ttk.Button(f, text="Quit ChestLab", command=lambda: self.on_action("quit")).pack(fill="x", padx=12, pady=(12, 0))
        ttk.Label(f, textvariable=self.status, foreground="#555").pack(anchor="w", padx=12, pady=(8, 0))

    # ---- building blocks
    def _combo(self, parent, text, key, choices, value):
        row = ttk.Frame(parent)
        row.pack(fill="x", padx=8, pady=4)
        ttk.Label(row, text=text, width=26).pack(side="left")
        v = tk.StringVar(value=label_of(choices, value))
        box = ttk.Combobox(row, textvariable=v, values=[t for t, _ in choices], state="readonly", width=32)
        box.pack(side="left", fill="x", expand=True)
        box.bind("<<ComboboxSelected>>", lambda e, k=key, c=choices: self.on_choice(k, c))
        self.vars[key] = v

    def _check(self, parent, text, key, value):
        v = tk.BooleanVar(value=value)
        ttk.Checkbutton(parent, text=text, variable=v, command=lambda: self.on_check(key)).pack(anchor="w", padx=8, pady=4)
        self.vars[key] = v

    def _route_label(self, route_now):
        return next((t for t, v in self.routes if v == str(route_now)), self.routes[0][0]) if route_now else self.routes[0][0]

    # ---- what the user does
    def _report(self, msg):
        self.status.set(msg or "")

    def on_choice(self, key, choices):
        self._report(self.set_value(key, value_of(choices, self.vars[key].get())))

    def on_check(self, key):
        self._report(self.set_value(key, bool(self.vars[key].get())))

    def on_route(self):
        self._report(self.do("route:" + value_of(self.routes, self.route_var.get())))

    def on_action(self, name):
        self._report(self.do(name))


class SettingsWindow:
    """A SettingsPanel in a Tk window on its own thread. start() returns once it is built (or failed: .error)."""

    def __init__(self, cfg, routes, route_now, set_value, do, hidden=True):
        self.args, self.hidden = (cfg, routes, route_now, set_value, do), hidden
        self.root = None
        self.error = None
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True, name="settings")

    def start(self):
        self.thread.start()
        self.ready.wait(10)
        return self.error is None

    def _run(self):
        try:
            self.root = root = tk.Tk()
            root.withdraw()
            self.panel = SettingsPanel(root, *self.args)
            root.protocol("WM_DELETE_WINDOW", root.iconify)                  # the close button only tucks it away: Quit is a button
            if self.hidden:
                root.iconify()
            else:
                root.deiconify()
            self.ready.set()
            root.mainloop()
        except Exception as e:
            self.error = e
            self.ready.set()

    def close(self):
        r = self.root
        if r is not None:
            try:
                r.after(0, r.destroy)
            except Exception:
                pass
