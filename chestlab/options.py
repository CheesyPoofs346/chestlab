"""The options the settings screens offer (the start-up window and the ChestLab settings window share them): (label, value) per config key."""

RADAR_AT = [("Right side (default)", "default"), ("Left side", "left"), ("Top left", "top-left"), ("Top right", "top-right"),
            ("Bottom left", "bottom-left"), ("Bottom right", "bottom-right")]
RADAR_M = [("Auto: zooms in as you reach a chest", 0.0), ("20 m", 20.0), ("40 m", 40.0), ("80 m", 80.0)]
RADAR_UP = [("Ahead is up (turns with you)", True), ("North is up", False)]
FPS = [("60 (recommended)", 60), ("90", 90), ("120", 120), ("144", 144)]
FULLMAP = [("On", True), ("Off", False)]

# the start-up window has less room: short labels
SHORT = {"radar_at": [("Right", "default"), ("Left", "left"), ("Top left", "top-left"), ("Top right", "top-right"),
                      ("Bottom left", "bottom-left"), ("Bottom right", "bottom-right")],
         "radar_m": [("Auto", 0.0), ("20 m", 20.0), ("40 m", 40.0), ("80 m", 80.0)],
         "radar_up": [("Ahead is up", True), ("North is up", False)],
         "fullmap": FULLMAP,
         "fps": [("60", 60), ("90", 90), ("120", 120), ("144", 144)]}
ROWS = [("Radar position", "radar_at"), ("Radar range", "radar_m"), ("Radar turns with you", "radar_up"),
        ("Chests on the game's full map (M)", "fullmap"), ("Overlay refresh rate (fps)", "fps")]
DEFAULTS = dict(radar_at="default", radar_m=0.0, radar_up=True, fullmap=True, fps=60)


def label_of(choices, value):
    return next((t for t, v in choices if v == value), choices[0][0])


def value_of(choices, text):
    return next((v for t, v in choices if t == text), choices[0][1])
