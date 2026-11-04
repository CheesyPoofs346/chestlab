"""Camera heading as a continuous, time-stamped estimate. The game prints the heading as an integer on the compass bar;
reading it is the slow part (~5 ms), so the readings arrive at 50-100 Hz with dropouts, quantisation and the odd
misread. This filter turns them into a smooth heading + turn rate that can be asked for at any instant (the renderer
asks every frame, 60+ Hz), extrapolating over the gaps. State: unwrapped angle theta (deg) and rate omega (deg/s)."""
import threading

import numpy as np

OCR_SIGMA = 0.40      # deg: integer readout (uniform +-0.5 -> 0.29) plus a little for floor-vs-round
BAR_SIGMA = 1.10      # deg: heading from the label comb on the compass bar (measured against OCR on real shots)
ACCEL_SIGMA = 1800.0  # deg/s^2: how hard a mouse flick can change the turn rate (process noise)
MAX_RATE = 1500.0     # deg/s: anything faster is not a human turn


def wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def extrapolate(th, om, t_state, t_meas, t):
    """Heading at time t from a filter state (angle th and rate om valid at t_state; last reading used at t_meas).
    -> (heading deg in [0, 360), rate deg/s, seconds since the last reading)."""
    dt = min(t - t_state, 0.25)             # never coast for long: after that the heading is a guess anyway
    om = om * (0.5 if dt > 0.12 else 1.0)
    return (th + om * dt) % 360.0, om, t - t_meas


class HeadingFilter:
    def __init__(self):
        self.lock = threading.Lock()
        self.reset()

    def reset(self):
        self.th = self.om = None
        self.P = np.diag([4.0, 400.0 ** 2])
        self.t = None
        self.odd = None          # (t, deg) of a rejected reading, waiting for a second one that agrees
        self.n = 0
        self.last_meas_t = None

    def _predict(self, t):
        dt = t - self.t
        if dt <= 0:
            return
        F = np.array([[1.0, dt], [0.0, 1.0]])
        Q = ACCEL_SIGMA ** 2 * np.array([[dt ** 4 / 4, dt ** 3 / 2], [dt ** 3 / 2, dt * dt]])
        self.th += self.om * dt
        self.P = F @ self.P @ F.T + Q
        self.t = t

    def update(self, t, meas, sigma=OCR_SIGMA, gate=None):
        """meas: heading in degrees (any wrap). Returns True if it was used. A reading that disagrees with the
        prediction is held back until a second one agrees with it (a real flick shows up in both; a misread does not)."""
        with self.lock:
            if self.th is None:
                self.th, self.om, self.t = float(meas), 0.0, t
                self.P = np.diag([sigma ** 2, 300.0 ** 2])
                self.last_meas_t, self.n = t, 1
                return True
            if t < self.t:
                return False
            self._predict(t)
            e = wrap180(meas - self.th)
            s = np.sqrt(self.P[0, 0] + sigma ** 2)
            lim = gate if gate is not None else max(2.0, 5.0 * s)
            if abs(e) > lim:
                o = self.odd
                if o is not None and t - o[0] < 0.25 and abs(wrap180(meas - o[1])) < 2.0 + MAX_RATE * (t - o[0]) * 0.2:
                    self.th, self.om, self.t = float(meas), 0.0, t         # two readings agree: genuine jump, take it
                    self.P = np.diag([sigma ** 2, 300.0 ** 2])
                    self.odd, self.last_meas_t = None, t
                    return True
                self.odd = (t, float(meas))
                return False
            self.odd = None
            H = np.array([1.0, 0.0])
            S = self.P[0, 0] + sigma ** 2
            K = self.P @ H / S
            self.th += K[0] * e
            self.om = float(np.clip(self.om + K[1] * e, -MAX_RATE, MAX_RATE))
            self.P = (np.eye(2) - np.outer(K, H)) @ self.P
            self.last_meas_t = t
            self.n += 1
            return True

    def at(self, t):
        """-> (heading deg in [0, 360), rate deg/s, seconds since the last reading used) or None before the first one."""
        with self.lock:
            if self.th is None:
                return None
            return extrapolate(self.th, self.om, self.t, self.last_meas_t, t)

    def export(self):
        """(th, om, t_state, t_meas) for another process to extrapolate from, or None before the first reading."""
        with self.lock:
            return None if self.th is None else (self.th % 360.0, self.om, self.t, self.last_meas_t)

    def prior(self, t):
        r = self.at(t)
        return None if r is None else r[0]
