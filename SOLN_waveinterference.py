import numpy as np
import astropy.units as u

# ── DIMENSIONS ────────────────────────────────────────────────────────────
len_UNIT    = u.m
time_UNIT   = u.s
angle_UNIT  = u.rad
k_UNIT      = angle_UNIT / len_UNIT
omega_UNIT  = angle_UNIT / time_UNIT


class WaveInterference2D:
    """
    Simulates N circular waves from N point sources at arbitrary locations
    on a 2D rectangular domain, interfering with each other. Each source
    may have its own wavelength, amplitude and frequency.

    If kick_time_i is None (the default), source i is on from the start:
        y_i(r, t) = A_i * cos(k_i*r - w_i*t)

    If kick_time_i is given, source i switches on at kick_time_i instead.
    is_ripple_i (default True) decides HOW it switches on:
      - is_ripple_i=True:  switches on at distance r only once the
        wavefront gets there, t = kick_time_i + r/v_i (v_i = w_i/k_i) --
        an expanding ring, using self.v.
      - is_ripple_i=False: switches on everywhere at once, at exactly
        kick_time_i -- self.v is not used at all.
    Either way, once on, it stays on:
        on = heaviside(t - kick_time_i - r/v_i)   if is_ripple_i
        on = heaviside(t - kick_time_i)           otherwise
        y_i(r, t) = A_i * on * cos(k_i*r - w_i*t)

    decay_tau_i (default None) makes amplitude drop after switch-on,
    for either kind of source, measured from whenever that point/source
    actually turned on (its own "elapsed" above, clipped at 0):
        y_i(r, t) = A_i * on * exp(-max(elapsed, 0)/decay_tau_i) * cos(k_i*r - w_i*t)

    All physical inputs may be passed in ANY compatible unit -- converted
    to the fixed internal units (len_UNIT, time_UNIT, k_UNIT, omega_UNIT)
    in __init__.

    Calculation only -- no plotting here, see wave_interference_plot.py.
    """

    def __init__(self, grid_size=200,
                 x_range=[-10.0, 10.0] * u.m,
                 y_range=[-10.0, 10.0] * u.m,
                 t_span=np.arange(0, 10, 0.05) * u.s,
                 sources=([-3, 0] * u.m, [3, 0] * u.m),
                 wavelength=(1.0 * u.m, 1.0 * u.m),
                 amplitude=(1.0 * u.m, 1.0 * u.m),
                 freq=(1.0 * u.Hz, 1.0 * u.Hz),
                 kick_time=(None, None),
                 is_ripple=(True, True),
                 decay_tau=(None, None)):

        x_range = x_range.to(len_UNIT).value
        y_range = y_range.to(len_UNIT).value
        self.x = np.linspace(x_range[0], x_range[1], grid_size)
        self.y = np.linspace(y_range[0], y_range[1], grid_size)
        self.X, self.Y = np.meshgrid(self.x, self.y)

        self.t_span = t_span.to(time_UNIT).value

        self.sources = [tuple(s.to(len_UNIT).value) for s in sources]
        self.n_sources = len(self.sources)

        self.wavelength_val = np.array([w.to(len_UNIT).value for w in wavelength])
        self.amplitude_val = np.array([a.to(len_UNIT).value for a in amplitude])
        self.freq_val = np.array([f.to(1 / time_UNIT).value for f in freq])
        self.kick_time = [None if kt is None else kt.to(time_UNIT).value for kt in kick_time]
        self.is_ripple = list(is_ripple)
        self.decay_tau = [None if dt_ is None else dt_.to(time_UNIT).value for dt_ in decay_tau]

        self.k = 2 * np.pi / self.wavelength_val   # (n_sources,), k_UNIT = rad/len_UNIT
        self.w = 2 * np.pi * self.freq_val          # (n_sources,), omega_UNIT = rad/time_UNIT
        self.A = self.amplitude_val                 # (n_sources,)
        self.v = self.w / self.k                     # (n_sources,), phase speed = wavelength*freq

        # distance from every grid point to every source: (n_sources, grid, grid)
        self.r = np.stack([
            np.sqrt((self.X - sx) ** 2 + (self.Y - sy) ** 2) for sx, sy in self.sources
        ])

    def field_at(self, t):
        """Return the combined wave height at every grid point for time t."""
        total = np.zeros_like(self.r[0])
        for i, kt in enumerate(self.kick_time):
            if kt is None:
                elapsed = t
                on = 1.0
            elif self.is_ripple[i]:
                elapsed = t - kt - self.r[i] / self.v[i]   # self.v used ONLY here
                on = np.heaviside(elapsed, 1.0)
            else:
                elapsed = t - kt                            # self.v not used
                on = np.heaviside(elapsed, 1.0)
            tau = self.decay_tau[i]
            if tau is not None:
                on = on * np.exp(-np.maximum(elapsed, 0.0) / tau)
            total += self.A[i] * on * np.cos(self.k[i] * self.r[i] - self.w[i] * t)
        return total

    def slice_waves_at(self, t, y0=0.0):
        """1D cross-section along y = y0: each source's wave and their sum."""
        waves = []
        for i, (sx, sy) in enumerate(self.sources):
            r = np.sqrt((self.x - sx) ** 2 + (y0 - sy) ** 2)
            kt = self.kick_time[i]
            if kt is None:
                elapsed = t
                on = 1.0
            elif self.is_ripple[i]:
                elapsed = t - kt - r / self.v[i]   # self.v used ONLY here
                on = np.heaviside(elapsed, 1.0)
            else:
                elapsed = t - kt                    # self.v not used
                on = np.heaviside(elapsed, 1.0)
            tau = self.decay_tau[i]
            if tau is not None:
                on = on * np.exp(-np.maximum(elapsed, 0.0) / tau)
            waves.append(self.A[i] * on * np.cos(self.k[i] * r - self.w[i] * t))
        total = np.sum(waves, axis=0)
        return waves, total


if __name__ == "__main__":
    # source 0: a true ripple (expands outward, self.v matters)
    # source 1: a constant source that just starts late (instant everywhere, self.v unused)
    sim = WaveInterference2D(
        grid_size=50,
        x_range=[-10.0, 10.0] * u.m,
        y_range=[-10.0, 10.0] * u.m,
        sources=([0, 0] * u.m, [5, 0] * u.m),
        wavelength=(1.0 * u.m, 1.0 * u.m),
        amplitude=(1.0 * u.m, 1.0 * u.m),
        freq=(1.0 * u.Hz, 1.0 * u.Hz),
        kick_time=(3.0 * u.s, 3.0 * u.s),
        is_ripple=(True, False),
    )
    v = sim.v[0]
    for r in [0, 1, 5, 8]:
        print(f"ripple (source 0): point at r={r}m switches on at t={3.0 + r/v}s")
    print("constant-starts-late (source 1): switches on everywhere at t=3.0s, no r dependence")