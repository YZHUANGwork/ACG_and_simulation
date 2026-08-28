

import numpy as np
import astropy.units as u
from astropy.timeseries import LombScargle

# ── DIMENSIONS ────────────────────────────────────────────────────────────

time_UNIT   = u.d
freq_UNIT = 1/time_UNIT
ang_UNIT = u.rad
ang_v_UNIT = ang_UNIT/time_UNIT


class TimeSeriesPeriodogram:
    """
    Build a (optionally noisy) periodic counts time series and run a
    Lomb-Scargle periodogram over a scanned frequency range.
    """

    def __init__(self,
                 Ad=0.03,
                 Phase=0 * u.deg,
                 N_in_bin=10, Nbkg_in_bin=0,
                 P=1*u.yr, T=5 * u.yr, dt=5*u.day,
                 freqs=np.linspace(((1/(1*u.yr)).to(1/u.s)).value,1, 2000)/u.s ,
                 noise_type='none', threshold = 0.1,
                 seed=1):
        """
        Parameters
        ----------
        Ad           : float     fractional modulation amplitude, unitless
        Phase        : Quantity  signal phase [angle]
        N_in_bin     : float     mean signal counts per bin [#]
        Nbkg_in_bin  : float     mean background counts per bin [#]
        P            : Quantity  signal period [time]
        T            : Quantity  total observation span [time]
        dt           : Quantity  bin width [time]
        freqs        : ndarray or None  frequencies to scan [Hz, i.e. 1/s]
        noise_type   : 'poisson' | 'none'
        seed         : int or None, RNG seed for reproducibility
        """
        self.Ad = Ad
        self.Phase = Phase
        self.Phase_rad = self.Phase.to(ang_UNIT).value

        self.N_in_bin = N_in_bin
        self.Nbkg_in_bin = Nbkg_in_bin

        self.P = P
        self.T = T
        self.dt = dt
        self.period = self.P.to(time_UNIT)

        self.freqs = freqs.to(freq_UNIT)

        self.noise_type = noise_type
        self.rng = np.random.default_rng(seed)

        self.time_bins, self.time_bin_centers = self.make_time_bins()

        # populated as the pipeline runs
        self.noiseless_ts = None
        self.counts_ts = None
        self.power = None
        self.distance_matrix = None
        self.recurrence_matrix = None
        self.threshold = threshold

    # ── time base ────────────────────────────────────────────────────────
    def make_time_bins(self):
        """

        time_bins        : bin edges [time_UNIT], length N+1
        time_bin_centers : bin centers [time_UNIT], length N
        """
        T = self.T.to(time_UNIT)
        dt = self.dt.to(time_UNIT)
        time_bins = np.arange(0.0, (T + dt).value, dt.value) * time_UNIT
        time_bin_centers = 0.5 * (time_bins[1:] + time_bins[:-1])
        return time_bins, time_bin_centers

    # ── signal construction ─────────────────────────────────────────────
    def make_signal_timeseries(self):
        """
        Build the noiseless expected-counts time series for a single
        sinusoidal signal riding on a flat background.

        Returns
        -------
        total_counts_ts : ndarray  noiseless expected counts per bin [#]
        """
        period = self.period.to(time_UNIT)
        dt = self.dt.to(time_UNIT)

        omega = 2 * np.pi * u.rad / period              # [rad/s]
        t0 = self.Phase_rad / (2 * np.pi) * period       # [s]

        time_var = (np.sin(omega * (self.time_bins[1:] - t0)) -
                    np.sin(omega * (self.time_bins[:-1] - t0))) / omega   

        mod_true_counts = self.Ad * self.N_in_bin                          
        modulation_ts = (mod_true_counts * time_var / dt).to_value(
            u.dimensionless_unscaled, equivalencies=u.dimensionless_angles())  
        total_counts_ts = ((self.N_in_bin + self.Nbkg_in_bin) *
                            np.ones(len(self.time_bin_centers)) + modulation_ts)

        self.noiseless_ts = total_counts_ts
        return total_counts_ts

    # ── noise ────────────────────────────────────────────────────────────
    def add_noise(self):

        if self.noiseless_ts is None:
            self.make_signal_timeseries()
        input_counts_ts = self.noiseless_ts

        if self.noise_type == 'none':
            noisy_ts = input_counts_ts.copy()
        elif self.noise_type == 'poisson':
            noisy_ts = self.rng.poisson(np.maximum(input_counts_ts, 0)).astype(float)
        else:
            raise ValueError(f"unknown noise_type: {self.noise_type}")

        self.counts_ts = noisy_ts
        return noisy_ts

    # ── periodogram ──────────────────────────────────────────────────────
    def run_periodogram(self):
        
        if self.freqs is None:
            raise ValueError("self.freqs is None; pass freqs= to the constructor")

        if self.counts_ts is None:
            self.add_noise()
        data = self.counts_ts

        t = self.time_bin_centers.to(time_UNIT)
        f = self.freqs.to(freq_UNIT)
        power = (LombScargle(t, data, normalization='psd').power(f) / data.var(ddof=1))

        self.power = power
        return power

    # ── recurrence plot ──────────────────────────────────────────────────
    def recurrence_plot(self):
        """
        recurrence matrix: recurrence_matrix[i, j] = True if
        |data[i] - data[j]| < threshold.

        Parameters
        ----------
        threshold : float  distance threshold, same units as the counts data [#]

        Returns
        -------
        distance_matrix    : ndarray (N, N)  |data[i] - data[j]| [#]
        recurrence_matrix  : ndarray (N, N), bool  distance_matrix < threshold
        """
        if self.counts_ts is None:
            self.add_noise()
        data = self.counts_ts

        N = len(data)
        distance_matrix = np.zeros((N, N))
        for i in range(N):
            for j in range(N):
                distance_matrix[i, j] = np.abs(data[i] - data[j])

        recurrence_matrix = distance_matrix < self.threshold
        
        self.distance_matrix = distance_matrix
        self.recurrence_matrix = recurrence_matrix
        return distance_matrix, recurrence_matrix


if __name__ == "__main__":
    Period = 1*u.yr
    f0 = (1/Period).to(1/u.s)
    
    freqs = np.linspace(f0.value,1, 2000)/u.s   # [Hz]

    tsp = TimeSeriesPeriodogram(
        Ad=0.1, Phase=0.5 * u.rad, N_in_bin=100, Nbkg_in_bin=20,
        P=Period, T=5 * u.yr, dt=5*u.day,
        freqs=freqs,
        noise_type='poisson', seed=42,
    )

    power = tsp.run_periodogram()
    best_freq = freqs[np.argmax(power)]
    print(f"n_bins: {len(tsp.counts_ts)}")
    print(f"best-fit period: {1.0 / best_freq:.1f} s (true: {tsp.period.to(u.s).value} s)")

    distance_matrix, recurrence_matrix = tsp.recurrence_plot(threshold=5.0)
    print(f"recurrence fraction: {recurrence_matrix.mean():.3f}")