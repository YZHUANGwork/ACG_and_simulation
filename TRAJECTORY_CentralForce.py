import numpy as np
import astropy.units as u
import astropy.constants as const
from scipy.integrate import solve_ivp

len_UNIT = u.m
time_UNIT = u.s
mass_UNIT = u.kg
vel_UNIT = len_UNIT / time_UNIT
acc_UNIT = vel_UNIT / time_UNIT
"r(nu) = l / (1+e*cos(phi))"


class KeplerOrbit:
    """
    Represents a Keplerian central-force orbit (ellipse, parabola, or
    hyperbola) and provides two kinds of trajectory:

      1. Shape only:      r(nu) = l / (1 + e * cos(nu))
         (purely geometric, no notion of time -- `shape()`)

      2. Time-domain:     (x(t), y(t)) obtained by numerically solving
         Kepler's Time Equation for the anomaly at each requested time
         (`trajectory_over_time()`).

    Parameters
    ----------
    eccentricity : float
        Orbital eccentricity e >= 0.
        e == 0        -> circle
        0 < e < 1     -> ellipse
        e == 1        -> parabola
        e > 1         -> hyperbola
    semi_major_axis : float
        For ellipse/hyperbola: the (magnitude of the) semi-major axis 'a'
        (meters, or consistent units). For a parabola there is no finite
        semi-major axis, so this parameter is instead interpreted as the
        periapsis distance q (l = 2*q).
    semi_major_axis : float or astropy.units.Quantity
        For ellipse/hyperbola: the (magnitude of the) semi-major axis 'a'.
        For a parabola there is no finite semi-major axis, so this
        parameter is instead interpreted as the periapsis distance q
        (l = 2*q). Accepts a plain float (assumed meters) or an astropy
        Quantity with any length unit, e.g. `1.0 * u.AU` -- convenient
        for solar-system-scale orbits around the Sun.
    mu : float or astropy.units.Quantity, optional
        Gravitational parameter GM of the central body. Accepts a plain
        float (assumed m^3/s^2) or an astropy Quantity. Defaults to the
        Sun's GM if not supplied.
    """

    def __init__(self, eccentricity, semi_major_axis, mu=None):
        if eccentricity < 0:
            raise ValueError("eccentricity must be >= 0")
        self.e = float(eccentricity)

        # Remember the unit the caller used (for convenient display later),
        # but always do the actual numerics in SI (meters).
        if isinstance(semi_major_axis, u.Quantity):
            self.a_unit = semi_major_axis.unit
            self.a = semi_major_axis.to_value(u.m)
        else:
            self.a_unit = u.m
            self.a = float(semi_major_axis)

        if mu is not None:
            self.mu = mu.to_value(u.m ** 3 / u.s ** 2) if isinstance(mu, u.Quantity) else float(mu)
        else:
            self.mu = const.GM_sun.to_value(u.m ** 3 / u.s ** 2)

        # Semi-latus rectum l (a.k.a. p) -- always stored in meters
        if self.e < 1.0:
            self.l = self.a * (1 - self.e ** 2)
        elif self.e == 1.0:
            # 'a' is treated as periapsis distance q for a parabola
            self.l = 2 * self.a
        else:
            self.l = self.a * (self.e ** 2 - 1)

    def to_unit(self, x_meters, unit=None):
        """
        Convenience: convert a plain-float array of meters (as returned by
        `shape()` / `trajectory_over_time()`) into an astropy Quantity in
        `unit` (defaults to whatever unit `semi_major_axis` was given in,
        e.g. u.AU), for display purposes.
        """
        unit = unit or self.a_unit
        return (x_meters * u.m).to(unit)

    # ------------------------------------------------------------------
    # 1) Orbit shape: r(nu) = l / (1 + e*cos(nu))
    # ------------------------------------------------------------------
    def shape(self, num_points=500, nu=None):
        """
        Returns the static orbit shape (no time dependence).

        Parameters
        ----------
        num_points : int
            Number of samples, used only if `nu` is not supplied.
        nu : array-like, optional
            True-anomaly values (radians) at which to evaluate r(nu).
            If omitted, a sensible range is generated automatically
            depending on orbit type (full 2*pi for ellipse, clipped
            near the asymptotes for parabola/hyperbola).

        Returns
        -------
        nu, r, x, y : ndarray
        """
        if nu is None:
            if self.e < 1.0:
                nu = np.linspace(-np.pi, np.pi, num_points)
            elif self.e == 1.0:
                nu_max = np.pi - 0.1
                nu = np.linspace(-nu_max, nu_max, num_points)
            else:
                nu_asymptote = np.arccos(-1.0 / self.e)
                nu_max = nu_asymptote - 0.05
                nu = np.linspace(-nu_max, nu_max, num_points)
        else:
            nu = np.asarray(nu, dtype=float)

        r = self.l / (1 + self.e * np.cos(nu))
        x = r * np.cos(nu)
        y = r * np.sin(nu)
        return nu, r, x, y

    # ------------------------------------------------------------------
    # 2) Kepler's Time Equation, solved numerically for each orbit type
    # ------------------------------------------------------------------
    def mean_motion(self):
        """n = sqrt(mu / |a|^3). Undefined (returns None) for a parabola."""
        if self.e == 1.0:
            return None
        return np.sqrt(self.mu / self.a ** 3)

    def solve_kepler_elliptic(self, M, tol=1e-12, max_iter=100):
        """
        Solve Kepler's equation  M = E - e*sin(E)  for the eccentric
        anomaly E, via Newton-Raphson iteration.
        """
        M = np.atleast_1d(np.asarray(M, dtype=float))
        # Wrap into [-pi, pi] for a numerically stable starting guess,
        # keeping track of how many full revolutions were wrapped away.
        M_wrapped = np.mod(M + np.pi, 2 * np.pi) - np.pi
        E = np.where(self.e < 0.8, M_wrapped, np.sign(M_wrapped) * np.pi)

        for _ in range(max_iter):
            f = E - self.e * np.sin(E) - M_wrapped
            fp = 1 - self.e * np.cos(E)
            dE = -f / fp
            E = E + dE
            if np.max(np.abs(dE)) < tol:
                break

        return E + (M - M_wrapped)

    def solve_kepler_hyperbolic(self, M, tol=1e-12, max_iter=200):
        """
        Solve the hyperbolic Kepler equation  M = e*sinh(H) - H  for the
        hyperbolic anomaly H, via Newton-Raphson iteration.
        """
        M = np.atleast_1d(np.asarray(M, dtype=float))
        with np.errstate(divide="ignore", invalid="ignore"):
            H = np.sign(M) * np.log(2 * np.abs(M) / self.e + 1.8)
        H = np.where(M == 0, 0.0, H)

        for _ in range(max_iter):
            f = self.e * np.sinh(H) - H - M
            fp = self.e * np.cosh(H) - 1
            dH = -f / fp
            H = H + dH
            if np.max(np.abs(dH)) < tol:
                break

        return H

    def solve_kepler_parabolic(self, M):
        """
        Solve Barker's equation for parabolic motion,
        M = D + D^3/3   (D = tan(nu/2)),
        using Cardano's closed-form solution of the depressed cubic
        D^3 + 3D - 3M = 0. Writing D = w - 1/w reduces this to the
        quadratic w^3 - 1/w^3 = 3M in disguise, giving
        w = cbrt((3M + sqrt(9M^2 + 4)) / 2), D = w - 1/w.
        """
        M = np.atleast_1d(np.asarray(M, dtype=float))
        A = 3 * M
        B = np.sqrt(A ** 2 + 4)
        C = np.cbrt((B + A) / 2)
        C_safe = np.where(C == 0, 1e-300, C)
        D = C_safe - 1 / C_safe
        return D

    # ------------------------------------------------------------------
    # time -> anomaly / position
    # ------------------------------------------------------------------
    def time_to_true_anomaly(self, t, t_peri=0.0):
        """
        Given time(s) `t` and the time of periapsis passage `t_peri`
        (same units as `t`, consistent with mu), return the true
        anomaly nu(t) obtained by numerically solving Kepler's
        Time Equation for the appropriate orbit type.
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        dt = t - t_peri

        if self.e < 1.0:
            n = self.mean_motion()
            M = n * dt
            E = self.solve_kepler_elliptic(M)
            nu = 2 * np.arctan2(
                np.sqrt(1 + self.e) * np.sin(E / 2),
                np.sqrt(1 - self.e) * np.cos(E / 2),
            )
        elif self.e == 1.0:
            # Barker's equation mean anomaly for a parabola
            M = np.sqrt(self.mu / (2 * self.l ** 3)) * dt
            D = self.solve_kepler_parabolic(M)
            nu = 2 * np.arctan(D)
        else:
            n = np.sqrt(self.mu / self.a ** 3)
            M = n * dt
            H = self.solve_kepler_hyperbolic(M)
            nu = 2 * np.arctan2(
                np.sqrt(self.e + 1) * np.sinh(H / 2),
                np.sqrt(self.e - 1) * np.cosh(H / 2),
            )
        return nu

    def trajectory_over_time(self, t, t_peri=0.0):
        """
        Full time-domain trajectory: for each time in `t`, solves
        Kepler's Time Equation to get the true anomaly, then converts
        to polar/Cartesian coordinates.

        Parameters
        ----------
        t : array-like
            Times at which to evaluate the trajectory.
        t_peri : float
            Time of periapsis passage (same time reference as `t`).

        Returns
        -------
        dict with keys 't', 'nu', 'r', 'x', 'y' (all ndarray, same length as t)
        """
        t = np.atleast_1d(np.asarray(t, dtype=float))
        nu = self.time_to_true_anomaly(t, t_peri)
        r = self.l / (1 + self.e * np.cos(nu))
        x = r * np.cos(nu)
        y = r * np.sin(nu)
        return {"t": t, "nu": nu, "r": r, "x": x, "y": y}

    def true_anomaly_to_time(self, nu, t_peri=0.0):
        """
        Inverse of `time_to_true_anomaly`: given true anomaly/anomalies
        `nu` (radians), return the time(s) since periapsis passage
        `t_peri`, by inverting the same E/H/D anomaly relations and then
        applying Kepler's equation forward (M -> t).
        """
        nu = np.atleast_1d(np.asarray(nu, dtype=float))
        e = self.e

        if e < 1.0:
            E = 2 * np.arctan2(
                np.sqrt(1 - e) * np.sin(nu / 2),
                np.sqrt(1 + e) * np.cos(nu / 2),
            )
            M = E - e * np.sin(E)
            n = self.mean_motion()
            dt = M / n
        elif e == 1.0:
            D = np.tan(nu / 2)
            M = D + D ** 3 / 3
            n = np.sqrt(self.mu / (2 * self.l ** 3))
            dt = M / n
        else:
            H = 2 * np.arctanh(np.sqrt((e - 1) / (e + 1)) * np.tan(nu / 2))
            M = e * np.sinh(H) - H
            n = np.sqrt(self.mu / self.a ** 3)
            dt = M / n

        return dt + t_peri

    def get_trajectory(self, num_points=500):
        """
        Class-method form of the module-level `get_trajectory()` function:
        returns just (x, y) for this instance's static orbit shape.
        Equivalent to `self.shape(num_points)[2:4]`. Kept so callers can
        do `orbit.get_trajectory(...)` instead of re-deriving the shape
        logic; `shape()` remains the richer method (also returns nu, r).
        """
        _, _, x, y = self.shape(num_points=num_points)
        return x, y


def get_trajectory(eccentricity, semi_major_axis, num_points=500):
    """
    Module-level convenience wrapper, kept for backward compatibility with
    older code that calls `get_trajectory(e, a, num_points)` directly
    instead of going through the class. All the actual logic now lives on
    KeplerOrbit; this just builds an instance and delegates to it.

    Parameters:
    - eccentricity (float): e >= 0
    - semi_major_axis (float): 'a' parameter (periapsis distance q if e == 1)
    """
    return KeplerOrbit(eccentricity, semi_major_axis).get_trajectory(num_points=num_points)



class CentralForceScattering:
    """
    off center scattering , repulsive or attractive
    Parameters
    ----------
    k : length^3/time^2
    
    """
 
    def __init__(self, k, force_type="attractive"):
        self.k = k.to_value(len_UNIT ** 3 / time_UNIT ** 2) if isinstance(k, u.Quantity) else float(k)
        self.sign = -1.0 if force_type == "attractive" else 1.0
 
    def rhs(self, t, state):
        """The equation of motion: dr/dt = v, dv/dt = sign*k*r/|r|^3."""
        r, v = state[:3], state[3:]
        a = self.sign * self.k * r / np.linalg.norm(r) ** 3
        return np.concatenate([v, a])
 
    def integrate_trajectory(
        self, r0, v0, t_span, num_points=500,
        rtol=1e-10, atol=1e-10, method="DOP853",
    ):
        """
        solve_ivp:
 
            dr/dt = v
            dv/dt = sign * k * r / |r|^3     
 
        
        -------
        
            sol.t          -- time, in time_UNIT (s)
            sol.y[0:3]     -- position (x, y, z), in len_UNIT (m)
            sol.y[3:6]     -- velocity (vx, vy, vz), in vel_UNIT (m/s)
        """
        state0 = np.concatenate([r0.to_value(len_UNIT), v0.to_value(vel_UNIT)])
        t0, t1 = t_span.to_value(time_UNIT)
 
        return solve_ivp(self.rhs, (t0, t1), state0, t_eval=np.linspace(t0, t1, num_points),
                          rtol=rtol, atol=atol, method=method)
 
if __name__ == "__main__":
    # Quick self-test / demonstration for all three orbit types.
    mu_sun = const.GM_sun.to(u.m ** 3 / u.s ** 2).value

    for e, a, label in [
        (0.6, 1.5e11, "ellipse"),
        (1.0, 1.0e11, "parabola (a = periapsis q)"),
        (1.5, 1.0e11, "hyperbola"),
    ]:
        orbit = KeplerOrbit(eccentricity=e, semi_major_axis=a, mu=mu_sun)

        nu, r, x, y = orbit.shape(num_points=10)
        print(f"[{label}] shape(): nu[0:3]={nu[:3]}, r[0:3]={r[:3]}")

        if e < 1.0:
            period = 2 * np.pi / orbit.mean_motion()
            t = np.linspace(0, period, 10)
        else:
            t = np.linspace(-1e7, 1e7, 10)

        traj = orbit.trajectory_over_time(t)
        print(f"[{label}] trajectory_over_time(): x[0:3]={traj['x'][:3]}")
        print()

    # ---- Demonstrating astropy Quantity input, e.g. distances in AU ----
    earth_like = KeplerOrbit(eccentricity=0.4, semi_major_axis=1.0 * u.AU)  # mu defaults to Sun's GM
    period_years = (2 * np.pi / earth_like.mean_motion() * u.s).to(u.yr)
    print(f"[AU input] a = 1.0 AU, e = 0.4 -> period = {period_years:.3f}")

    _, r, x, y = earth_like.shape(num_points=5)
    print(f"[AU input] r(nu) in AU: {earth_like.to_unit(r, u.AU)}")