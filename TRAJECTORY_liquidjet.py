"""
liquid jet: cross-section exit -> coherent streamline -> Rayleigh-Plateau
breakup -> ballistic droplet spray.

Stage 1 - Bernoulli (cross-section exit velocity):
continuity equation 

Stage 2 - 
the length where stream line will break up into drops, NOT SURE ABOUT THE FORMULA AND VALUE
source are 1966 unaccessible ,LINE 125-143
https://arxiv.org/pdf/2412.15974 
https://aiche.onlinelibrary.wiley.com/doi/abs/10.1002/aic.690120411

Stage 3 - drops trajectory:
    state = [x, z, vx, vz, s]   (s = arc length traveled along path)
    while s < L_breakup:  no drag, pure projectile motion
        xddot = 0
        zddot = -g
    once s >= L_breakup: quadratic air drag on the (now discrete) droplet
        m*vdot = -m*g*zhat - 1/2*rho_air*Cd_drop*A_drop*|v|*v
   
"""
import numpy as np
from scipy.integrate import solve_ivp
from scipy.special import iv               
from scipy.optimize import minimize_scalar
import astropy.units as u
import astropy.constants as const
import random


def _rayleigh_growth_shape(x):
    """omega^2/omega_0^2 = x*I1(x)/I0(x)*(1-x^2), x=k*a, omega_0^2=sigma/(rho*a^3).
    Rayleigh (1878/79), Proc. R. Soc. London. Eq. (4.2) in arXiv:2012.01887."""
    return x * iv(1, x) / iv(0, x) * (1.0 - x ** 2)



def _solve_rayleigh_constants():
    """Maximizes _rayleigh_growth_shape -> x_max=0.697, no closed form.
    Returns drop_diameter/jet_diameter=1.89.
    x_max: link.aps.org/accepted/10.1103/PhysRevFluids.4.113603
    1.89: grokipedia.com/page/Plateau%E2%80%93Rayleigh_instability"""
    res = minimize_scalar(lambda x: -_rayleigh_growth_shape(x),
                           bounds=(1e-6, 1.0 - 1e-9), method='bounded')
    x_max = res.x                              # ~0.697
    lambda_over_d = np.pi / x_max              # lambda = 2*pi*a/x_max, a = d/2 -> ~4.51
    # one breakup wavelength of jet volume collapses (surface tension,
    # volume-conserving) into one sphere:
    #   (pi/4) d^2 * lambda_max = (pi/6) d_drop^3  ->  d_drop/d = (1.5*lambda/d)^(1/3)
    drop_over_d = (1.5 * lambda_over_d) ** (1.0 / 3.0)   # ~1.89
    return drop_over_d


RAYLEIGH_DROP_OVER_D = _solve_rayleigh_constants()


# ── DIMENSIONS ────────────────────────────────────────────────────────────
length_UNIT   = u.m
time_UNIT     = u.s
angle_UNIT    = u.rad
mass_UNIT     = u.kg
accel_UNIT    = length_UNIT / time_UNIT**2
velocity_UNIT = length_UNIT / time_UNIT

pressure_UNIT        = u.Pa                      # P_pump - P_atm, gauge
density_UNIT         = mass_UNIT / length_UNIT**3 # rho, rho_air
surface_tension_UNIT = u.N / length_UNIT          # sigma
viscosity_UNIT       = u.Pa * time_UNIT           # mu (dynamic viscosity)
temperature_UNIT     = u.K                        # T, for thermal capillary wave amplitude

class FountainJet:
    """Water jet from a pressurized/pumped cross section: Bernoulli exit velocity,
    Rayleigh-Plateau breakup length, then ballistic spray of droplets."""

    def __init__(self,
                 pump_pressure=6.0e4*pressure_UNIT,        # gauge pressure at cross section, P_pump - P_atm
                 cross_section_diameter=0.006*length_UNIT,          # d
                 angle=80.0*u.deg,                           # launch angle from horizontal
                 origin_x=0.0*length_UNIT,                    # x position of the cross section (launch point)
                 origin_z=0.0*length_UNIT,                    # z position of the cross section (launch point)
                 
                 rho=1000.0*density_UNIT,                     # water density
                 rho_air=1.225*density_UNIT,                  # air density
                 surface_tension=0.0728*surface_tension_UNIT, # sigma, water-air
                 viscosity=1.0e-3*viscosity_UNIT,             # mu, water
                 discharge_coeff=0.9,                          # discharge coefficient, cross section (dimensionless, ~0.6-0.98)
                 drag_coeff=0.47,                              # drag coefficient, droplet ~ sphere (dimensionless, ~0.47)
                 temperature=293.15*temperature_UNIT,          # T, for thermal capillary wave amplitude (Kooij et al. 2024)
                 t_total=2.0*time_UNIT, dt=0.001*time_UNIT,
                 n_droplets=25,                                # number of discrete droplets in the spray
                 jitter_angle=3.0*angle_UNIT,                  # random angle spread applied at breakup (turbulence)
                 jitter_speed_frac=0.06,                       # random speed spread applied at breakup, fraction of v0
                 seed=42):                                     # RNG seed for reproducible spray jitter

        self.d       = cross_section_diameter.to(length_UNIT).value
        self.theta   = angle.to(angle_UNIT).value
        self.x0      = origin_x.to(length_UNIT).value
        self.z0      = origin_z.to(length_UNIT).value
        self.rho     = rho.to(density_UNIT).value
        self.rho_air = rho_air.to(density_UNIT).value
        self.sigma   = surface_tension.to(surface_tension_UNIT).value
        self.mu      = viscosity.to(viscosity_UNIT).value
        self.P       = pump_pressure.to(pressure_UNIT).value
        self.discharge_coeff = discharge_coeff
        self.drag_coeff   = drag_coeff
        self.g       = const.g0.to(accel_UNIT).value
        self.T       = temperature.to(temperature_UNIT).value
        self.t_total = t_total.to(time_UNIT).value
        self.dt      = dt.to(time_UNIT).value

        self.n_droplets        = n_droplets
        self.jitter_angle      = jitter_angle.to(angle_UNIT).value
        self.jitter_speed_frac = jitter_speed_frac
        self.seed              = seed

        # v0 = Cd * sqrt(2*(P_pump - P_atm)/rho)
        #INITIAL VELOCITY
        self.v0 = self.discharge_coeff * np.sqrt(max(2.0 * self.P / self.rho, 0.0))

        #FLUID DYNAMICS PARAMETERS
        Oh = self.mu / np.sqrt(self.rho * self.sigma * self.d)#https://en.wikipedia.org/wiki/Ohnesorge_number
        We = self.rho * self.d * self.v0 ** 2 / self.sigma #https://en.wikipedia.org/wiki/Weber_number
        self.Re = self.rho * self.d * self.v0 / self.mu #https://en.wikipedia.org/wiki/Reynolds_number
        #https://aiche.onlinelibrary.wiley.com/doi/epdf/10.1002/aic.690120411
        #"Newtonian Jet Stability" by Rollin P. Grant and Stanley Middleman
        #unsure about this
        #
        #https://arxiv.org/pdf/2412.15974
        kB = const.k_B.to(u.J / u.K).value
        delta0 = np.sqrt(kB * self.T / (np.sqrt(8.0) * np.pi ** 2 * self.sigma))
        self.breakup_length = self.d * np.log(self.d / (2.0 * delta0)) * np.sqrt(We) * (1.0 + 3.0 * Oh)
 
        self.breakup_length_is_extrapolated = self.Re >= 2300.0
        if self.breakup_length_is_extrapolated:
            import warnings
            warnings.warn(
                f"Re={self.Re:.0f} >= 2300: jet is turbulent, outside the "
                "laminar regime Kooij et al. 2024 validated (arXiv:2412.15974). "
                "breakup_length is an extrapolation, not a verified prediction.",
                stacklevel=2,
            )
            
        # A = pi*(d/2)^2
        self.cross_section_area = np.pi * (self.d / 2.0) ** 2
 
        # Q = A * v0
        self.flow_rate = self.cross_section_area * self.v0
 

        d_drop = RAYLEIGH_DROP_OVER_D * self.d
        area_drop = np.pi * (d_drop / 2.0) ** 2
        volume_drop = (4.0 / 3.0) * np.pi * (d_drop / 2.0) ** 3
        mass_drop = self.rho * volume_drop
        self.k_drag = 0.5 * self.rho_air * self.drag_coeff * area_drop / mass_drop

        self.sol = None             # last solve_ivp result (single trajectory)
        self._spray = None          # last solve_spray() result




    # state = [x, z, vx, vz, s]   (s = cumulative arc length, sets regime)
    def eom(self, t, state):
        x, z, vx, vz, s = state
        speed = np.hypot(vx, vz)
        if s < self.breakup_length:
            ax, az = 0.0, -self.g
        else:
            ax = -self.k_drag * speed * vx
            az = -self.g - self.k_drag * speed * vz
        return [vx, vz, ax, az, speed]
 
    def _landing_event(t, state):
        return state[1]  # z crosses zero
    _landing_event.terminal = True
    _landing_event.direction = -1
    _landing_event = staticmethod(_landing_event) 
    
    
    def solve(self):
        """Integrate one deterministic trajectory (no droplet jitter).
        Returns the raw solve_ivp result; sol.y = [x, z, vx, vz, s]."""
        vx0 = self.v0 * np.cos(self.theta)
        vz0 = self.v0 * np.sin(self.theta)
        y0 = [self.x0, self.z0, vx0, vz0, 0.0]
 
        self.sol = solve_ivp(
            self.eom,
            t_span=(0.0, self.t_total),
            y0=y0,
            method='RK45',
            t_eval=np.arange(0.0, self.t_total, self.dt),
            events=self._landing_event,
            rtol=1e-9, atol=1e-11,
        )
        return self.sol

    # ── spray of N discrete droplets ─────────────────────────────────────
    def solve_spray(self):
        """
        All droplets share the identical coherent-jet trajectory up to
        breakup_length (still one body of water there) -- reuses solve()'s
        result instead of re-integrating that phase from scratch. At
        breakup each droplet peels off with its own small random
        angle/speed jitter (turbulence, magnitude set by self.jitter_angle
        and self.jitter_speed_frac) and is integrated independently under
        gravity + drag. Returns a list of self.n_droplets trajectories,
        each a list of {t, x, z, regime} dicts spanning t=0 -> landing
        (or t_total).
        """
        if self.sol is None:
            self.solve()
        rng = random.Random(self.seed)
 
        # read solution from th streamline, pick the one close to breakup length
        t, x, z, vx, vz, s = self.sol.t, *self.sol.y
        idx = min(int(np.searchsorted(s, self.breakup_length)), len(t) - 1)
        t_split = t[idx]
        x_s, z_s, vx_s, vz_s, s_s = x[idx], z[idx], vx[idx], vz[idx], s[idx]
        shared_traj = [
            {"t": t[i], "x": x[i], "z": z[i], "regime": "jet"} for i in range(idx + 1)
        ]
 
        # phase 2: each droplet peels off independently -- reuse self.eom
        # directly (s_s is already >= breakup_length, so it always takes
        # the drag branch; no separate drag-only equations needed).
        all_droplets = []
        for _ in range(self.n_droplets):
            dtheta = rng.uniform(-self.jitter_angle, self.jitter_angle)
            dspeed = 1.0 + rng.uniform(-self.jitter_speed_frac, self.jitter_speed_frac)
            speed0 = np.hypot(vx_s, vz_s) * dspeed
            base_angle = np.arctan2(vz_s, vx_s) + dtheta
            dvx0 = speed0 * np.cos(base_angle)
            dvz0 = speed0 * np.sin(base_angle)
 
            drop_sol = solve_ivp(
                self.eom, t_span=(t_split, self.t_total),
                y0=[x_s, z_s, dvx0, dvz0, s_s], method='RK45',
                t_eval=np.arange(t_split, self.t_total, self.dt),
                events=self._landing_event, rtol=1e-9, atol=1e-11,
            )
 
            drop_traj = list(shared_traj) + [
                {"t": drop_sol.t[i], "x": drop_sol.y[0][i], "z": drop_sol.y[1][i], "regime": "droplet"}
                for i in range(len(drop_sol.t))
            ]
            all_droplets.append(drop_traj)
 
        self._spray = all_droplets
        return all_droplets


if __name__ == "__main__":
    jet = FountainJet(
        pump_pressure=6.0e4 * pressure_UNIT,
        cross_section_diameter=0.006 * length_UNIT,
        angle=80.0 * u.deg,
        discharge_coeff=0.9,
    )

    v0 = jet.exit_velocity()
    L = jet.compute_breakup_length()
    print(f"Exit velocity v0     = {v0:.3f} m/s")
    print(f"Cross-section area          = {jet.cross_section_area()*1e6:.2f} mm^2")
    print(f"Flow rate Q          = {jet.flow_rate()*1000:.3f} L/s")
    print(f"Breakup length       = {L*1000:.2f} mm")

    sol = jet.solve()
    t, x, z, s = sol.t, sol.y[0], sol.y[1], sol.y[4]
    print(f"\n{'t (s)':>8} {'x (m)':>10} {'z (m)':>10} {'regime':>10}")
    for i in range(0, len(t), 100):
        regime = "jet" if s[i] < jet.breakup_length else "droplet"
        print(f"{t[i]:8.3f} {x[i]:10.4f} {z[i]:10.4f} {regime:>10}")

    x_final = np.interp(0.5, t, x)
    z_final = np.interp(0.5, t, z)
    print(f"\nPosition at t=0.5s: x={x_final:.4f} m, z={z_final:.4f} m")
    print("done")