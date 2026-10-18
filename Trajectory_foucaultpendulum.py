import numpy as np
from scipy.integrate import solve_ivp
import astropy.units as u
import astropy.constants as const
# ── DIMENSIONS ────────────────────────────────────────────────────────────
length_UNIT = u.m
time_UNIT   = u.s
speed_UNIT  = length_UNIT / time_UNIT
acc_UNIT    = speed_UNIT / time_UNIT

angle_UNIT  = u.rad
ang_v_UNIT  = angle_UNIT / time_UNIT


class foucault_pendulum:
    """
    https://farside.ph.utexas.edu/teaching/336k/Newton/node59.html
    """

    def __init__(self,
                 l=50.0*u.m,                      
                 g=const.g0,                       
                 OMEGA_earth=7.2921159e-5*u.rad/u.s,   # Earth's sidereal rotation rate
                 latitude=30*u.deg,                 # + = N hemisphere, - = S hemisphere

                 x0_prime=2.0*u.m,                    # initial horizontal displacement
                 y0_prime=0.0*u.m,
                 vx0_prime=0.0*u.m/u.s,                # initial horizontal velocity
                 vy0_prime=0.0*u.m/u.s,

                 t_total=6*u.hr,
                 dt=600.0*u.s):

        self.latitude = latitude.to(angle_UNIT).value

        self.l = l.to(length_UNIT).value
        self.g = g.to(acc_UNIT).value
        self.omega0 = np.sqrt(self.g / self.l)        # pendulum's own oscillation frequency, sqrt(g/l)

        self.OMEGA_earth = OMEGA_earth.to(ang_v_UNIT).value
        self.Omega_z = self.OMEGA_earth * np.sin(self.latitude)   # Omega*sin(lat): the Coriolis term / precession rate

        self.x0 = x0_prime.to(length_UNIT).value
        self.y0 = y0_prime.to(length_UNIT).value
        self.vx0 = vx0_prime.to(speed_UNIT).value
        self.vy0 = vy0_prime.to(speed_UNIT).value

        self.t_total = t_total.to(time_UNIT).value
        self.dt = dt.to(time_UNIT).value
        self.initial_state = [self.x0, self.y0, self.vx0, self.vy0]

    def eom(self, t, state):  # local frame; x, y are REAL horizontal displacement, in meters
        x, y, vx, vy = state

        x_dd =  -(self.g / self.l) * x + 2.0 * self.Omega_z * vy
        y_dd =  -(self.g / self.l) * y - 2.0 * self.Omega_z * vx

        return [vx, vy, x_dd, y_dd]

    def solve(self):
        return solve_ivp(
            self.eom,
            t_span=(0.0, self.t_total),
            y0=self.initial_state,
            method='RK45',
            t_eval=np.arange(0.0, self.t_total, self.dt),
            rtol=1e-9, atol=1e-11,
        )

    def solve_analytic(self, t_eval=None):
        """
        Closed-form solution (Eqs. 444-452 in the source): with
        s = x + i*y, the EOM becomes s_dd + 2i*Omega_z*s_dot + (g/l)*s = 0,
        solved by s(t) = e^{-i*Omega_z*t} * [s(0)*cos(omega_p*t)
                                              + (s_dot(0)/omega_p)*sin(omega_p*t)]
        with omega_p = sqrt(g/l + Omega_z^2) ~ sqrt(g/l) for Omega_z << omega0.
        No integration needed -- near-instant for any t_total/dt.
        """
        if t_eval is None:
            t_eval = np.arange(0.0, self.t_total, self.dt)

        omega_p = np.sqrt(self.g/self.l + self.Omega_z**2)

        s0 = self.x0 + 1j*self.y0
        s0_dot = self.vx0 + 1j*self.vy0

        s = np.exp(-1j*self.Omega_z*t_eval) * (
            s0*np.cos(omega_p*t_eval) + (s0_dot/omega_p)*np.sin(omega_p*t_eval)
        )

        return {'t': t_eval, 'x': s.real, 'y': s.imag}

