"""
Setup: a rigid stick (mass m, length l_cyl) hangs from a MASSLESS string
of length l(t) (prescribed, growing in time). The string is tied to the
stick off-center, a distance d_tie from one end -- so the stick's center
of mass sits a distance r = l_cyl/2 - d_tie from the tie point, and the
stick is free to tilt on its own about the tie point, independent of the
string's swing. This is a double pendulum: string (angle theta, pivot
fixed) + stick (angle phi, pivot = the string's own moving endpoint).
 
Stick's moment of inertia about its own center of mass (thin rod):
    I_cm = (1/12) m l_cyl^2  ->  k2 = I_cm/m = l_cyl^2/12
 
Lagrangian (per unit mass, so m=1 throughout):
 
    L = (1/2) v_cm^2 + (1/2) k2 phidot^2 - U
 
where v_cm is the STICK'S CENTER OF MASS velocity -- not the tie point's.
Center of mass position (pivot at origin, l(t) prescribed):
 
    x_cm = l(t) sin(theta) + r sin(phi)
    z_cm = -l(t) cos(theta) - r cos(phi)
 
Differentiate (l, theta, phi all vary with t):
 
    vx_cm = ldot sin(theta) + l thetadot cos(theta) + r phidot cos(phi)
    vz_cm = -ldot cos(theta) + l thetadot sin(theta) + r phidot sin(phi)
 
v_cm^2 = vx_cm^2 + vz_cm^2 expands (via cos(a)cos(b)+sin(a)sin(b) type
identities) to:
 
    v_cm^2 = ldot^2 + l^2 thetadot^2 + r^2 phidot^2
             + 2 r ldot phidot sin(theta-phi) + 2 r l thetadot phidot cos(theta-phi)
 
U = -g l cos(theta) - g r cos(phi)  (height of tie point + COM offset)
 
Generalized coordinates: q = (theta, phi)
    theta -- string angle from vertical, at the fixed pivot
    phi   -- stick's own tilt from vertical, at the tie point
 
Euler-Lagrange, d/dt(dL/dqdot) - dL/dq = 0, gives the undamped EOM
(see eom() for the algebra, rearranged to isolate thetaddot/phiddot).
 
Extended EOM with friction (non-conservative generalized forces Q_i):
    d/dt(dL/dthetadot) - dL/dtheta = Q_theta = -b_pivot l^2 thetadot
    d/dt(dL/dphidot)   - dL/dphi   = Q_phi   = -b_wall (r^2+k2) phidot
"""
import numpy as np
from scipy.integrate import solve_ivp
import astropy.units as u
import astropy.constants as const

# ── DIMENSIONS ────────────────────────────────────────────────────────────
length_UNIT = u.m
mass_UNIT   = u.kg
time_UNIT   = u.s
angle_UNIT  = u.rad
accel_UNIT  = length_UNIT / time_UNIT**2
omega_UNIT  = angle_UNIT / time_UNIT
lrate_UNIT  = length_UNIT / time_UNIT     # d(length)/dt


class Pendulum:
    """String (length growing linearly) + stick tied off-center, tilts
    independently (theta = string angle, phi = stick's own tilt)."""

    def __init__(self,
                 l0=0.5*length_UNIT, l_final=3.0*length_UNIT, t_total=30.0*time_UNIT,
                 theta0=-90.0*u.deg, omega0=0.0*omega_UNIT,
                 l_cyl=1.5*length_UNIT, d_tie=0.3*length_UNIT,
                 phi0=0.0*angle_UNIT, phidot0=0.0*omega_UNIT,
                 b_pivot=0.2/time_UNIT, b_wall=0.2/time_UNIT,
                 z_pivot=5.0*length_UNIT,
                 dt=0.05*time_UNIT):
        self.g = const.g0.to(accel_UNIT).value
        self.l0 = l0.to(length_UNIT).value
        self.k  = ((l_final - l0) / t_total).to(lrate_UNIT).value
        self.t_total = t_total.to(time_UNIT).value
        self.dt = dt.to(time_UNIT).value

        self.theta0 = theta0.to(angle_UNIT).value
        self.omega0 = omega0.to(omega_UNIT).value
        self.phi0    = phi0.to(angle_UNIT).value
        self.phidot0 = phidot0.to(omega_UNIT).value

        self.l_cyl = l_cyl.to(length_UNIT).value
        self.r  = (l_cyl/2 - d_tie).to(length_UNIT).value       # tie point -> stick COM
        self.k2 = (l_cyl**2/12).to(length_UNIT**2).value        # I_cm / mass, thin-rod

        self.b_pivot = b_pivot.to(1/time_UNIT).value
        self.b_wall  = b_wall.to(1/time_UNIT).value
        self.z_pivot = z_pivot.to(length_UNIT).value

    def length(self, t):
        return self.l0 + self.k*t

    def eom(self, t, state):
        theta, thetadot, phi, phidot = state
        l = self.length(t)
        r, k2, k, g = self.r, self.k2, self.k, self.g
        s, c = np.sin(theta-phi), np.cos(theta-phi)

        # d/dt(del L/del thetadot) - del L/del theta = Q_theta  ->  isolate ddots:
        Q_theta = -self.b_pivot * l**2 * thetadot
        Q_phi   = -self.b_wall * (r**2 + k2) * phidot

        rhs_theta = -2.0*l*k*thetadot - r*k*c*phidot - r*l*s*phidot**2 - g*l*np.sin(theta) + Q_theta
        rhs_phi   = -2.0*r*k*c*thetadot + r*l*s*thetadot**2 - g*r*np.sin(phi) + Q_phi

        m11, m12 = l**2, r*l*c
        m21, m22 = r*l*c, r**2 + k2
        det = m11*m22 - m12*m21

        thetaddot = (rhs_theta*m22 - rhs_phi*m12) / det
        phiddot   = (rhs_phi*m11 - rhs_theta*m21) / det
        return [thetadot, thetaddot, phidot, phiddot]

    def solve(self):
        return solve_ivp(
            self.eom,
            t_span=(0.0, self.t_total),
            y0=[self.theta0, self.omega0, self.phi0, self.phidot0],
            method='RK45',
            t_eval=np.arange(0.0, self.t_total, self.dt),
            rtol=1e-9, atol=1e-11,
        )


class Drip:
    """A drop released from the stick tip with velocity (vx, vz), sliding
    on the wall under gravity and its own (fluid) friction b_drip."""

    def __init__(self, b_drip=20.0/time_UNIT, dt_spawn=1.*time_UNIT,
                 drip_radius=0.1*length_UNIT, radius_decay=0.9/time_UNIT,
                 dt=0.02*time_UNIT):
        self.g = const.g0.to(accel_UNIT).value
        self.b_drip = b_drip.to(1/time_UNIT).value
        self.dt_spawn = dt_spawn.to(time_UNIT).value
        self.drip_radius = drip_radius.to(length_UNIT).value
        self.radius_decay = radius_decay.to(1/time_UNIT).value
        self.dt = dt.to(time_UNIT).value

    def radius_at(self, elapsed):
        return self.drip_radius * np.exp(-self.radius_decay * elapsed)

    def eom(self, t, state):
        x, vx, z, vz = state
        b = self.b_drip
        return [vx, -b*vx, vz, -self.g - b*vz]

    def solve(self, t_spawn, x0, z0, vx0, vz0, t_end):
        t_eval = np.arange(t_spawn, t_end, self.dt)
        t_eval = t_eval[(t_eval >= t_spawn) & (t_eval <= t_end)]
        return solve_ivp(
            self.eom,
            t_span=(t_spawn, t_end),
            y0=[x0, vx0, z0, vz0],
            method='RK45',
            t_eval=t_eval,
            rtol=1e-9, atol=1e-11,
        )


if __name__ == "__main__":
    pend = Pendulum()
    sol = pend.solve()
    theta, phi = sol.y[0], sol.y[2]
    print(f"theta: {pend.theta0*180/np.pi:.0f} -> {np.degrees(theta[-1]):.1f} deg")
    print(f"phi:   {pend.phi0*180/np.pi:.0f} -> {np.degrees(phi[-1]):.1f} deg")