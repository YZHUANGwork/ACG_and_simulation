"""
a magnetized pin (mass m, length l0) is pivoted at 
center of mass, free to rotate in a fixed uniform external field B. The
pin carries a magnetic dipole moment mu along its own axis.
 
Moment of inertia of the pin about its own center (thin rod):
I = (1/12) m l0^2

polar coordinate,
1/2 I omega**2
omega = d theta / dt 

L = T - U = (1/2) I thetadot^2 - (-mu . B)
          = (1/2) I thetadot^2 + |mu| * |B| cos(theta_muB)


mu_vec = mu * (cos theta,   sin theta)
B_vec  = B  * (cos theta_B, sin theta_B)

U = -mu_vec . B_vec = -|mu| * |B| * cos(theta - theta_B)
L = (1/2) I thetadot^2 + |mu| * |B| * cos(theta - theta_B)

el eqn, # d/dt(del L/delthetadot) - del L/deltheta = Q_theta

I*thetaddot - (-|mu| * |B|*sin(theta-theta_B)) = -b*thetadot
I*thetaddot + b*thetadot + |mu| * |B|*sin(theta-theta_B) = 0

          
              
Kapitza pendulum: pendulum of length l, pivot driven vertically as
y(t) = a*cos(w*t). phi measured from the downward vertical.
x = l sin(phi), y = y_pivot(t) - l cos(phi)
L = 1/2 (xdot^2 + ydot^2) - g*y
  = 1/2 [ l^2 phidot^2 + ydot_pivot^2 + 2 l ydot_pivot phidot sin(phi) ]
    - g*y_pivot(t) + g*l*cos(phi)
EL: d/dt(del L/del phidot) - del L/del phi = 0
  l*phiddot + (g + yddot_pivot(t))*sin(phi) = 0
with y_pivot(t) = a*cos(w*t)  =>  yddot_pivot(t) = -a*w^2*cos(w*t)
  phiddot = -(g - a*w^2*cos(w*t)) * sin(phi) / l
"""
import numpy as np
from scipy.integrate import solve_ivp
import astropy.units as u
import astropy.constants as const
import matplotlib.pyplot as plt

 
# ── DIMENSIONS ────────────────────────────────────────────────────────────
length_UNIT = u.m
mass_UNIT   = u.kg
time_UNIT   = u.s
angle_UNIT  = u.rad
omega_UNIT  = angle_UNIT / time_UNIT
inertia_UNIT = mass_UNIT * length_UNIT**2
 
moment_UNIT = u.A * u.m**2              # mu: magnetic dipole moment
field_UNIT  = u.T                        # B: magnetic field
torque_UNIT = u.N * u.m                  # mu*B, I*thetaddot both reduce to this
damping_UNIT = u.N * u.m * u.s           # b*thetadot must match I*thetaddot [N*m]

class MagPin:
    """Rigid magnetized pin, pivoted at center, relaxing into alignment
    with a fixed external field B under damping. theta is the pin's own
    polar angle; theta_B is the field's direction, a separate constant."""
 
    def __init__(self,
                 mu=0.582*moment_UNIT,                  # sourced: real compass-needle problem
                 B=3.3e-3*field_UNIT,                     # sourced: Earth field, same problem
                 l0=0.03*length_UNIT, M=0.001*mass_UNIT,   # typical needle size/mass, not from a citation
                 zeta=0.05,                                 # small -> allows several loops before capture
                 theta0=0.0*angle_UNIT, omega0=900.0*omega_UNIT,
                 theta_B=0.0*angle_UNIT,                     # field direction, separate from theta's own definition
                 t_total=0.5*time_UNIT, dt=0.005*time_UNIT):
 
        self.l0  = l0.to(length_UNIT).value
        self.M   = M.to(mass_UNIT).value
        self.mu  = mu.to(moment_UNIT).value
        self.B   = B.to(field_UNIT).value
        self.muB = (mu * B).to(torque_UNIT).value
        self.I   = (1/12) * self.M * self.l0**2
        wn = np.sqrt(self.muB / self.I)
        self.b   = zeta * 2 * self.I * wn                 # b derived from I, so it's always consistent
 
        self.theta0 = theta0.to(angle_UNIT).value
        self.omega0 = omega0.to(omega_UNIT).value
        self.theta_B = theta_B.to(angle_UNIT).value
 
        self.t_total = t_total.to(time_UNIT).value
        self.dt      = dt.to(time_UNIT).value
 
 
    def eom(self, t, state):
        theta, omega = state
        I, muB, b, theta_B = self.I, self.muB, self.b, self.theta_B
 
        # d/dt(del L/delthetadot) - del L/deltheta = Q_theta
        Q_theta = -b * omega
        rhs_theta = -muB * np.sin(theta - theta_B) + Q_theta
        thetaddot = rhs_theta / I
        return [omega, thetaddot]
 
    def solve(self):
        return solve_ivp(
            self.eom,
            t_span=(0.0, self.t_total),
            y0=[self.theta0, self.omega0],
            method='RK45',
            t_eval=np.arange(0.0, self.t_total, self.dt),
            rtol=1e-9, atol=1e-11,
        )
 
accel_UNIT  = length_UNIT / time_UNIT**2
drive_UNIT  = angle_UNIT / time_UNIT      # omega_drive 
class KapitzaPendulum:
    """Rigid pendulum (point mass on a massless rod), xz plane, pivot
    height vibrating vertically: z_pivot(t) = A*cos(omega_drive*t).
    phi = angle of the rod from the positive z-axis"""
 
    def __init__(self,
                 l=0.1*length_UNIT,
                 A=0.005*length_UNIT,                        # pivot vibration amplitude
                 omega_drive=400*drive_UNIT,                # pivot vibration angular frequency
                 phi0=1.0*u.deg, phidot0=0.0*omega_UNIT,  # start near inverted (phi=0 = up)
                 t_total=0.47*time_UNIT, dt=0.005*time_UNIT):
 #A · ω_drive > √(2·g·l) OTERWISE THE PENDULUM WILL DROP
        self.g = const.g0.to(accel_UNIT).value
        self.l = l.to(length_UNIT).value
        self.A = A.to(length_UNIT).value
        self.omega_drive = omega_drive.to(drive_UNIT).value
 
        self.phi0 = phi0.to(angle_UNIT).value
        self.phidot0 = phidot0.to(omega_UNIT).value
 
        self.t_total = t_total.to(time_UNIT).value
        self.dt = dt.to(time_UNIT).value
 
    def z_pivot(self, t):
        return self.A * np.cos(self.omega_drive * t)
 
    def eom(self, t, state):
        phi, phidot = state
        g, l, A, omega_drive = self.g, self.l, self.A, self.omega_drive
 
        zpivotddot = -A * omega_drive**2 * np.cos(omega_drive * t)
 
        # l*phiddot = (g + zpivotddot)*sin(phi)  ->  isolate phiddot:
        phiddot = (g + zpivotddot) * np.sin(phi) / l
        return [phidot, phiddot]
 
    def solve(self):
        return solve_ivp(
            self.eom,
            t_span=(0.0, self.t_total),
            y0=[self.phi0, self.phidot0],
            method='RK45',
            t_eval=np.arange(0.0, self.t_total, self.dt),
            rtol=1e-10, atol=1e-12,
        )
''' 
class KapitzaPendulum:
    def __init__(self, l=0.5, a=0.05, w=80.0, g=9.81,
                 phi0=170*np.pi/180, phidot0=0.0):
        self.l, self.a, self.w, self.g = l, a, w, g
        self.phi0, self.phidot0 = phi0, phidot0
 
    def eom(self, t, state):
        phi, phidot = state
        yddot_pivot = -self.a * self.w**2 * np.cos(self.w * t)
        phiddot = -(self.g + yddot_pivot) * np.sin(phi) / self.l
        return [phidot, phiddot]
 
    def solve(self, t_total=5.0, dt=0.001):
        t_eval = np.arange(0, t_total, dt)
        return solve_ivp(self.eom, (0, t_total), [self.phi0, self.phidot0],
                          t_eval=t_eval, method='RK45', rtol=1e-9, atol=1e-11)
''' 
if __name__ == "__main__":
    pin = MagPin()
    sol = pin.solve()
    theta, omega = sol.y[0], sol.y[1]

    print(f"I = {pin.I:.3e} kg*m^2")
    print(f"theta: {np.degrees(pin.theta0):.0f} -> {np.degrees(theta[-1]):.1f} deg")

    plot_results(pin, sol)
    animate(pin, sol)
    print("done")