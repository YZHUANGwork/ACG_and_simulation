"""
Electron trajectory in an  magnetic dipole field.

DIPOLE 
    B_r(r, theta)     = 2 C cos(theta) / r^3
    B_theta(r, theta) =   C sin(theta) / r^3
    B_phi             = 0

C_DIPOLE CONSTANT

"""

import numpy as np
from scipy.integrate import solve_ivp
import astropy.units as u
import astropy.constants as const

# ── UNITS (working system: everything internal is expressed in these) ─────────
length_UNIT = u.km
time_UNIT   = u.s
vel_UNIT    = length_UNIT / time_UNIT  
charge_UNIT = u.C
mass_UNIT   = u.GeV / const.c**2
B_UNIT      = u.T
energy_UNIT = u.GeV
angle_UNIT  = u.rad


class ElectronDipoleTrajectory:
    

    Q_E     = (-const.e.si).to(charge_UNIT).value       # C
    M_E     = const.m_e.value                            # kg
    C_LIGHT = const.c.to(vel_UNIT).value                  # km/s (matches vel_UNIT)

    def __init__(self,
                 C_DIPOLE=1.0e4 * B_UNIT * length_UNIT**3,   # dipole strength constant
                 theta0=15.0 * u.deg,          # launch polar angle (from +z)
                 phi0=0.0 * u.deg,             # launch azimuthal angle
                 r0_mag=7000.0 * length_UNIT,  # launch radius
                 beta=0.7,                      # v0/c (dimensionless)
                 pitch_angle=10.0 * u.deg,      # angle between v0 and B0
                 t_max=0.01 * time_UNIT,
                 n_eval=600):
        # convert every input into the working unit system, once
        self.C_DIPOLE = C_DIPOLE.to(B_UNIT * length_UNIT**3).value    # T . km^3
        self.theta0 = theta0.to(angle_UNIT).value
        self.phi0 = phi0.to(angle_UNIT).value
        self.r0_mag = r0_mag.to(length_UNIT).value                    # km
        self.beta = beta
        self.pitch_angle = pitch_angle.to(angle_UNIT).value
        self.t_max = t_max.to(time_UNIT).value                        # s
        self.n_eval = n_eval

        self.r0 = self.r0_mag * np.array([
            np.sin(self.theta0) * np.cos(self.phi0),
            np.sin(self.theta0) * np.sin(self.phi0),
            np.cos(self.theta0),
        ])                                                            # km

        B_at_r0 = self.get_B_cartesian(self.r0)
        self.b_hat = B_at_r0 / np.linalg.norm(B_at_r0)

        # any direction perpendicular to B works -- no preferred phase for gyration start
        perp_hat = np.cross(self.b_hat, [0.0, 0.0, 1.0])
        self.perp_hat = perp_hat / np.linalg.norm(perp_hat)

        speed0 = self.beta * self.C_LIGHT                             # km/s
        v_par  = speed0 * np.cos(self.pitch_angle)
        v_perp = speed0 * np.sin(self.pitch_angle)
        self.v0 = v_par * self.b_hat + v_perp * self.perp_hat         # km/s

        self.X0 = np.hstack((self.r0, self.v0))
        self.t_eval = np.linspace(0.0, self.t_max, self.n_eval)

        self.sol = None

    def dipole_B_spherical(self, r, theta):
        """Analytic dipole field components (Br, Btheta, Bphi) in spherical
        coordinates. r in km (length_UNIT), theta = polar angle from +z."""
        Br     = 2.0 * self.C_DIPOLE * np.cos(theta) / r**3
        Btheta =       self.C_DIPOLE * np.sin(theta) / r**3
        Bphi   = 0.0
        return Br, Btheta, Bphi

    @staticmethod
    def spherical_to_cartesian_rotation_matrix(theta, phi):
        """3x3 rotation matrix R such that B_cartesian = R @ [Br, Btheta, Bphi]."""
        st, ct = np.sin(theta), np.cos(theta)
        sp, cp = np.sin(phi), np.cos(phi)
        return np.array([
            [st * cp, ct * cp, -sp],
            [st * sp, ct * sp,  cp],
            [ct,      -st,       0.],
        ])

    def get_B_cartesian(self, position_km):
        """position_km: [x, y, z] in km -> B in Tesla, Cartesian components."""
        x, y, z = position_km
        r = np.sqrt(x * x + y * y + z * z)
        theta = np.arccos(z / r)
        phi = np.arctan2(y, x)
        Br, Btheta, Bphi = self.dipole_B_spherical(r, theta)
        R = self.spherical_to_cartesian_rotation_matrix(theta, phi)
        return R @ np.array([Br, Btheta, Bphi])

    def eom(self, t, X):
        """
        dv/dt = (q / (gamma m)) * (v x B)
        X = [x, y, z, vx, vy, vz]   (working units: km, km/s)
        """
        v = X[3:]
        speed = np.linalg.norm(v)
        gamma = 1.0 / np.sqrt(1.0 - (speed / self.C_LIGHT) ** 2)
        B = self.get_B_cartesian(X[:3])
        a = (self.Q_E / (gamma * self.M_E)) * np.cross(v, B)
        return np.hstack((v, a))

    def solve(self):
        """Integrate the electron trajectory and return the solve_ivp solution."""
        self.sol = solve_ivp(
            self.eom,
            t_span=(0.0, self.t_max),
            y0=self.X0,
            method='RK45',
            t_eval=self.t_eval,
            rtol=1e-8,
            atol=1e-10,
            max_step=2e-6,
        )
        return self.sol


if __name__ == "__main__":
    traj = ElectronDipoleTrajectory()
    sol = traj.solve()