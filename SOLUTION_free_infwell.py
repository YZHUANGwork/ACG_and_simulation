import numpy as np
import astropy.units as u
import astropy.constants as const

# ── DIMENSIONS ────────────────────────────────────────────────────────────

len_UNIT    = u.nm
time_UNIT   = u.fs
energy_UNIT = u.eV
mass_UNIT   = u.eV
angle_UNIT  = u.rad
k_UNIT      = angle_UNIT/len_UNIT
omega_UNIT  = angle_UNIT/time_UNIT


class SchrodingerBase:
    def __init__(self, convert=const.hbar * const.c,
                 m=const.m_e * const.c ** 2,
                 x=np.linspace(-5, 5, 2000) * u.nm,
                 t=0 * u.fs):
        """
        Parameters
        ----------
        m       : Quantity  particle rest energy m*c^2 [energy]
        x       : Quantity  spatial grid [length]
        t       : Quantity  the time at which phi(x, t) is evaluated
                  [time]; default 0
        """
        self.convert = convert.to(u.MeV * u.fm)
        self.hbar = const.hbar

        self.m = m.to(mass_UNIT)
        self.x = x.to(len_UNIT)
        self.t = t.to(time_UNIT)

    # ── wavenumber from energy: k = sqrt(2m(E-V))/hbar ────────────────────
    def energy_to_k(self, E, V):
        """
        k = sqrt(2 m (E - V)) / hbar, complex. Real when E>=V (propagating).

        Returns
        -------
        k : Quantity (complex), k_UNIT
        """
        E = E.to(energy_UNIT)
        V = V.to(energy_UNIT)
        dE = (E - V).to(energy_UNIT)

        p = np.sqrt(2 * self.m * dE + 0j * energy_UNIT ** 2)   # complex sqrt, still a Quantity throughout
        k = (p / self.convert).to(k_UNIT, equivalencies=u.dimensionless_angles())   # k = p / hbar

        self.k = k
        return k

    # ── inverse: energy from wavenumber, E = hbar^2 k^2/(2m) + V ──────────
    def k_to_energy(self, k, V):
        """
        E = (hbar k)^2 / (2m) + V, the inverse of energy_to_k -- given a
        wavenumber fixed by a boundary condition, returns the
        corresponding energy.

        Returns
        -------
        E : Quantity, energy_UNIT
        """
        V = V.to(energy_UNIT)
        p = (k * self.convert / angle_UNIT).to(energy_UNIT, equivalencies=u.dimensionless_angles())   # p = hbar k
        return (p ** 2 / (2 * self.m) + V).to(energy_UNIT)

    @staticmethod
    def _time_factor(E, t):
        """exp(-i*omega*t), with omega = E/hbar."""
        omega = (E / const.hbar).to(omega_UNIT, equivalencies=u.dimensionless_angles())
        wt = (omega * t).to_value(angle_UNIT)
        return np.exp(-1j * wt)


class PlaneWave(SchrodingerBase):
    """
    Free particle (V=0 everywhere), the general solution of
        -hbar^2/(2m) psi''(x) = E psi(x)
    is a combination of the two independent solutions e^{ikx}
    (right-moving) and e^{-ikx} (left-moving):
        psi(x)   = A * e^{ikx} + B * e^{-ikx}
        phi(x,t) = psi(x) * e^{-i E t/hbar}
    No boundary condition constrains this problem, so k (and E) stay
    free, continuous inputs -- nothing here gets quantized. Note this
    solution is never normalizable (|psi|^2 is constant over all space),
    unlike InfiniteWell's bound eigenstates.
    """

    def __init__(self, convert=const.hbar * const.c,
                 m=const.m_e * const.c ** 2,
                 x=np.linspace(-5, 5, 2000) * u.nm,
                 t=0 * u.fs,
                 E=1 * u.eV,
                 A=1.0, B=0.0):
        """
        Parameters
        ----------
        E    : Quantity  particle energy [energy]
        A, B : complex   coefficients of e^{ikx} (right-moving) and
               e^{-ikx} (left-moving). Default A=1, B=0: a single
               rightward wave.
        """
        super().__init__(convert=convert, m=m, x=x, t=t)
        self.E = E.to(energy_UNIT)
        self.A = A
        self.B = B

        self.k = None
        self.energy_to_k(self.E, V=0 * u.eV)

    def solve(self, t=None):
        """
        phi(x,t) = [A e^{ikx} + B e^{-ikx}] * e^{-iEt/hbar}, evaluated at
        one time t (defaults to self.t, i.e. t=0 unless overridden).

        Returns
        -------
        phi : ndarray (complex), shape (len(x),)
        """
        t = self.t if t is None else t.to(time_UNIT)

        k = self.energy_to_k(self.E, V=0 * u.eV)
        kv = k.to_value(k_UNIT)
        xv = self.x.to_value(len_UNIT)
        psi = self.A * np.exp(1j * kv * xv) + self.B * np.exp(-1j * kv * xv)
        phi = psi * self._time_factor(self.E, t)

        self.phi = phi
        return phi


class InfiniteWell(SchrodingerBase):
    """
        V=0 for 0<=x<=L,
        V=inf elsewhere,
        
        solution form inside psi(x)   = A * e^{ikx} + B * e^{-ikx}
        Boundary condition psi(0) = 0, psi(L)=0,
        
        two free param, two BC, solution is all set 
        
        E_n = n^2 pi^2 hbar^2 / (2 m L^2),
        psi_n(x) = sqrt(2/L) sin(n pi x / L)
        for n = 1 .. n_max.
 
        Returns
        -------
        psi_sum  : ndarray (real), shape (len(x),)  equal-weight sum of
                   the n_max eigenstates
        energies : Quantity (n_max,)  E_n
        states   : ndarray (n_max, len(x))  psi_n(x), zero outside [0, L]
        """

    def __init__(self, convert=const.hbar * const.c,
                 m=const.m_e * const.c ** 2,
                 x=np.linspace(-0.5, 1.5, 2000) * u.nm,
                 t=0 * u.fs,
                 L=1 * u.nm,
                 n_max=5):
        """
        Parameters
        ----------
        L     : Quantity  well width [length]
        n_max : int       highest quantum number to find
        """
        super().__init__(convert=convert, m=m, x=x, t=t)
        self.L = L.to(len_UNIT)
        self.n_max = n_max

    def solve(self, t=None, n_max=None):
        
        t = self.t if t is None else t.to(time_UNIT)
        n_max = self.n_max if n_max is None else n_max

        n = np.arange(1, n_max + 1)
        L = self.L

        # boundary condition psi(0)=psi(L)=0 fixes k_n = n*pi/L; V=0
        # inside the well, so E_n follows from the same E<->k relation
        # used by PlaneWave (energy_to_k's inverse)
        k_n = n * np.pi * angle_UNIT / L
        energies = self.k_to_energy(k_n, V=0 * u.eV)

        xv = self.x.to_value(len_UNIT)
        Lv = L.to_value(len_UNIT)
        inside = (xv >= 0) & (xv <= Lv)
        norm = np.sqrt(2 / Lv)

        states = np.zeros((n_max, len(xv)))
        for i, ni in enumerate(n):
            psi_n = norm * np.sin(ni * np.pi * xv / Lv)
            psi_n[~inside] = 0.0
            states[i] = psi_n

        # equal-weight superposition, normalized to total probability 1:
        # each orthonormal state gets coefficient 1/sqrt(n_max)
        phi_sum = np.zeros(len(xv), dtype=complex)
        for i in range(n_max):
            phi_sum += (states[i] / np.sqrt(n_max)) * self._time_factor(energies[i], t)

        self.energies = energies
        self.states = states
        return phi_sum, energies, states


if __name__ == "__main__":
    x = np.linspace(-5, 5, 2000) * u.nm

    pw = PlaneWave(x=x, t=0 * u.fs, E=1 * u.eV)
    phi = pw.solve(t=0 * u.fs)
    print(f"PlaneWave: phi(x,t=0) shape {phi.shape}, |phi| mean {np.abs(phi).mean():.3f}")

    # E is irrelevant for InfiniteWell -- but there's no E parameter to
    # even pass anymore, since it genuinely doesn't apply to this class
    iw = InfiniteWell(x=np.linspace(-0.5, 1.5, 2000) * u.nm, t=0 * u.fs, L=1 * u.nm, n_max=5)
    phi_sum, energies, states = iw.solve(t=0 * u.fs)
    print(f"InfiniteWell energies: {energies}")

    xv = iw.x.to_value(u.nm)
    total_prob = np.trapezoid(np.abs(phi_sum) ** 2, xv)
    print(f"InfiniteWell |phi_sum|^2 integral: {total_prob:.4f} (should be 1)")