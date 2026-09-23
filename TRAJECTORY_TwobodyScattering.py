import numpy as np
import astropy.units as u
import astropy.constants as const

len_UNIT = u.m
time_UNIT = u.s
mass_UNIT = u.kg
vel_UNIT = len_UNIT / time_UNIT
acc_UNIT = vel_UNIT / time_UNIT

class CollisionKineticEnergyBase:
    
    def __init__(self, m1, m2, v_i_1, v_i_2, v_f_1, v_f_2, rtol=1e-9):
        self.m1 = m1.to(mass_UNIT)
        self.m2 = m2.to(mass_UNIT)
        self.v_i_1 = v_i_1.to(vel_UNIT)
        self.v_i_2 = v_i_2.to(vel_UNIT)
        self.v_f_1 = v_f_1.to(vel_UNIT)
        self.v_f_2 = v_f_2.to(vel_UNIT)

        self.KE_initial = self._KE(self.v_i_1, self.v_i_2)
        self.KE_final = self._KE(self.v_f_1, self.v_f_2)
        self.is_elastic = bool(np.isclose(self.KE_initial.value, self.KE_final.value, rtol=rtol))

    def _KE(self, v1, v2):
        ke = 0.5*self.m1*np.dot(v1.value, v1.value)*vel_UNIT**2 + 0.5*self.m2*np.dot(v2.value, v2.value)*vel_UNIT**2
        return ke.to(u.J)


class TwoBodyCollision:
    """
    r_contact, v_i_1, v_i_2 : lab frame input param 
    
    solving v_f_1, v_f_2 using 
    
        momentum:     m1*v_f_1 + m2*v_f_2 = m1*v_i_1 + m2*v_i_2
        restitution:    -v_f_1 +    v_f_2 = e*(v_i_1 - v_i_2)
    solve_trajectory(params, t): BEFORE-collision path -- backward from r_contact using v_i (for t<0)
                                 AFTER-collision       --forward from r_contact using v_f (for t>=0)
    
    shared position:  r_contact at t=0
    """

    def __init__(self, m1=1*u.kg, m2=1*u.kg, r_contact=(0, 0)*u.m,
                 v_i_1=(1, 0)*u.m/u.s, v_i_2=(0, 0)*u.m/u.s, e=1.0):
        self.m1 = m1.to(mass_UNIT)
        self.m2 = m2.to(mass_UNIT)
        self.r_contact = r_contact.to(len_UNIT)
        self.v_i_1 = v_i_1.to(vel_UNIT)
        self.v_i_2 = v_i_2.to(vel_UNIT)
        self.e = e #coefficient of restitution,  e=1 elastic, e=0 inelastic

    def _solve_v_f(self, v_i_1, v_i_2):
        """solve for TWO UNKNOWNS  v_f_1, v_f_2.
                TWO EQUATIONS, 
                momentum:     m1*v_f_1 + m2*v_f_2 = m1*v_i_1 + m2*v_i_2
                restitution:    -v_f_1 +    v_f_2 = e*(v_i_1 - v_i_2)
        """
        A = np.array([[self.m1.value, self.m2.value],
                      [-1.0,          1.0        ]])
        b = np.stack([self.m1.value*v_i_1.value + self.m2.value*v_i_2.value,
                      self.e*(v_i_1.value - v_i_2.value)])   # (2 eqns, 2 xy-components)
        v_f_1v, v_f_2v = np.linalg.solve(A, b)
        return v_f_1v * vel_UNIT, v_f_2v * vel_UNIT

    def lab_frame(self):
        """Returns (r_contact, v_i_1, v_i_2, v_f_1, v_f_2) in the lab frame."""
        v_f_1, v_f_2 = self._solve_v_f(self.v_i_1, self.v_i_2)
        return self.r_contact, self.v_i_1, self.v_i_2, v_f_1, v_f_2

    def cm_frame(self):
        """
        Returns (r_contact, v_i_1, v_i_2, v_f_1, v_f_2) in the COM frame.
        r_contact_com is always the origin: both boxes are at r_contact at
        the moment of contact, so their mass-weighted average position
        (the COM) is r_contact too, regardless of m1, m2 -- shifting to the
        COM frame (subtracting the COM's own position) sends it to 0.
        """
        M = self.m1 + self.m2
        V_cm = (self.m1*self.v_i_1 + self.m2*self.v_i_2) / M

        v_i_1_com = self.v_i_1 - V_cm
        v_i_2_com = self.v_i_2 - V_cm
        v_f_1_com, v_f_2_com = self._solve_v_f(v_i_1_com, v_i_2_com)   # solved IN the COM frame

        r_contact_com = self.r_contact - self.r_contact   # = 0, see docstring above
        return r_contact_com, v_i_1_com, v_i_2_com, v_f_1_com, v_f_2_com

def eom(r_i, v_i, t, a=None):
    
    if a is None:
        a = (0, 0, 0) * acc_UNIT
    return (r_i + v_i*t + 0.5*a.to(acc_UNIT)*t**2).to(len_UNIT)
 
 
def _before_after_contact(v_i, v_f, r_contact, t, before, a=None):
    """One body's trajectory: eom backward from r_contact using v_i (t<0),
    eom forward from r_contact using v_f (t>=0) -- both pass through
    r_contact exactly at t=0, by construction."""
    r_before = eom(r_contact, v_i, t, a=a)
    r_after = eom(r_contact, v_f, t, a=a)
    return np.where(before[:, None], r_before.value, r_after.value) * len_UNIT
 
 
def solve_trajectory(params, t, a=None):
    """
    Turns r_contact, v_i_1, v_i_2, v_f_1, v_f_2 = params  into a trajectory. 
    
    t0 , r_contact ,
   
    """
    r_contact, v_i_1, v_i_2, v_f_1, v_f_2 = params
    t_col = t[:, None]   # one (x,y) pair per requested time, see eom
    before = t < 0*time_UNIT
 
    r1 = _before_after_contact(v_i_1, v_f_1, r_contact, t_col, before, a=a)
    r2 = _before_after_contact(v_i_2, v_f_2, r_contact, t_col, before, a=a)
    return r1, r2


if __name__ == "__main__":
    # 1D case (y=0) -- must match the pure-1D formulas exactly
    c1d = TwoBoxCollision2D(m1=1*u.kg, m2=3*u.kg, r_contact=(0, 0)*u.m,
                             v_i_1=(3, 0)*u.m/u.s, v_i_2=(0, 0)*u.m/u.s, e=1.0)
    _, v_i_1, v_i_2, v_f_1, v_f_2 = c1d.lab_frame()
    kinetic_energy = CollisionKineticEnergyBase(c1d.m1, c1d.m2, v_i_1, v_i_2, v_f_1, v_f_2)
    print("1D check (y=0), elastic v_f_1,v_f_2 =", v_f_1, v_f_2)
    print(f"  KE_i={kinetic_energy.KE_initial:.3f}, KE_f={kinetic_energy.KE_final:.3f} -> "
          f"{'ELASTIC' if kinetic_energy.is_elastic else 'INELASTIC'}")

    # genuine 2D elastic collision -- ANY v_i_1, v_i_2 works now, no
    # collision-course requirement, since r_contact is given directly
    c2d = TwoBoxCollision2D(m1=1*u.kg, m2=1*u.kg, r_contact=(1, 0.5)*u.m,
                             v_i_1=(2, 1)*u.m/u.s, v_i_2=(-1, 0)*u.m/u.s, e=1.0)
    lab_params = c2d.lab_frame()
    kinetic_energy = CollisionKineticEnergyBase(c2d.m1, c2d.m2, lab_params[1], lab_params[2], lab_params[3], lab_params[4])
    print("2D elastic v_f_1,v_f_2 =", lab_params[3], lab_params[4])
    print(f"  KE_i={kinetic_energy.KE_initial:.3f}, KE_f={kinetic_energy.KE_final:.3f} -> "
          f"{'ELASTIC' if kinetic_energy.is_elastic else 'INELASTIC'}")

    # 2D perfectly inelastic -- v_f_1 == v_f_2 == (m1*v_i_1+m2*v_i_2)/(m1+m2)
    c2d_in = TwoBoxCollision2D(m1=2*u.kg, m2=1*u.kg, r_contact=(1, 0.5)*u.m,
                                v_i_1=(2, 1)*u.m/u.s, v_i_2=(-1, 0)*u.m/u.s, e=0.0)
    _, v_i_1_in, v_i_2_in, v_f_1_in, v_f_2_in = c2d_in.lab_frame()
    kinetic_energy = CollisionKineticEnergyBase(c2d_in.m1, c2d_in.m2, v_i_1_in, v_i_2_in, v_f_1_in, v_f_2_in)
    print("2D inelastic v_f_1,v_f_2 =", v_f_1_in, v_f_2_in)
    print(f"  KE_i={kinetic_energy.KE_initial:.3f}, KE_f={kinetic_energy.KE_final:.3f} -> "
          f"{'ELASTIC' if kinetic_energy.is_elastic else 'INELASTIC'}")

    # caller picks the frame, then feeds its parameters to solve_trajectory;
    # t=0 is contact, so a symmetric window around it makes sense
    t = np.linspace(-2, 2, 5) * u.s
    r1_lab, r2_lab = c2d.solve_trajectory(lab_params, t)
    print("r1 lab, r2 lab at t=0 (must be equal -- that's r_contact):", r1_lab[2], r2_lab[2])

    com_params = c2d.cm_frame()
    r1_com, r2_com = c2d.solve_trajectory(com_params, t)
    print("r1 com, r2 com at t=0 (must both be the origin):", r1_com[2], r2_com[2])

    print("lab frame r1:\n", r1_lab)
    print("COM frame r1:\n", r1_com)