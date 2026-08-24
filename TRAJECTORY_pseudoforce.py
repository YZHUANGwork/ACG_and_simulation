
import numpy as np
from scipy.integrate import solve_ivp
import astropy.units as u
import astropy.constants as const
# ── DIMENSIONS ────────────────────────────────────────────────────────────
length_UNIT = u.m
time_UNIT   = u.s
speed_UNIT  = length_UNIT / time_UNIT
acc_UNIT = speed_UNIT/ time_UNIT

angle_UNIT  = u.rad
rate_UNIT   = 1 / time_UNIT          
ang_v_UNIT  = angle_UNIT / time_UNIT 
                                     


class pseudo_force:
    """
    local frame v_prime, r_prime, a_prime,  bending is measured
    global frame v, r, a has some omega ,
    project omega onto prime basis
    
    F_fic = - m * a_fic
    a_prime = a_real_prime +a_fic_prime = a_real_prime + (-2 Omega_prime cross v_prime) + ( - Omega_prime cross (Omega_prime cross r_prime))
    
   
    """

    def __init__(self,
                 a_real_prime = [5, 0, 0]*u.m/u.s/u.s,
                 OMEGA_z = 1*u.rad / u.s ,
                 latitude=45.0*u.deg,          # + = N hemisphere, - = S hemisphere
                      
                 v0_prime=[15.0, 0.0, 0.0]*u.m/u.s,
                 r0_prime=[0.0, 0.0, 0.0]*u.m ,
                 
                 t_total=10*u.s,
                 dt=0.1*u.s):

        self.latitude = latitude.to(angle_UNIT).value  
        self.theta_rad = np.pi/2 - self.latitude
        
        
        self.OMEGA_z = OMEGA_z.to(ang_v_UNIT).value
        
        
        self.omega_prime_vec = np.array([0.0, self.OMEGA_z*np.sin(self.theta_rad),self.OMEGA_z*np.cos(self.theta_rad)])
        
        self.v0_prime = v0_prime.to(speed_UNIT).value
        self.r0_prime = r0_prime.to(length_UNIT).value
        
        self.a_real_vec = a_real_prime.to(acc_UNIT).value  
        vx0, vy0, vz0 = self.v0_prime
        x0, y0, z0 = self.r0_prime
        self.t_total = t_total.to(time_UNIT).value
        self.dt = dt.to(time_UNIT).value
        self.initial_state = [x0, y0, z0, vx0, vy0, vz0]  

    def eom(self, t, state):#local frame
        r = np.array(state[:3])#local frame
        v = np.array(state[3:])#local frame
        a = self.a_real_vec  -2.0 * np.cross(self.omega_prime_vec, v) -np.cross(self.omega_prime_vec, np.cross(self.omega_prime_vec, r)) 
        return [v[0], v[1], v[2], a[0], a[1], a[2]]

    def solve(self):
        return solve_ivp(
            self.eom,
            t_span=(0.0, self.t_total),
            y0=self.initial_state,  
            method='RK45',
            t_eval=np.arange(0.0, self.t_total, self.dt),
            rtol=1e-9, atol=1e-11,
        )

if __name__ == "__main__":
    p = pseudo_force()
    sol = p.solve()
    x, y, z = sol.y[0], sol.y[1], sol.y[2]
    print(f"latitude: {p.latitude*180/np.pi:.0f} deg")
    print(f"end position: x={x[-1]:.1f} m, y={y[-1]:.1f} m, z={z[-1]:.1f} m")
    print("done")