import numpy as np
import astropy.units as u
import astropy.constants as const

len_UNIT = u.m
time_UNIT = u.s
mass_UNIT = u.kg
vel_UNIT = len_UNIT / time_UNIT


class HarmonicOscillator:
    """
   input initial q and q dot
    """
 
    def __init__(self, k, m, orientation="horizontal", g=None, q0=0.0 * u.m, v0=0.0 * vel_UNIT):
        self.k = k.to_value(mass_UNIT / time_UNIT ** 2)
        if isinstance(m, (list, tuple)):          # coupled_trajectory case: m = [m1, m2]
            self.m = [m[0].to_value(mass_UNIT), m[1].to_value(mass_UNIT)]
        else:                                      # single-mass case: trajectory_1d, ring_trajectory
            self.m = m.to_value(mass_UNIT)
            self.w = np.sqrt(self.k / self.m)
        self.g = g.to_value(len_UNIT / time_UNIT ** 2) if g is not None else const.g0.to_value(len_UNIT / time_UNIT ** 2)
 
        self.orientation = orientation
        if isinstance(q0, (list, tuple)):          # coupled_trajectory case: q0 = [x1_0, x2_0]
            self.q0 = [q0[0].to_value(len_UNIT), q0[1].to_value(len_UNIT)]
        else:                                       # single-mass case
            self.q0 = q0.to_value(len_UNIT)
        if isinstance(v0, (list, tuple)):          # coupled_trajectory case: v0 = [v1_0, v2_0]
            self.v0 = [v0[0].to_value(vel_UNIT), v0[1].to_value(vel_UNIT)]
        else:                                       # single-mass case
            self.v0 = v0.to_value(vel_UNIT)
        
    def trajectory_1d(self, t):
        """
        q is x or y
        pivot at height h, spring natural length l, so U = m*g*(h-l+q) + 1/2*k*q^2)
            L = T-U = 1/2*m*qdot^2 - m*g*(h-l+q) - 1/2*k*q^2
 
        Euler-Lagrange:
            vertical q_ddot = -(k/m)*q - g    
            horizontal q_ddot = -(k/m)*q
            
            x(t) = x0 *cos(w*t) + (vx0/w)*sin(w*t)
            
            y(t) = -m*g/k + (y0 - (-m*g/k))*cos(w*t) + (vy0/w)*sin(w*t)
            
            
        """
        t = t.to_value(time_UNIT)
        if self.orientation == "vertical":
            q = -self.m * self.g / self.k \
                + (self.q0 - (-self.m * self.g / self.k)) * np.cos(self.w * t) \
                + (self.v0 / self.w) * np.sin(self.w * t)
        elif self.orientation == "horizontal":
            q = self.q0 * np.cos(self.w * t) + (self.v0 / self.w) * np.sin(self.w * t)
        return q
 
    def coupled_trajectory(self, t):
        """
        L = 1/2*m1*x1dot^2 + 1/2*m2*x2dot^2 - 1/2*k*(x2-x1)^2
 
        analytic solution
        """
        m1, m2 = self.m[0], self.m[1]
        x1_0, x2_0 = self.q0[0], self.q0[1]
        v1_0, v2_0 = self.v0[0], self.v0[1]
        t = np.atleast_1d(t.to_value(time_UNIT))
 
        # [x1_ddot, x2_ddot] = K @ [x1, x2], straight from the two EOMs
        K = np.array([[-self.k / m1, self.k / m1],
                      [self.k / m2, -self.k / m2]])
 
        w = np.sqrt(-(K[0, 0] + K[1, 1]))
 
        s = (m1 * x1_0 + m2 * x2_0) / (m1 + m2) + (m1 * v1_0 + m2 * v2_0) / (m1 + m2) * t
        d = (x2_0 - x1_0) * np.cos(w * t) + (v2_0 - v1_0) / w * np.sin(w * t)
 
        x1 = s - (m2 / (m1 + m2)) * d
        x2 = s + (m1 / (m1 + m2)) * d
        return x1, x2
   