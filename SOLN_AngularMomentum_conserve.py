
import numpy as np
import astropy.units as u
import astropy.constants as const

len_UNIT = u.m
time_UNIT = u.s
mass_UNIT = u.kg
angle_UNIT = u.rad
freq_UNIT = u.Hz
vel_UNIT = len_UNIT / time_UNIT
accel_UNIT = len_UNIT / time_UNIT ** 2
omega_UNIT = angle_UNIT / time_UNIT


class RotatingStick:
    """L = 1/2 I omega**2 +1/2 mv**2 - mgh
    I = (1/12) m l^2 
    """
    def __init__(self,
                 m=0.2 * u.kg, l=0.6 * u.m,                     
                 x0=0.0 * u.m, y0=0.0 * u.m,                      
                 vx0=0.0 * u.m / u.s, vy0=4.0 * u.m / u.s,        
                 theta0=0.0 * u.rad, omega0=15.0 * u.rad / u.s,   
                 g=None):
 
        self.m = m.to_value(mass_UNIT)
        self.l = l.to_value(len_UNIT)
        self.I = (1 / 12) * self.m * self.l ** 2
 
        self.x0 = x0.to_value(len_UNIT)
        self.y0 = y0.to_value(len_UNIT)
        self.vx0 = vx0.to_value(vel_UNIT)
        self.vy0 = vy0.to_value(vel_UNIT)
 
        self.theta0 = theta0.to_value(angle_UNIT)
        self.omega0 = omega0.to_value(omega_UNIT)
 
        self.g = (g.to_value(accel_UNIT) if g is not None
                  else const.g0.to_value(accel_UNIT))
 
    def trajectory(self, t):
        """x_cm(t), y_cm(t), theta(t) """
        tt = t.to_value(time_UNIT)
        x_cm = self.x0 + self.vx0 * tt
        y_cm = self.y0 + self.vy0 * tt - 0.5 * self.g * tt ** 2
        theta = self.theta0 + self.omega0 * tt
        return x_cm, y_cm, theta



class SpinningStick:
    """
    dL/dt = 0, L = I omega, I_i * ometa_i = I_f * ometa_f
    """

    def __init__(self,
                 m=60.0 * u.kg,                                  
                 l_initial=1. * u.m, l_final=0.5 * u.m,           
                 omega_initial=np.pi/2 * u.rad / u.s,                  
                 theta0=0.0 * u.rad):                              
 
        self.m = m.to_value(mass_UNIT)
        self.l_initial = l_initial.to_value(len_UNIT)
        self.l_final = l_final.to_value(len_UNIT)
        self.omega_initial = omega_initial.to_value(omega_UNIT)
        self.theta0 = theta0.to_value(angle_UNIT)
 
        self.I_initial = self.I(self.l_initial)
        self.I_final = self.I(self.l_final)
 
        self.omega_final = self.I_initial * self.omega_initial / self.I_final
 
    def I(self, l):
        return (1 / 12) * self.m * l ** 2
 
    def theta_initial(self, t):
        tt = t.to_value(time_UNIT)
        return self.theta0 + self.omega_initial * tt
 
    def theta_final(self, t):
        tt = t.to_value(time_UNIT)
        return self.theta0 + self.omega_final * tt