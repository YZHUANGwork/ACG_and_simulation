import numpy as np
import astropy.units as u
import astropy.constants as const

len_UNIT = u.m
time_UNIT = u.s
vel_UNIT = len_UNIT / time_UNIT


class LorentzTransform:
    """
    input the boost velocity of the primed frame as a fraction of c:
        beta = [beta_x, beta_y, beta_z] = [vx/c, vy/c, vz/c]   (dimensionless)
    input the x^nu = (x^0, x^1, x^2, x^3), 
        x^0 = c*t   
        x^i = x, y, z 

    gamma = 1 / sqrt(1 - beta . beta)

    Λ^mu_nu:
        Λ^0_0   = gamma
        Λ^0_i   = -gamma * beta_i
        Λ^i_0   = -gamma * beta_i
        Λ^i_j   = delta_ij + (gamma - 1) * beta_i * beta_j / beta^2

    x'^mu = Λ^mu_nu x^nu
    """

    def __init__(self, beta):
        self.beta = np.asarray(beta, dtype=float)          # v_i/c, dimensionless
        if self.beta.shape != (3,):
            raise ValueError("beta must be length-3: [beta_x, beta_y, beta_z] = [vx/c, vy/c, vz/c]")
        self.beta2 = np.dot(self.beta, self.beta)
        if self.beta2 >= 1.0:
            raise ValueError("|beta| must be < 1 (speed must be less than c)")

        self.gamma = 1.0 / np.sqrt(1.0 - self.beta2)
        self.c = const.c.to_value(vel_UNIT)
        self.Lambda = self._build_matrix()

    def _build_matrix(self):
        """
        builds the 4x4 matrix Lambda^mu_nu for a general boost with velocity beta*c
        """
        gamma, beta, beta2 = self.gamma, self.beta, self.beta2

        Lam = np.eye(4)
        Lam[0, 0] = gamma
        Lam[0, 1:] = -gamma * beta
        Lam[1:, 0] = -gamma * beta

        if beta2 > 0:
            outer = np.outer(beta, beta) / beta2
            Lam[1:, 1:] = np.eye(3) + (gamma - 1.0) * outer
        else:
            Lam[1:, 1:] = np.eye(3)

        return Lam
    
    def transform(self, x_nu):
        """
        x_nu = (t, x, y, z) as a tuple/list of astropy Quantities
               t in time_UNIT, x,y,z in len_UNIT

        returns x'_nu = (t', x', y', z') as astropy Quantities, same units
        """
        t, x, y, z = x_nu
        t_val = t.to_value(time_UNIT)
        x_val = x.to_value(len_UNIT)
        y_val = y.to_value(len_UNIT)
        z_val = z.to_value(len_UNIT)

        x0 = self.c * t_val                        # x^0 = c*t
        x_vec = np.array([x0, x_val, y_val, z_val])

        x_prime = self.Lambda @ x_vec               # x'^mu = Lambda^mu_nu x^nu

        t_prime = (x_prime[0] / self.c) * time_UNIT
        x_prime_ = x_prime[1] * len_UNIT
        y_prime_ = x_prime[2] * len_UNIT
        z_prime_ = x_prime[3] * len_UNIT

        return t_prime, x_prime_, y_prime_, z_prime_
    
    def spatial_length(self, x_nu):
        t, x, y, z = x_nu
        L = np.sqrt(x**2 + y**2 + z**2).to(len_UNIT)

        t_p, x_p, y_p, z_p = self.transform(x_nu)
        L_prime = np.sqrt(x_p**2 + y_p**2 + z_p**2).to(len_UNIT)

        return L, L_prime