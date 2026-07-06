import numpy as np
import matplotlib
import math

from scipy.interpolate import interp1d

import astropy.units as u
import astropy.constants as const 

data = np.loadtxt('ciexyz31_1nm.txt', delimiter=',')#https://www.site.uottawa.ca/~edubois/mdsp/data/index.html
wl_table, xbar_table, ybar_table, zbar_table = data[:,0], data[:,1], data[:,2], data[:,3]

xbar_interp = interp1d(wl_table, xbar_table, kind='cubic', bounds_error=False, fill_value=0)
ybar_interp = interp1d(wl_table, ybar_table, kind='cubic', bounds_error=False, fill_value=0)
zbar_interp = interp1d(wl_table, zbar_table, kind='cubic', bounds_error=False, fill_value=0)

def power_to_temperature(flux):
    """Stefan-Boltzmann law: T = (flux/sigma)^(1/4)"""
    return (flux / const.sigma_sb)**0.25

def temperature_to_peak_wavelength(T):
    """Wien's displacement law: lambda_max = b/T"""
    return (const.b_wien / T).to(u.nm)

def planck_law(wavelength, T):
    """Planck's law: full spectral radiance B(lambda, T)"""
    
    exponent = (const.h * const.c / (wavelength * const.k_B * T)).to(u.dimensionless_unscaled)
    B = (2*const.h*const.c**2 / wavelength**5) / (np.exp(exponent) - 1)
    return B.to(u.W / u.m**2 / u.nm)



def blackbody_to_rgb_CIE1931(T):
    d_wl = 1
    wl_nm = np.arange(min(wl_table), max(wl_table)+d_wl, d_wl) * u.nm
    B = planck_law(wl_nm, T)   # W/m^2/nm, matches the integration axis unit directly

    xbar = xbar_interp(wl_nm.value) * u.dimensionless_unscaled
    ybar = ybar_interp(wl_nm.value) * u.dimensionless_unscaled
    zbar = zbar_interp(wl_nm.value) * u.dimensionless_unscaled
    
    X = np.trapz(B * xbar, wl_nm)   # Quantity, W/m^2 after integration
    Y = np.trapz(B * ybar, wl_nm)
    Z = np.trapz(B * zbar, wl_nm)

    XYZ = u.Quantity([X, Y, Z])
    XYZ = (XYZ / XYZ[1]).to(u.dimensionless_unscaled)   # normalize luminance

    M = np.array([[ 3.2406, -1.5372, -0.4986],
                  [-0.9689,  1.8758,  0.0415],
                  [ 0.0557, -0.2040,  1.0570]])
    RGB_linear = np.clip(M @ XYZ.value, 0, None)

    gamma = lambda c: np.where(c <= 0.0031308, 12.92*c, 1.055*c**(1/2.4)-0.055)
    return np.clip(gamma(RGB_linear), 0, 1)


def clamp_temperature(T, T_min=1000, T_max=20000):
    T_kelvin = T.to(u.K).value
    return max(T_min, min(T_max, T_kelvin))


def random_temperatures(n, T_min=1000, T_max=20000, log_scale=True, rng=42):
    rng = np.random.default_rng(rng)
    if log_scale:
        log_T = rng.uniform(np.log10(T_min), np.log10(T_max), size=n)
        T = 10**log_T
    else:
        T = rng.uniform(T_min, T_max, size=n)
    return T * u.K