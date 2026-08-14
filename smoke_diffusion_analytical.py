"""
smoke_diffusion_analytical.py
-----------------------------
Spherical gas , diffusion + decay.
3d spherical coordinate
    
    dC(r,t)/dt = D * Laplacian(C) - beta*C,      C(r,0) = C0 for r<=R0, else 0 
C(r,t) concentration, mass is conserved 

Substituting u(r,t) = r*C(r,t) turns the 3D radial diffusion equation into
an ordinary 1D diffusion equation for u (odd-extended about r=0), which has
a closed-form solution since the initial data u0(r)=C0*r is piecewise
linear:

    u(r,t) = C0*exp(-beta*t) * [
                 sqrt(D*t/pi) * (exp(-(R0+r)^2/(4Dt)) - exp(-(R0-r)^2/(4Dt)))
                 + (r/2) * (erf((R0-r)/sqrt(4Dt)) + erf((R0+r)/sqrt(4Dt)))
             ]

    C(r,t) = u(r,t) / r        (r > 0)
    C(0,t) = C0*exp(-beta*t) * (erf(R0/sqrt(4Dt)) - (R0/sqrt(pi*D*t))*exp(-R0^2/4Dt))

Mathematics of Diffusion crank
#Diffusion of an initial sphere into an infinite medium

Reference (this is the beta=0 case; the exp(-beta*t) decay factor just
multiplies on top, it doesn't change the r-dependence):
    https://notebook.community/simulkade/FVTool/Examples/External/Diffusion1DSpherical_Analytic-vs-FVTool-vs-Fipy/diffusion1Dspherical_analytic_vs_FVTool_vs_Fipy

"""
import numpy as np
from scipy.special import erf
import astropy.units as u

# dimension:
length_UNIT = u.m
time_UNIT   = u.s
mass_UNIT   = u.g
density_UNIT   = mass_UNIT / length_UNIT**3  

# ── parameters ──────────────────────────────────────────────────────────────
D    = (1.7 * length_UNIT**2 / time_UNIT).value   # diffusion coefficient
BETA = (0.025 / time_UNIT).value                  # decay rate
C0   = (1.0 * density_UNIT).value                    # initial concentration inside the sphere
R0   = (3.0 * length_UNIT).value                  # sphere radius
CENTER = (np.array([0.,0.,0.]) * length_UNIT).value   # m, sphere center

DOMAIN_L = (3.0 * length_UNIT).value   # cubic domain, m per side
NGRID = 64                              # grid points per axis

T_FINAL  = (6.0 * time_UNIT).value
N_FRAMES = 100
NGRID = 100

def analytic_C(r, t, D=D, beta=BETA, C0=C0, R0=R0):
    """solution in spherical coordinate, spherical symmetry, Closed-form C(r,t) above, vectorized over r."""
    
    r_safe = np.where(r < 1e-9, 1e-9, r)
    s = np.sqrt(4*D*t)
    term1 = np.sqrt(D*t/np.pi) * (np.exp(-(R0+r_safe)**2/(4*D*t))
                                   - np.exp(-(R0-r_safe)**2/(4*D*t)))
    term2 = 0.5*r_safe*(erf((R0-r_safe)/s) + erf((R0+r_safe)/s))
    u_val = C0*np.exp(-beta*t)*(term1 + term2)
    C = u_val/r_safe

    C_center = C0*np.exp(-beta*t)*(erf(R0/s) - (R0/np.sqrt(np.pi*D*t))*np.exp(-R0**2/(4*D*t)))
    return np.where(r < 1e-6, C_center, C)


def C_xyz(x, y, z, t, center=CENTER, D=D, beta=BETA, C0=C0, R0=R0):
    
    """
    solution in cartesian coordinate,  input a point , return the C(t) at that point
    """
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2 + (z - center[2])**2)
    return analytic_C(r, t, D=D, beta=beta, C0=C0, R0=R0)
 
def C_projected_grid(margin_factor=1.5, z_margin_factor=5, ngrid=NGRID,
                      t_final=T_FINAL, n_frames=N_FRAMES, margin_t_final=T_FINAL,
                      center=CENTER, D=D, beta=BETA, C0=C0, R0=R0):
    """
    solution in cartesian coordinate,  2d, integrate over the third dimension over line of sight 
    compress 3d into 2d 
    
    Read the analytical solution and build it on an (x,y,t) grid, projected
    (integrated) along the line of sight z:
 
        rho(x,y,t) = integral C(x,y,z,t) dz
 
    Grid is centered on `center` and sized generously (R0 plus
    margin_factor diffusion lengths, using margin_t_final -- the module's
    own T_FINAL default, NOT the (possibly shorter) t_final actually
    simulated -- so the margin stays conservative even if t_final is
    reduced) so C~0 well before the edge. Returns (x, y, t_eval, proj_t),
    proj_t shape (ngrid, ngrid, n_frames).
    """
    margin = R0 + margin_factor * np.sqrt(D * margin_t_final)
    x = np.linspace(center[0] - margin, center[0] + margin, ngrid)
    y = np.linspace(center[1] - margin, center[1] + margin, ngrid)
    X, Y = np.meshgrid(x, y, indexing='ij')
 
    z_max = z_margin_factor * R0   # z-integration range, checked convergence earlier
    z_line = np.linspace(center[2] - z_max, center[2] + z_max, 400)
 
    t_eval = np.linspace(t_final / n_frames, t_final, n_frames)
 
    # integrate over z to compress 3D into a 2D line-of-sight integral,
    
    proj_t = np.empty((ngrid, ngrid, n_frames))
    for i, t in enumerate(t_eval):
        C = C_xyz(X[:, :, None], Y[:, :, None], z_line[None, None, :], t,
                  center=center, D=D, beta=beta, C0=C0, R0=R0)
        proj_t[..., i] = np.trapz(C, z_line, axis=2)
 
    return x, y, t_eval, proj_t
 
    
if __name__ == "__main__":
    t_test = 10.0
    M = C0*(4/3)*np.pi*R0**3
    c_sphere = analytic_C(np.array([0.0]), t_test)[0]
    c_point = M/(4*np.pi*D*t_test)**1.5*np.exp(-BETA*t_test)
    print(f"sphere formula: {c_sphere:.6f}, point-source approx: {c_point:.6f}")
 
    # C_xyz works at any point(s), no grid required -- e.g. a handful of
    # arbitrary 3D coordinates:
    xs = np.array([0.0, 1.0, 5.0])
    ys = np.array([0.0, 0.0, 0.0])
    zs = np.array([0.0, 0.0, 0.0])
    print("C_xyz at 3 sample points, t=10:", C_xyz(xs, ys, zs, 10.0))