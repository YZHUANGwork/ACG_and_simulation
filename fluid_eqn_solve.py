"""


∂V/∂t + (V·∇)V = -(1/rho)∇p + \nu ∇²V + f
∇·V(x,y, t) = 0, V = vx x̂ + vy ŷ,  d(vx)/dx + d(vy)/dy = 0

vorticity–stream function method
change parameter from V to w = ∇×V
w ẑ = ∇ × V = [d(vy)/dx - d(vx)/dy]  ẑ


∇×[∂V/∂t + (V·∇)V] = ∇×[-(1/rho)∇p + \nu ∇²V + f]

∂w/∂t + ∇×[(V·∇)V]=∇×(\nu ∇²V + f)

def ψ
1 ) vx =  ∂ψ/∂y
2 ) vy = -∂ψ/∂x
satisty ∇·V(x,y) = 0, 

∇²ψ = −w
3 ) ∂w/∂t + (V·∇)w = \nu ∇²w + ∇×f

dw/dt ẑ= -( vx * dw/dx + vy * dw/dy )
                  + \nu * ( d2(w)/dx2 + d2(w)/dy2 )
                  + F(x,y,t)- eta * mask_outside(x,y) * w  <- confines to r < R

    d(dye)/dt      = -(u . grad) dye + kappa * laplacian(dye)   <- passive tracer,
                      rides the flow, makes the winding spiral visible

"""
import numpy as np
from scipy.integrate import solve_ivp
import astropy.units as u

# ── dimensions (matches heat_equation_analytical.py) ─────────────────────────
length_UNIT = u.cm
time_UNIT   = u.s
visc_UNIT   = length_UNIT**2 / time_UNIT
 
# ── DOMAIN SHAPE ───────────────────────────────────────────────────────────────
DOMAIN_R_DEFAULT = (15.0 * length_UNIT).value   # domain radius, cm (~30 cm across)
# ── PHYSICAL DOMAIN ───────────────────────────────────────────────────────────
DOMAIN_W_DEFAULT = (40.0 * length_UNIT).value   # x-extent, cm
DOMAIN_H_DEFAULT = (40.0 * length_UNIT).value   # y-extent, cm
NCOL_DEFAULT = 64                               # grid columns across DOMAIN_W
NROW_DEFAULT = 64                               # grid rows across DOMAIN_H
# (dx = dy = 0.625 cm at these defaults)


def _build_grid(domain_w=DOMAIN_W_DEFAULT, domain_h=DOMAIN_H_DEFAULT,
                 ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT):
    """
    Build (x, y, X, Y, dx, dy) for a given domain size/resolution. Same
    signature/behavior as heat_equation_analytical.py's _build_grid(), except
    x/y run [0, domain_w)/[0, domain_h) with endpoint=False, since this
    solver's Poisson step needs a periodic domain (FFT-based).
    """
    x = np.linspace(0, domain_w, ncol, endpoint=False)
    y = np.linspace(0, domain_h, nrow, endpoint=False)
    X, Y = np.meshgrid(x, y, indexing="ij")
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return x, y, X, Y, dx, dy


x, y, X, Y, dx, dy = _build_grid()
NCOL, NROW = NCOL_DEFAULT, NROW_DEFAULT

 
# ── DEFAULT PHYSICAL PARAMETERS ────────────────────────────────────────────────
NU_DEFAULT    = (0.25 * visc_UNIT).value        # spin viscosity, cm^2/s (raised so continuous
                                                  # stirring reaches a STEADY swirl -- energy
                                                  # dissipated balances energy the forcing puts in)
# ── THE FORCING TERM (this is F(x,y,t)) ─────────────────────────────────────────
FORCE_A_DEFAULT       = (1.0 / time_UNIT**2).value    # push strength, 1/s^2
FORCE_SIGMA_DEFAULT   = (3.0 * length_UNIT**2).value  # forced region size^2, cm^2
FORCE_BASE_SPEED_DEFAULT = (0.5 / time_UNIT).value    # base angular speed, rad/s (slow -> laminar)
T_END_DEFAULT         = (60.0 * time_UNIT).value      # total simulated time, s

#F(x, y, t) = force_A * exp( -[(x - sx(t))² + (y - sy(t))²] / force_sigma )
    
def _wavenumbers(domain_w, domain_h, ncol, nrow):
    kx = np.fft.fftfreq(ncol, d=domain_w / ncol) * 2 * np.pi
    ky = np.fft.fftfreq(nrow, d=domain_h / nrow) * 2 * np.pi
    KX, KY = np.meshgrid(kx, ky, indexing="ij")
    K2 = KX**2 + KY**2
    K2_inv = np.where(K2 == 0, 1, K2)
    return KX, KY, K2, K2_inv

def speed_from_spin(w, KX, KY, K2_inv):
    """vx, vy from w via the Poisson equation for psi (see module docstring)."""
    w_hat = np.fft.fft2(w)
    psi_hat = w_hat / K2_inv
    psi_hat[0, 0] = 0
    vx = np.real(np.fft.ifft2(1j * KY * psi_hat))
    vy = np.real(np.fft.ifft2(-1j * KX * psi_hat))
    return vx, vy
 
class Solution:
    """
    The solution. Every w frame is converted to (vx, vy) once, up front, when
    the Solution is built -- call solution.velocity(t) to get it back.
    """
    def __init__(self, t_eval, w_frames, r, KX, KY, K2_inv, domain_r, force_position):
        self._t_eval = t_eval
        self._w_frames = w_frames
        self._v_frames = [speed_from_spin(w, KX, KY, K2_inv) for w in w_frames]   # w -> (vx,vy), done once
        self.r = r                    # distance from domain center, cm (grid, for masking/plotting)
        self.domain_r = domain_r      # radius of the confined region, cm
        self.force_position = force_position   # callable t -> (sx, sy), where F(x,y,t) is centered
 
    def _nearest_index(self, t):
        return int(np.argmin(np.abs(self._t_eval - t)))
 
    def w(self, t):
        """Spin field at time t (nearest solved frame)."""
        return self._w_frames[self._nearest_index(t)]
 
    def velocity(self, t):
        """(vx, vy) at time t (nearest solved frame) -- this is THE solution."""
        return self._v_frames[self._nearest_index(t)]
 
 
def solve_fluid_flow(domain_w=DOMAIN_W_DEFAULT, domain_h=DOMAIN_H_DEFAULT,
                      ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT,
                      domain_r=DOMAIN_R_DEFAULT, domain_cx=None, domain_cy=None,
                      nu=NU_DEFAULT,
                      force_A=FORCE_A_DEFAULT, force_sigma=FORCE_SIGMA_DEFAULT,
                      force_base_speed=FORCE_BASE_SPEED_DEFAULT, t_end=T_END_DEFAULT,
                      n_frames=100, rtol=1e-2, atol=1e-4):
    """
    Solve for the velocity of a swirl confined to a circular fluid domain of
    radius domain_r, driven by a wobbly, outward-spiraling forcing term.
 
    Returns a Solution object -- call solution.velocity(t) to get (vx, vy).
    """
    x_l, y_l, X_l, Y_l, dx_l, dy_l = _build_grid(domain_w, domain_h, ncol, nrow)
    KX, KY, K2, K2_inv = _wavenumbers(domain_w, domain_h, ncol, nrow)
    npts = ncol * nrow
 
    cx = domain_cx if domain_cx is not None else domain_w / 2
    cy = domain_cy if domain_cy is not None else domain_h / 2
    r2 = (X_l - cx) ** 2 + (Y_l - cy) ** 2
    r = np.sqrt(r2)
    inside = (r <= domain_r).astype(float)   # 1 inside the domain, 0 outside -- the wall
 
    def force_position(t):
        theta = force_base_speed * t + 0.3 * np.sin(0.17 * t)
        radius = 1.0 + 1. * t + 1.5 * np.sin(0.23 * t + 1.0)
        radius = min(radius, domain_r * 0.85)
        wander_x = 0.8 * np.sin(0.11 * t + 2.0)
        wander_y = 0.8 * np.sin(0.13 * t + 0.5)
        sx = cx + wander_x + radius * np.cos(theta)
        sy = cy + wander_y + radius * np.sin(theta)
        return sx, sy
 
    def ddx(fh): return np.real(np.fft.ifft2(1j * KX * fh))
    def ddy(fh): return np.real(np.fft.ifft2(1j * KY * fh))
    def lap(fh): return np.real(np.fft.ifft2(-K2 * fh))
 
    def rhs(t, w_flat):
        w = w_flat.reshape(ncol, nrow)
        w_hat = np.fft.fft2(w)
        vx, vy = speed_from_spin(w, KX, KY, K2_inv)
 
        # hard wall part 1: block advection across the boundary
        vx = vx * inside
        vy = vy * inside
 
        sx, sy = force_position(t)
        force_r2 = (X_l - sx) ** 2 + (Y_l - sy) ** 2
        F = force_A * np.exp(-force_r2 / force_sigma)
 
        dw = -(vx * ddx(w_hat) + vy * ddy(w_hat)) + nu * lap(w_hat) + F
        return dw.flatten()
 
    w0 = np.zeros((ncol, nrow))
    state0 = w0.flatten()
 
    # hard wall part 2: integrate in short chunks, clamp w to zero outside
    # the domain between them (blocks diffusion from leaking spin across
    # the edge, which masking velocity alone does not stop)
    t_eval = np.linspace(0, t_end, n_frames)
    w_frames = []
    state = state0.copy()
    t_prev = 0.0
    for t_now in t_eval:
        if t_now > t_prev:
            sol = solve_ivp(rhs, (t_prev, t_now), state, method="RK45",
                             rtol=rtol, atol=atol, dense_output=False)
            state = sol.y[:, -1] * inside.flatten()   # hard clamp
        w_frames.append(state.reshape(ncol, nrow).copy())
        t_prev = t_now
 
    return Solution(t_eval, w_frames, r, KX, KY, K2_inv, domain_r, force_position)
 
 
if __name__ == "__main__":
    solution = solve_fluid_flow()
    print(f"grid: {NCOL_DEFAULT} x {NROW_DEFAULT}, domain radius: {DOMAIN_R_DEFAULT} cm, "
          f"domain: {DOMAIN_W_DEFAULT} x {DOMAIN_H_DEFAULT} cm")
    for t in solution._t_eval:   # every solved frame, no arbitrary skipping
        vx, vy = solution.velocity(t)
        speed = np.hypot(vx, vy)
        print(f"t={t:5.1f} s   max speed inside = {speed[solution.r < solution.domain_r].max():.2f} cm/s")