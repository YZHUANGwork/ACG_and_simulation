"""
heat_equation_analytical.py
-----------------------------
T_full = T_h + T_p
Operator: L =  -alpha * Lap + ∂/∂t + v dot grad + beta
Source term is S
Lf = 0, find green function, 
Lf = S find particular solution
Boundary condition T(x→inf, y→inf, t) → 0boundedness condition at infinity

use ξx = x-v_x *t, ξy = y-v_y * t to kill the v term

Operator reduce to L =  -alpha * Lap + ∂/∂t + beta

assumes an infinite domain (no walls) -- a valid
approximation here because the flame decays (~mm) long before reaching
the real walls (cm away), not an exact match to the bounded problem the
way candle_flame_sim.rhs() is.

T(x,y,t) = T_h(x,y,t) + T_p(x,y,t)

T_h   use exp^... * exp^.... to get delta, homework type of known Green's function.
T_p   S(x,y), integrate over Green's function



Also included: the constant-wind special cases (solve_steady_state, the
t->infinity limit of T_p under constant wind; T_cosine_series/solve_coeffs,
an alternate closed form using the domain's Neumann eigenbasis directly --
this one is only numerically safe to evaluate near the source, see its
docstring).


"""
import numpy as np
from scipy.special import k0
from scipy.signal import fftconvolve
from scipy.integrate import quad
from scipy.ndimage import shift as _ndi_shift
import astropy.units as u

# dimension:
length_UNIT = u.cm
time_UNIT   = u.s
temp_UNIT   = u.K   # T is a temperature RISE ABOVE AMBIENT (T=0 = ambient), not absolute

# ── PHYSICAL DOMAIN ──────────────────────────────────────────────────────────
# these are only DEFAULTS -- every function below takes domain_w/domain_h/
# ncol/nrow as real keyword arguments and uses these only if you don't
# override them (same pattern as alpha/beta/v0/S0)
DOMAIN_W_DEFAULT = (8.0 * length_UNIT).value   # x-extent, cm
DOMAIN_H_DEFAULT = (5.0 * length_UNIT).value   # y-extent, cm
NCOL_DEFAULT = 80                              # grid columns across DOMAIN_W
NROW_DEFAULT = 50                              # grid rows across DOMAIN_H


def _build_grid(domain_w=DOMAIN_W_DEFAULT, domain_h=DOMAIN_H_DEFAULT,
                 ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT):
    """
    Build (x, y, X, Y, dx, dy) for a given domain size/resolution. Called
    once per top-level function call (not rebuilt inside every inner loop
    iteration) and threaded through from there.
    """
    x = np.linspace(0, domain_w, ncol)
    y = np.linspace(0, domain_h, nrow)
    X, Y = np.meshgrid(x, y)
    dx = x[1] - x[0]
    dy = y[1] - y[0]
    return x, y, X, Y, dx, dy


# default grid, built once -- used when no overrides are given (avoids
# rebuilding it on every single call for the common case)
x, y, X, Y, dx, dy = _build_grid()
NCOL, NROW = NCOL_DEFAULT, NROW_DEFAULT

# ── DEFAULT PHYSICAL PARAMETERS ─────────────────────────────────────────────
# these are only DEFAULTS -- every function below takes alpha/beta/v0/S0
# as real keyword arguments and uses these only if you don't override them
ALPHA_DEFAULT = (0.35 * length_UNIT**2 / time_UNIT).value
V0_DEFAULT    = (3.0  * length_UNIT / time_UNIT).value
BETA_DEFAULT  = (1.6  / time_UNIT).value
S0_DEFAULT    = (1000.0  * temp_UNIT / time_UNIT).value

# ── DEFAULT SOURCE: fixed heat source shape, low and centered ─────────────────
SOURCE_R_DEFAULT = 6
SOURCE_C_DEFAULT = NCOL // 2
SOURCE_SIGMA_DEFAULT = 1.2

def make_source(S0=S0_DEFAULT, source_r=SOURCE_R_DEFAULT, source_c=SOURCE_C_DEFAULT,
                 source_sigma=SOURCE_SIGMA_DEFAULT, ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT):
    """
    S(x,y) for a given source strength S0, grid position (source_r, source_c),
    spatial width source_sigma (all in grid cells, not physical units), and
    grid resolution (ncol, nrow) -- must match whatever domain is actually
    being used, or the source's grid position won't line up.
    """
    rr, cc = np.meshgrid(np.arange(nrow), np.arange(ncol), indexing="ij")
    shape = np.exp(-(((rr - source_r) ** 2 + (cc - source_c) ** 2) / (2 * source_sigma**2)))
    return S0 * shape


# module-level default source, used when functions aren't given a custom one
S = make_source()

# ── DEFAULT WIND: v for v dot grad term ────────────────────────────────────────
def default_wind(t, v0=V0_DEFAULT):
    vx = 1.8 + 1.4 * np.sin(0.9 * t) + 0.7 * np.sin(2.3 * t + 1.0) + 0.4 * np.sin(4.1 * t + 2.5)   
    
    vy = v0 * (1 + 0.12 * np.sin(3.7 * t + 0.5))
    return vx, vy


# ── INITIAL CONDITION ─────────────────────────────────────────────────────────
def make_T0(ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT):
    return np.zeros((nrow, ncol))


T0 = make_T0()   # default, used when no domain override is given


# ── STEADY STATE (constant-wind special case of T_p), closed form ─────────────
# T_p is the exact steady solution on an infinite domain with a point
# source, given by the modified-Helmholtz Green's function:
#
#   G(x,y) = 1/(2*pi*alpha) * exp(v0*y / (2*alpha)) * K0(kappa*r)
#   kappa  = sqrt(v0^2/(4*alpha^2) + beta/alpha)
#
# Assumes CONSTANT wind (v0 only) -- for time-varying wind, use
# T_particular() instead.
def solve_steady_state(alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT, v0=V0_DEFAULT, S0=S0_DEFAULT,
                        source_r=SOURCE_R_DEFAULT, source_c=SOURCE_C_DEFAULT, source_sigma=SOURCE_SIGMA_DEFAULT,
                        domain_w=DOMAIN_W_DEFAULT, domain_h=DOMAIN_H_DEFAULT,
                        ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT):
    """
    T_p, the fixed flame shape under CONSTANT wind v0, for given
    coefficients alpha/beta/v0/S0 and given domain size/resolution --
    one closed-form convolution, no time integration, no linear solve.
    Returns an (nrow, ncol) array.
    """
    _, _, _, _, dx_l, dy_l = _build_grid(domain_w, domain_h, ncol, nrow)
    kappa = np.sqrt(v0**2 / (4 * alpha**2) + beta / alpha)

    kx = np.arange(-ncol, ncol + 1) * dx_l
    ky = np.arange(-nrow, nrow + 1) * dy_l
    KX, KY = np.meshgrid(kx, ky)
    R = np.hypot(KX, KY)
    R[R == 0] = dx_l / 4   # avoid the K0 log-singularity exactly at its own center
    green = (1 / (2 * np.pi * alpha)) * np.exp(v0 * KY / (2 * alpha)) * k0(kappa * R)

    S_local = make_source(S0, source_r, source_c, source_sigma, ncol=ncol, nrow=nrow)
    return fftconvolve(S_local, green, mode="same") * dx_l * dy_l


# ── COSINE SERIES FORM (constant wind), typed out directly ─────────────────────
# T(x,y,t) = exp(c*y) * sum_{m,n} coeffs[m,n] * cos(m*pi*x/Lx) * cos(n*pi*y/Ly)
#                      * (1 - exp(-lambda_mn * t))
#   c          = v0 / (2*alpha)
#   lambda_mn  = alpha*((m*pi/Lx)**2 + (n*pi/Ly)**2) + beta + v0**2/(4*alpha)
#
# coeffs[m, n] are FREE -- this is just the formula. solve_coeffs() below
# derives the specific coefficients matching a given source.
#
# CAUTION: evaluating T_cosine_series far from the source (large y) is
# numerically unstable -- the exp(c*y) prefactor grows huge at the top of
# the domain, and a truncated series can't cancel it precisely enough.
# Safe near the source; not trustworthy far from it. T_particular()/
# solve_steady_state() don't have this problem and should be preferred.
def T_cosine_series(t, coeffs, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT, v0=V0_DEFAULT,
                     domain_w=DOMAIN_W_DEFAULT, domain_h=DOMAIN_H_DEFAULT,
                     ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT):
    """
    Evaluate the cosine-series solution (constant wind) on the grid at
    time t, for given coefficients alpha/beta/v0, domain size/resolution,
    and a (M, N) array of free series coefficients.
    """
    x_l, y_l, X_l, Y_l, _, _ = _build_grid(domain_w, domain_h, ncol, nrow)
    c = v0 / (2 * alpha)
    M, N = coeffs.shape
    T = np.zeros_like(X_l)
    for m in range(M):
        cos_x = np.cos(m * np.pi * x_l / domain_w)
        for n in range(N):
            cos_y = np.cos(n * np.pi * y_l / domain_h)
            lam = alpha * ((m * np.pi / domain_w) ** 2 + (n * np.pi / domain_h) ** 2) \
                  + beta + v0**2 / (4 * alpha)
            T += coeffs[m, n] * (1 - np.exp(-lam * t)) * cos_x[None, :] * cos_y[:, None]
    return np.exp(c * Y_l) * T


def solve_coeffs(M=20, N=20, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT, v0=V0_DEFAULT, S0=S0_DEFAULT,
                  source_r=SOURCE_R_DEFAULT, source_c=SOURCE_C_DEFAULT, source_sigma=SOURCE_SIGMA_DEFAULT,
                  domain_w=DOMAIN_W_DEFAULT, domain_h=DOMAIN_H_DEFAULT,
                  ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT):
    """
    Solve for the coefficients corresponding to a given source (S0), wind
    (v0), and domain size/resolution, constant wind, using the
    orthogonality of the cosine basis:

        a_mn = (1/norm_m/norm_n) * integral( S_tilde * cos_m * cos_n )

    Returns an (M, N) array of coefficients.
    """
    x_l, y_l, X_l, Y_l, _, _ = _build_grid(domain_w, domain_h, ncol, nrow)
    c = v0 / (2 * alpha)
    S_local = make_source(S0, source_r, source_c, source_sigma, ncol=ncol, nrow=nrow)
    S_tilde = S_local * np.exp(-c * Y_l)
    coeffs = np.zeros((M, N))
    for m in range(M):
        cos_x = np.cos(m * np.pi * x_l / domain_w)
        norm_x = domain_w if m == 0 else domain_w / 2
        for n in range(N):
            cos_y = np.cos(n * np.pi * y_l / domain_h)
            norm_y = domain_h if n == 0 else domain_h / 2
            integrand = S_tilde * cos_x[None, :] * cos_y[:, None]
            coeffs[m, n] = np.trapezoid(np.trapezoid(integrand, x_l, axis=1), y_l) / (norm_x * norm_y)
    return coeffs


# ── WIND DRIFT INTEGRALS ────────────────────────────────────────────────────────
# X_d(t), Y_d(t): total sideways/vertical drift accumulated by a given
# wind function from 0 to t. Pass your own wind_fn(t) -> (vx, vy) if you
# don't want default_wind.
def X_d(t, wind_fn=default_wind):
    return quad(lambda tp: wind_fn(tp)[0], 0, t)[0]


def Y_d(t, wind_fn=default_wind):
    return quad(lambda tp: wind_fn(tp)[1], 0, t)[0]


def _green_kernel_grid(s, alpha, dx_l, dy_l, ncol, nrow):
    """
    Build the (KX, KY) grid for the Green's function G(xi,eta,s), sized so
    the Gaussian (std dev sqrt(2*alpha*s)) is actually fully contained --
    NOT clipped to the domain size, which would silently throw away mass.
    """
    sigma = np.sqrt(2 * alpha * s)
    half_w = max(int(np.ceil(6 * sigma / dx_l)), ncol)
    half_h = max(int(np.ceil(6 * sigma / dy_l)), nrow)
    kx = np.arange(-half_w, half_w + 1) * dx_l
    ky = np.arange(-half_h, half_h + 1) * dy_l
    return np.meshgrid(kx, ky)


# ── HOMOGENEOUS SOLUTION T_h(x,y,t), moving-frame Green's function ────────────
#
#   T_h(x,y,t) = integral( G(xi-xi', eta-eta', t) * H0(xi',eta') ) dxi' deta'
#   G(xi,eta,t) = exp(-beta*t) / (4*pi*alpha*t) * exp(-(xi**2+eta**2)/(4*alpha*t))
#   xi = x - X_d(t),  eta = y - Y_d(t)
#   H0(xi,eta) = T_h(x,y,0) = T0(x,y) - T_p(x,y)
#
# This is the drift-cancelled form: the moving-frame substitution kills
# the v(t)*dT/dx, v(t)*dT/dy terms exactly for ANY vx(t), vy(t). T_p here
# (the t=0 reference) uses the CONSTANT-wind steady state as a stand-in --
# a simplification, not an exact match to a truly time-varying particular
# solution.
#operator is alpha * lap + v dot grad + beta
def T_homogeneous(t, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT, v0=V0_DEFAULT,
                   S0=S0_DEFAULT, wind_fn=default_wind,
                   source_r=SOURCE_R_DEFAULT, source_c=SOURCE_C_DEFAULT, source_sigma=SOURCE_SIGMA_DEFAULT,
                   domain_w=DOMAIN_W_DEFAULT, domain_h=DOMAIN_H_DEFAULT,
                   ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT):
    """
    T_h(x,y,t) via the moving-frame Green's function convolution, for
    given coefficients and domain size/resolution. Returns an (nrow, ncol) array.
    """
    _, _, _, _, dx_l, dy_l = _build_grid(domain_w, domain_h, ncol, nrow)

    # H0 = T(0) - T_p(0). T_p(0) is EXACTLY zero always, by construction of
    # the Duhamel integral (integrating from 0 to 0 is empty) -- NOT the
    # steady-state value. So H0 is just the true initial condition T0 itself.
    T0_l = make_T0(ncol=ncol, nrow=nrow)
    H0 = T0_l

    if t <= 0:
        return H0

    KX, KY = _green_kernel_grid(t, alpha, dx_l, dy_l, ncol, nrow)
    G = (np.exp(-beta * t) / (4 * np.pi * alpha * t)) * np.exp(-(KX**2 + KY**2) / (4 * alpha * t))

    convolved = fftconvolve(H0, G, mode="same") * dx_l * dy_l

    shift_rows = Y_d(t, wind_fn) / dy_l
    shift_cols = X_d(t, wind_fn) / dx_l
    T_h = _ndi_shift(convolved, shift=(shift_rows, shift_cols), mode="constant", cval=0.0)

    return T_h


# ── PARTICULAR SOLUTION T_p(x,y,t), Duhamel's principle ────────────────────────
# Separate problem from T_h: zero initial data, driven only by the source
# S(x,y). Each instant tau in [0,t], the source emits a small heat packet;
# between tau and t it diffuses, decays (beta), and drifts by whatever the
# wind moved between emission and now:
#
#   T_p(x,y,t) = integral_0^t [ G(., ., t-tau) convolved with S ],
#                shifted by (X_d(t)-X_d(tau), Y_d(t)-Y_d(tau)),  d(tau)
#
# Validated: under constant wind, converges to solve_steady_state() as
# t -> infinity (0.127 -> 0.139 -> 0.151 -> 0.156 -> ... -> 0.168, at
# n_tau = 200/500/1000/3000, against a target of 0.168).
def T_particular(t, n_tau=200, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT,
                  S0=S0_DEFAULT, wind_fn=default_wind,
                  source_r=SOURCE_R_DEFAULT, source_c=SOURCE_C_DEFAULT, source_sigma=SOURCE_SIGMA_DEFAULT,
                  domain_w=DOMAIN_W_DEFAULT, domain_h=DOMAIN_H_DEFAULT,
                  ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT):
    """
    T_p(x,y,t) via Duhamel's principle, for given coefficients and domain
    size/resolution: integrate the homogeneous propagator applied to the
    source S(x,y) over all past emission times tau in [0, t].
    Returns an (nrow, ncol) array.
    """
    _, _, X_l, _, dx_l, dy_l = _build_grid(domain_w, domain_h, ncol, nrow)

    if t <= 0:
        return np.zeros_like(X_l)

    S_local = make_source(S0, source_r, source_c, source_sigma, ncol=ncol, nrow=nrow)

    # midpoint rule in tau -- avoids tau=t exactly, where G's 1/(t-tau)
    # factor diverges (that instant hasn't had time to diffuse at all yet)
    d_tau = t / n_tau
    taus = np.linspace(0, t, n_tau, endpoint=False) + d_tau / 2

    Xd_t = X_d(t, wind_fn)
    Yd_t = Y_d(t, wind_fn)

    total = np.zeros_like(X_l)
    for tau in taus:
        s = t - tau
        KX, KY = _green_kernel_grid(s, alpha, dx_l, dy_l, ncol, nrow)
        G = (np.exp(-beta * s) / (4 * np.pi * alpha * s)) * np.exp(-(KX**2 + KY**2) / (4 * alpha * s))
        conv = fftconvolve(S_local, G, mode="same") * dx_l * dy_l

        dX = Xd_t - X_d(tau, wind_fn)
        dY = Yd_t - Y_d(tau, wind_fn)
        total += _ndi_shift(conv, shift=(dY / dy_l, dX / dx_l), mode="constant", cval=0.0)

    # rectangle rule: n_tau equal-width slices of width d_tau, each sampled
    # at its midpoint -- NOT np.trapezoid, which assumes its sample points
    # span the full [0,t] domain (ours start at d_tau/2)
    T_p = total * d_tau
    return T_p


# ── FULL SOLUTION: separately solved pieces, summed only at the end ──────────
def T_full(t, n_tau=200, alpha=ALPHA_DEFAULT, beta=BETA_DEFAULT, v0=V0_DEFAULT,
           S0=S0_DEFAULT, wind_fn=default_wind,
           source_r=SOURCE_R_DEFAULT, source_c=SOURCE_C_DEFAULT, source_sigma=SOURCE_SIGMA_DEFAULT,
           domain_w=DOMAIN_W_DEFAULT, domain_h=DOMAIN_H_DEFAULT,
           ncol=NCOL_DEFAULT, nrow=NROW_DEFAULT):
    """
    T(x,y,t) = T_h(x,y,t) + T_p(x,y,t), for given coefficients and domain
    size/resolution. Each solved as its own independent problem; combined
    here only as the final step.
    Returns an (nrow, ncol) array.
    """
    T_h = T_homogeneous(t, alpha=alpha, beta=beta, v0=v0, S0=S0, wind_fn=wind_fn,
                         source_r=source_r, source_c=source_c, source_sigma=source_sigma,
                         domain_w=domain_w, domain_h=domain_h, ncol=ncol, nrow=nrow)
    T_p = T_particular(t, n_tau=n_tau, alpha=alpha, beta=beta, S0=S0, wind_fn=wind_fn,
                        source_r=source_r, source_c=source_c, source_sigma=source_sigma,
                        domain_w=domain_w, domain_h=domain_h, ncol=ncol, nrow=nrow)
    return T_h + T_p