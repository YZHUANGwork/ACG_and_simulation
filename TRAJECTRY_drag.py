import numpy as np
from scipy.integrate import solve_ivp
import astropy.units as u
import astropy.constants as const

# dimension:
length_UNIT = u.m
mass_UNIT   = u.kg
time_UNIT   = u.s
vel_UNIT    = length_UNIT / time_UNIT
accel_UNIT  = length_UNIT / time_UNIT**2
force_UNIT  = mass_UNIT * length_UNIT / time_UNIT**2       # Newton
drag1_UNIT  = force_UNIT * time_UNIT / length_UNIT         # N·s/m
drag2_UNIT  = force_UNIT * time_UNIT**2 / length_UNIT**2   # N·s²/m²

# ── PARAMETERS ───────────────────────────────────────────────────────────────
RHO_AIR = (1.2   * u.kg / u.m**3).to(mass_UNIT / length_UNIT**3)
CD      = 1.0        # drag coefficient, dimensionless
MU_AIR  = (1.8e-5 * u.Pa * u.s).to(drag1_UNIT / length_UNIT)  # Pa·s = kg/(m·s)
T_SPAWN = 20.0       # s  (plain float)
T_MAX   = 25.0       # s
DT      = 0.1
speed    = 1.0
interval = DT * 1000 / speed
FPS         = round(1.0 / DT)


g_vec  = (const.g0 * np.array([0.0, 0.0, -1.])).to(accel_UNIT).value  # (3,)


def make_wind_vec(speed_qty, rng):
    # horizontal components drawn from wider range, vertical suppressed
    # -> tends horizontal but vertical not forbidden
    xy = rng.uniform(-1, 1, 2).astype(float)
    z  = rng.uniform(-0.2, 0.2)   # vertical component much smaller
    xyz = np.array([xy[0], xy[1], z])
    xyz /= np.linalg.norm(xyz)
    return xyz * speed_qty.to(vel_UNIT).value


def wind_vec(t):
    # returns plain float (3,) array in m/s
    for t_on, t_off, W_vec in wind_segments:
        if t_on <= t <= t_off:
            return W_vec
    return np.zeros(3)


# ── EOM ──────────────────────────────────────────────────────────────────────
def eom(t, X):
    # INPUT: X = [r_vec(3N), v_vec(3N)] plain floats, m and m/s
    r_vec = X[0:3*N].reshape(3, N)   # (3, N) m
    v_vec = X[3*N:  ].reshape(3, N)  # (3, N) m/s

    active = (t >= t_spawn) & (r_vec[2] > 0)     # (N,) bool
    v_eff  = v_vec * active[None, :]              # (3, N) m/s, zero for inactive

    vrel_vec  = v_eff - wind_vec(t)[:, None]      # (3, N) m/s
    vrel_mag  = np.linalg.norm(vrel_vec, axis=0)  # (N,)   m/s
    vrel_mag2 = vrel_mag**2                       # (N,)   m²/s²
    vrel_hat  = vrel_vec / np.where(vrel_mag > 0, vrel_mag, 1.0)  # (3, N) dimensionless

    # Force (N = kg·m/s²):
    #   gravity        M[kg] * g[m/s²]                   -> (3,N) N
    #   linear drag   -K1[N·s/m]  * vrel[m/s]            -> (3,N) N
    #   quad drag     -K2[N·s²/m²]* |vrel|²[m²/s²]*vhat  -> (3,N) N
    F_vec = (M_val[None, :] * g_vec[:, None]
             - K1_val[None, :] * vrel_vec
             - K2_val[None, :] * vrel_mag2 * vrel_hat)   # (3, N) N

    a_vec = F_vec / M_val[None, :]                       # (3, N) m/s²

    dv_vec_dt = np.where(active, a_vec, 0.0)             # (3, N) m/s²
    dr_vec_dt = np.where(active, v_eff, 0.0)             # (3, N) m/s

    # OUTPUT: plain floats — solve_ivp requires no units
    return np.concatenate([dr_vec_dt.ravel(), dv_vec_dt.ravel()])


# ── SOLVE ─────────────────────────────────────────────────────────────────────
def main(N_in=100, seed=42):
    """Run the petal-fall EOM and return the raw solve_ivp result (sol)."""
    global N, rng, t_spawn, M_val, K1_val, K2_val, wind_segments
    global sizes, x0, y0, z0, vx0, vy0, vz0, X_init

    N   = N_in
    rng = np.random.default_rng(seed)

    # ── INITIAL CONDITIONS ────────────────────────────────────────────────
    # size drives marker area, diameter, mass, and drag coefficients
    sizes  = rng.uniform(10, 20, N)                                        # marker size (pts²) — plain, for scatter
    D_rdm  = (np.interp(sizes, [10, 20], [5, 20])  * u.mm).to(length_UNIT)  # diameter: 5–20 mm -> m
    A_rdm  = (np.pi * (D_rdm / 2)**2).to(length_UNIT**2)                    # frontal area m²
    M_rdm  = (np.interp(sizes, [10, 20], [0.1, 0.5]) * u.g).to(mass_UNIT)  # mass: 0.1–0.5 g -> kg
    K1_rdm = (3 * np.pi * MU_AIR * D_rdm).to(drag1_UNIT)                  # Stokes linear drag N·s/m
    K2_rdm = (0.5 * RHO_AIR * CD * A_rdm).to(drag2_UNIT)                  # quadratic drag N·s²/m²

    # strip per-petal arrays to plain floats in base units — used inside eom
    M_val  = M_rdm.value    # (N,) kg
    K1_val = K1_rdm.value   # (N,) N·s/m
    K2_val = K2_rdm.value   # (N,) N·s²/m²

    # initial positions — plain floats in metres
    initial_height = 5.0 * u.m
    x0      = rng.uniform(0, 3, N)
    y0      = rng.uniform(1, 10, N)
    z0      = np.full(N, initial_height.to(length_UNIT).value)

    # initial velocities — zero, explicitly in vel_UNIT then stripped
    vx0     = (np.zeros(N) * vel_UNIT).to(vel_UNIT).value
    vy0     = (np.zeros(N) * vel_UNIT).to(vel_UNIT).value
    vz0     = (np.zeros(N) * vel_UNIT).to(vel_UNIT).value

    t_spawn = rng.uniform(0, T_SPAWN, N)   # s, plain float

    # state vector: [x(N), y(N), z(N), vx(N), vy(N), vz(N)] — plain floats, SI
    X_init  = np.concatenate([x0, y0, z0, vx0, vy0, vz0])

    wind_segments = [
        (1.0,  1.5,  make_wind_vec(rng.uniform(18, 19) * u.m/u.s, rng)),
        (2.0,  3.0,  make_wind_vec(rng.uniform(4, 5) * u.m/u.s, rng)),
        (3.0,  5.0,  make_wind_vec(rng.uniform(3, 5) * u.m/u.s, rng)),
        (7.0,  9.0,  make_wind_vec(rng.uniform(4, 5) * u.m/u.s, rng)),
        (12.0, 15.0, make_wind_vec(rng.uniform(1, 2) * u.m/u.s, rng)),
        (18.0, 20.0, make_wind_vec(rng.uniform(4, 5) * u.m/u.s, rng)),
    ]

    print(f"Solving EOM for {N} petals (seed={seed}) ...")
    t_eval = np.arange(0, T_MAX, DT)
    sol = solve_ivp(
        eom,
        t_span=(0, T_MAX),
        y0=X_init,
        method='RK45',
        t_eval=t_eval,
        rtol=1e-3,
        atol=1e-5,
    )
    print(f"Done — {sol.t.shape[0]} steps, t_end={sol.t[-1]:.1f}s")
    return sol


if __name__ == "__main__":
    main()