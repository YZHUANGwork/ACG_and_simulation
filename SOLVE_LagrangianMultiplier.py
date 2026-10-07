"""
SOLUTION_LagrangeMultiplier.py

Solve grad f = lambda * grad g along a constraint g(x, y) = 0, by:
  1. tracing the curve with solve_ivp (walking its tangent direction),
  2. locating turning points of f along that trace as initial guesses, and
  3. polishing each guess into an exact solution of the augmented system
     [fx - lambda*gx, fy - lambda*gy, g] = 0 via fsolve -- the standard
     textbook approach (Cornell, Notre Dame, Berkeley, Purdue all solve
     this same 3-equation, 3-unknown system for x, y, and lambda together).
"""

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (enables 3D projection)


class LagrangeMultiplier:

    def __init__(self, f, g, h=1e-6):
        self.f = f
        self.g = g
        self.h = h           # step size for finite-difference gradients
        self.path = None
        self.critical_points = None

    def grad(self, func, x, y):
        """Numerical gradient of `func` at (x, y) via central differences."""
        h = self.h
        fx = (func(x + h, y) - func(x - h, y)) / (2 * h)
        fy = (func(x, y + h) - func(x, y - h)) / (2 * h)
        return fx, fy

    def tangent(self, x, y):
        """Unit tangent to the curve g(x, y) = 0 at (x, y)."""
        gx, gy = self.grad(self.g, x, y)
        t = np.array([-gy, gx])
        return t / np.linalg.norm(t)

    def _ode(self, s, state):
        x, y = state
        return self.tangent(x, y)

    def trace_path(self, x0, y0, s_max, n=500):
        """Trace g(x, y) = 0 starting at (x0, y0) for arc length s in [0, s_max]."""
        sol = solve_ivp(self._ode, [0, s_max], [x0, y0],
                         dense_output=True, max_step=0.05, rtol=1e-8)
        s = np.linspace(0, s_max, n)
        xs, ys = sol.sol(s)
        zs = np.array([self.f(x, y) for x, y in zip(xs, ys)])
        self.path = {'s': s, 'x': xs, 'y': ys, 'z': zs}
        return self.path

    def lagrange_multiplier(self, x, y):
        """Least-squares lambda such that grad f = lambda * grad g, at (x, y)."""
        fx, fy = self.grad(self.f, x, y)
        gx, gy = self.grad(self.g, x, y)
        return (fx * gx + fy * gy) / (gx ** 2 + gy ** 2)

    def find_critical_points(self):
        """
        Points along the traced path where grad f = lambda * grad g holds.
        A quadratic fit around each sign change in f gives a good initial
        guess; fsolve then polishes (x, y, lambda) to machine precision on
        the augmented system -- the standard textbook solve.
        """
        if self.path is None:
            raise RuntimeError("call trace_path() first")
        xs, ys, zs = self.path['x'], self.path['y'], self.path['z']

        points = []
        for i in range(1, len(zs) - 1):
            # <= 0, not < 0: a perfectly symmetric f and a constraint that
            # passes exactly through its centerline (e.g. a horizontal-line
            # constraint straight through the summit) can land two adjacent
            # samples on an EXACTLY equal f value right at the peak, which
            # a strict < 0 test misses entirely (0 is not < 0).
            if (zs[i] - zs[i - 1]) * (zs[i + 1] - zs[i]) <= 0 and \
               (zs[i] != zs[i - 1] or zs[i] != zs[i + 1]):
                y0, y1, y2 = zs[i - 1], zs[i], zs[i + 1]
                denom = (y0 - 2 * y1 + y2)
                delta = 0.5 * (y0 - y2) / denom if denom != 0 else 0.0
                delta = np.clip(delta, -1.0, 1.0)

                if delta >= 0:
                    xr = xs[i] + delta * (xs[i + 1] - xs[i])
                    yr = ys[i] + delta * (ys[i + 1] - ys[i])
                else:
                    xr = xs[i] + delta * (xs[i] - xs[i - 1])
                    yr = ys[i] + delta * (ys[i] - ys[i - 1])

                lam = self.lagrange_multiplier(xr, yr)

                def augmented_system(v):
                    xv, yv, lv = v
                    fx, fy = self.grad(self.f, xv, yv)
                    gx, gy = self.grad(self.g, xv, yv)
                    return [fx - lv * gx, fy - lv * gy, self.g(xv, yv)]

                xr, yr, lam = fsolve(augmented_system, [xr, yr, lam])
                fr = self.f(xr, yr)

                kind = 'max' if zs[i] > zs[i - 1] else 'min'

                # a tie (zs[i] == zs[i-1] or zs[i] == zs[i+1]) can fire the
                # <= 0 test on two adjacent indices for the SAME plateau --
                # skip it if it's essentially the same point as the one
                # just found (within one sample's worth of (x, y))
                if points:
                    prev = points[-1]
                    if np.hypot(xr - prev['x'], yr - prev['y']) < 1e-6:
                        continue

                points.append({'x': xr, 'y': yr, 'f': fr,
                               'lambda': lam, 'kind': kind})
        self.critical_points = points
        return points

