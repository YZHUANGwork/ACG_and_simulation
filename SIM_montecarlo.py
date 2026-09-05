import numpy as np
from scipy.interpolate import interp1d


def linear_intp(x_test, originalx, originaly):
    try:
        interp_start = originalx[originalx < x_test].max()
        interp_end = originalx[originalx > x_test].min()
        i = list(originalx).index(interp_start)
        j = list(originalx).index(interp_end)
        new_y = interp1d([originalx[i], originalx[j]],
                          [originaly[i], originaly[j]],
                          kind='linear')(x_test)
    except ValueError:
        new_y = -1
    return new_y
 
 
class PDF_montecarlo:
    def __init__(self, x_y_, N, simulate_type='pdf', seed=1, spacing='linear',_cdf = None, _norm = None, 
                 pdf_func_2d=None, bounds_2d=None, pdf_max_2d=None):
        self.x = np.asarray(x_y_[0], dtype=float)
        self.pdf = np.asarray(x_y_[1], dtype=float)
        self.N = N
        self.simulate_type = simulate_type
        self.seed = seed
        self.spacing = spacing
        np.random.seed(self.seed)
 
        # used by pdf_sim_2d when called with no arguments (e.g. via run_sim)
        self.pdf_func_2d = pdf_func_2d
        self.bounds_2d = bounds_2d
        self.pdf_max_2d = pdf_max_2d
 
        self._cdf = _cdf
        self._norm = _norm
 
    def linear_Itp_pdf(self, low, high, binnum):
        original_x, original_y = self.x, self.pdf
        if self.spacing == 'log':
            x_itp = np.logspace(np.log10(low), np.log10(high), binnum)
        else:
            x_itp = np.linspace(low, high, binnum)
 
        mask = x_itp >= original_x.min()
        y_itp = [linear_intp(x_test, original_x, original_y)
                 for x_test in x_itp[mask]]
        if y_itp[0] < 0:
            y_itp[0] = 0
 
        y_below = np.ones(x_itp[~mask].shape) * y_itp[0]
        ys = np.concatenate([y_below, y_itp])
        return x_itp, ys
 
    def pdf2cdf(self):
        dx = np.diff(self.x)
        cdf = np.cumsum(self.pdf[:-1] * dx)
        self._cdf, self._norm = cdf, cdf[-1]
        return cdf, self._norm
 
    def pdf_sim(self):
        xMin, xMax = self.x.min(), self.x.max()
        yMax = self.pdf.max() * 1.001
 
        x_sim = np.empty(self.N)
        y_sim = np.empty(self.N)
        for k in range(self.N):
            rdm_x = xMin + (xMax - xMin) * np.random.uniform()
            rdm_y = yMax * np.random.uniform()
            while rdm_y > linear_intp(rdm_x, self.x, self.pdf):
                rdm_x = xMin + (xMax - xMin) * np.random.uniform()
                rdm_y = yMax * np.random.uniform()
            x_sim[k] = rdm_x
            y_sim[k] = rdm_y
        return x_sim, y_sim
 
    def cdf_sim(self):
        if self._cdf is None:
            self.pdf2cdf()
        f_cdf = interp1d(self._cdf / self._norm, self.x[:-1], bounds_error=False,
                          fill_value=(self.x[0], self.x[-2]))
 
        cdf_draws = np.random.uniform(size=self.N)
        x_sim = f_cdf(cdf_draws)
        pdf_sim_vals = np.array([linear_intp(xv, self.x, self.pdf) for xv in x_sim])
        return x_sim, cdf_draws, pdf_sim_vals
 
    def pdf_sim_2d(self, pdf_func=None, bounds=None, pdf_max=None):
        """
        accept-reject sampling for a 2D pdf -- e.g. a circle,
        oval, or any region defined by a piecewise function of (x, y).
 
        pdf_func : callable(x, y) -> density. Return 0 outside the shape
                   and >0 inside it (e.g. for a uniform-density ellipse,
                   return 1 if (x/a)**2 + (y/b)**2 <= 1 else 0).
                   Falls back to self.pdf_func_2d if not given.
        bounds   : (xmin, xmax, ymin, ymax) bounding box to draw from --
                   for a circle/oval this is just its bounding square/rect.
                   Falls back to self.bounds_2d if not given.
        pdf_max  : max value of pdf_func over the box. If None, falls
                   back to self.pdf_max_2d, then estimates it from a grid.
        """
        pdf_func = pdf_func if pdf_func is not None else self.pdf_func_2d
        bounds = bounds if bounds is not None else self.bounds_2d
        pdf_max = pdf_max if pdf_max is not None else self.pdf_max_2d
        if pdf_func is None or bounds is None:
            raise ValueError("pdf_sim_2d needs pdf_func and bounds, either "
                              "passed directly or set on the instance.")
 
        xmin, xmax, ymin, ymax = bounds
        if pdf_max is None:
            gx, gy = np.meshgrid(np.linspace(xmin, xmax, 200),
                                  np.linspace(ymin, ymax, 200))
            pdf_max = np.max(pdf_func(gx, gy)) * 1.001
 
        x_sim = np.empty(self.N)
        y_sim = np.empty(self.N)
        for k in range(self.N):
            rx = xmin + (xmax - xmin) * np.random.uniform()
            ry = ymin + (ymax - ymin) * np.random.uniform()
            rz = pdf_max * np.random.uniform()
            while rz > pdf_func(rx, ry):
                rx = xmin + (xmax - xmin) * np.random.uniform()
                ry = ymin + (ymax - ymin) * np.random.uniform()
                rz = pdf_max * np.random.uniform()
            x_sim[k] = rx
            y_sim[k] = ry
        return x_sim, y_sim
    def run_sim(self):
        if self.simulate_type == 'pdf':
            return self.pdf_sim()
        elif self.simulate_type == 'cdf':
            return self.cdf_sim()
        elif self.simulate_type == 'pdf2d':
            return self.pdf_sim_2d()
        else:
            raise ValueError(f"Unknown simulate_type: {self.simulate_type!r}")
 