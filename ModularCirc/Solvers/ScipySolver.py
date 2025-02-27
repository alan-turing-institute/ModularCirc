from scipy.integrate import solve_ivp
from .BaseSolver import BaseSolver

class ScipySolver(BaseSolver):
    def solve(self, fun, y0, t_span, t_eval, **kwargs):
        return solve_ivp(fun=fun, y0=y0, t_span=t_span, t_eval=t_eval, **kwargs)