import jax
import jax.numpy as jnp
from jax.experimental.ode import odeint
from .BaseSolver import BaseSolver

class JaxSolver(BaseSolver):
    def solve(self, fun, y0, t_span, t_eval, **kwargs):
        def wrapped_fun(y, t):
            return fun(t, y)

        y0 = jnp.array(y0)
        t_eval = jnp.array(t_eval)
        result = odeint(wrapped_fun, y0, t_eval)
        return jax.device_get(result)