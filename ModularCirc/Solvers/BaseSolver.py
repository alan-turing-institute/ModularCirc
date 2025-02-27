class BaseSolver:
    def solve(self, fun, y0, t_span, t_eval, **kwargs):
        raise NotImplementedError("This method should be implemented by subclasses")