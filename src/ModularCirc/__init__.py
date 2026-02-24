from ._BatchRunner import _BatchRunner as BatchRunner

# Try to use Cython-compiled HelperRoutines for better performance
try:
    from .HelperRoutines import HelperRoutinesCython as HelperRoutines
    _USING_CYTHON = True
except ImportError:
    from . import HelperRoutines
    _USING_CYTHON = False
