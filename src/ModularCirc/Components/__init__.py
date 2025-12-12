from .ComponentBase import ComponentBase

# Import the optimized version if Cython HelperRoutines is available
# Otherwise fall back to the original Numba-JIT version
try:
    import ModularCirc.HelperRoutines.HelperRoutinesCython
    # Cython available - use optimized factories without Numba JIT overhead
    from ._ComponentFactoriesOptimized import ComponentFunctionFactory, ElastanceFactory
except ImportError:
    import ModularCirc.HelperRoutines.HelperRoutines
    # Cython not available - use original Numba version
    from ._ComponentFactories import ComponentFunctionFactory, ElastanceFactory

from .HC_constant_elastance import HC_constant_elastance
from .HC_mixed_elastance import HC_mixed_elastance
from .HC_mixed_elastance_pp import HC_mixed_elastance_pp
from .R_component import R_component
from .Rc_component import Rc_component
from .Rlc_component import Rlc_component
from .Valve_non_ideal import Valve_non_ideal
from .Valve_simple_bernoulli import Valve_simple_bernoulli
from .Valve_maynard import Valve_maynard
from .Rc_nonlinear_component import Rc_nonlinear_component
