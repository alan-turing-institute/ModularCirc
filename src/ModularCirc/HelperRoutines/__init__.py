"""
Wrapper module that attempts to import Cython version of HelperRoutines,
falling back to Numba version if Cython build is not available.

This provides transparent performance optimization without breaking existing code.
"""

import sys
import warnings

# Try to import the Cython-compiled version first
USING_CYTHON = False
try:
    from ModularCirc.HelperRoutines.HelperRoutinesCython import (
        resistor_model_flow,
        resistor_upstream_pressure,
        resistor_impedance_flux_rate,
        grounded_capacitor_model_pressure,
        grounded_capacitor_model_volume,
        grounded_capacitor_model_dpdt,
        chamber_volume_rate_change,
        relu_max,
        softplus,
        get_softplus_max,
        non_ideal_diode_flow,
        simple_bernoulli_diode_flow,
        maynard_valve_flow,
        maynard_phi_law,
        maynard_impedance_dqdt,
        leaky_diode_flow,
        activation_function_1,
        activation_function_2,
        activation_function_3,
        active_pressure_law,
        passive_pressure_law,
        active_dpdt_law,
        passive_dpdt_law,
        volume_from_pressure_nonlinear,
        time_shift,
        GenTimeShifter,
        compute_derivatives_batch,
        compute_derivatives_batch_indexed,
        bold_text,
        gen_total_dpdt_fixed,
    )
    USING_CYTHON = True
    if '--verbose' in sys.argv or True:  # Always show for now
        print("✓ Using Cythonized HelperRoutines (C-compiled, no JIT overhead)")
except ImportError as e:
    warnings.warn(f"ImportError: {e}", ImportWarning, stacklevel=2)
    # Fall back to original Numba implementation
    from .HelperRoutines import (
        resistor_model_flow,
        resistor_upstream_pressure,
        resistor_impedance_flux_rate,
        grounded_capacitor_model_pressure,
        grounded_capacitor_model_volume,
        grounded_capacitor_model_dpdt,
        chamber_volume_rate_change,
        chamber_volume_rate_change_vectorized,
        relu_max,
        softplus,
        get_softplus_max,
        non_ideal_diode_flow,
        simple_bernoulli_diode_flow,
        maynard_valve_flow,
        maynard_phi_law,
        maynard_impedance_dqdt,
        leaky_diode_flow,
        activation_function_1,
        activation_function_2,
        activation_function_3,
        active_pressure_law,
        passive_pressure_law,
        active_dpdt_law,
        passive_dpdt_law,
        volume_from_pressure_nonlinear,
        time_shift,
        time_shift_inplace,
    )
    
    warnings.warn(
        "Cython version of HelperRoutines not found. Using Numba version.\n"
        "To build Cython version for faster startup, run:\n"
        "  python setup_cython.py build_ext --inplace",
        ImportWarning,
        stacklevel=2
    )
    
# Re-export TimeClass from original module
from .HelperRoutines import TimeClass

from .HelperRoutines import bold_text

__all__ = [
    'USING_CYTHON',
    'TimeClass',
    'resistor_model_flow',
    'resistor_upstream_pressure',
    'resistor_impedance_flux_rate',
    'grounded_capacitor_model_pressure',
    'grounded_capacitor_model_volume',
    'grounded_capacitor_model_dpdt',
    'chamber_volume_rate_change',
    'relu_max',
    'softplus',
    'get_softplus_max',
    'non_ideal_diode_flow',
    'simple_bernoulli_diode_flow',
    'maynard_valve_flow',
    'maynard_phi_law',
    'maynard_impedance_dqdt',
    'leaky_diode_flow',
    'activation_function_1',
    'activation_function_2',
    'activation_function_3',
    'active_pressure_law',
    'passive_pressure_law',
    'active_dpdt_law',
    'passive_dpdt_law',
    'volume_from_pressure_nonlinear',
    'time_shift',
    'GenTimeShifter',
    'compute_derivatives_batch',
    'compute_derivatives_batch_indexed',
    'bold_text',
    'gen_total_dpdt_fixed',
]
