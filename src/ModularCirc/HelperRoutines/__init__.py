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
    print(e)
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
    
    # Provide a Python fallback for gen_total_dpdt_fixed to keep API parity
    def gen_total_dpdt_fixed(_af, E_act: float, v_ref: float, E_pas: float, k_pas: float):
        """Fallback generator for total dp/dt when Cython is unavailable.

        Mirrors the Cython implementation using the Numba-accelerated law functions.
        Returns a Python callable with signature: func(t: float, y: array) -> float.
        """
        def total_dpdt(t, y,
                       _af=_af,
                       E_act=E_act, v_ref=v_ref, E_pas=E_pas, k_pas=k_pas):
            _af_t = _af(t, dt=False)
            _af_dt = _af(t, dt=True)
            return (
                _af_dt * (active_pressure_law(t, y, E_act, v_ref) -
                          passive_pressure_law(t, y, E_pas, k_pas, v_ref))
                + _af_t * active_dpdt_law(t, y, E_act)
                + (1.0 - _af_t) * passive_dpdt_law(t, y, E_pas, k_pas, v_ref)
            )
        return total_dpdt
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
