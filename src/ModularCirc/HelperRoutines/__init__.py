"""
HelperRoutines module - supports both Cython and Numba implementations.

Environment variables:
  MODULARCIRC_FORCE_NUMBA=1  - Force use of Numba implementation even if Cython is available
  MODULARCIRC_FORCE_NUMBA=0  - Use Cython if available (default)
"""

import os
import sys
import warnings
import numpy as np
from functools import partial

# Check environment variable for implementation preference
force_numba = os.environ.get('MODULARCIRC_FORCE_NUMBA', '0').lower() in ('1', 'true', 'yes')

# Try to import the Cython-compiled version first (unless forced to use Numba)
USING_CYTHON = False
if not force_numba:
    try:
        from ModularCirc.HelperRoutines.HelperRoutinesCython import (
            resistor_model_flow,
            resistor_upstream_pressure,
            resistor_impedance_flux_rate,
            grounded_capacitor_model_pressure,
            grounded_nonlinear_capacitor_model_pressure,
            grounded_capacitor_model_volume,
            grounded_nonlinear_capacitor_model_volume,
            grounded_capacitor_model_dpdt,
            grounded_nonlinear_capacitor_model_dpdt,
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
            bold_text,
            compute_derivatives_batch,
            compute_derivatives_batch_indexed,
            gen_total_dpdt_fixed,
            starling_resistor_flow,
        )
        USING_CYTHON = True
        if '--verbose' in sys.argv or os.environ.get('MODULARCIRC_VERBOSE', '0') == '1':
            print("✓ Using Cythonized HelperRoutines (C-compiled, no JIT overhead)")
    except ImportError as e:
        if '--verbose' in sys.argv or os.environ.get('MODULARCIRC_VERBOSE', '0') == '1':
            print(f"  Cython import failed: {e}")
        force_numba = True  # Fall back to Numba

if force_numba or not USING_CYTHON:
    # Fall back to Numba implementation
    from .HelperRoutines import (
        resistor_model_flow,
        resistor_upstream_pressure,
        resistor_impedance_flux_rate,
        grounded_capacitor_model_pressure,
        grounded_nonlinear_capacitor_model_pressure,
        grounded_capacitor_model_volume,
        grounded_nonlinear_capacitor_model_volume,
        grounded_capacitor_model_dpdt,
        grounded_nonlinear_capacitor_model_dpdt,
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
        TimeClass,
        bold_text,
        compute_derivatives_batch,
        compute_derivatives_batch_indexed,
        GenTimeShifter,
        gen_total_dpdt_fixed,
        starling_resistor_flow,
    )
    if '--verbose' in sys.argv or os.environ.get('MODULARCIRC_VERBOSE', '0') == '1':
        if os.environ.get('MODULARCIRC_FORCE_NUMBA', '0') == '1':
            print("ℹ Using Numba HelperRoutines (forced via MODULARCIRC_FORCE_NUMBA)")
        else:
            print("⚠ Using Numba HelperRoutines (Cython not available)")
            print("  To build Cython version for faster startup, run:")
            print("  pip install -e .[performance]")
            print("  python setup.py build_ext --inplace")

# Always import these from the Numba version (not in Cython version or not imported there)
if USING_CYTHON:
    from .HelperRoutines import (
        TimeClass,
        time_shift_inplace,
        chamber_volume_rate_change_vectorized,
        GenTimeShifter,
    )



__all__ = [
    'USING_CYTHON',
    'TimeClass',
    'resistor_model_flow',
    'resistor_upstream_pressure',
    'resistor_impedance_flux_rate',
    'grounded_capacitor_model_pressure',
    'grounded_nonlinear_capacitor_model_pressure',
    'grounded_capacitor_model_volume',
    'grounded_nonlinear_capacitor_model_volume',
    'grounded_capacitor_model_dpdt',
    'grounded_nonlinear_capacitor_model_dpdt',
    'chamber_volume_rate_change',
    'chamber_volume_rate_change_vectorized',
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
    'time_shift_inplace',
    'bold_text',
    'compute_derivatives_batch',
    'compute_derivatives_batch_indexed',
    'GenTimeShifter',
    'gen_total_dpdt_fixed',
    'starling_resistor_flow',
]