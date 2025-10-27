#!/usr/bin/env python3
"""
Compare outputs of ModularCirc.HelperRoutines.HelperRoutinesCython (Cython)
vs ModularCirc.HelperRoutines.HelperRoutines (Numba/Python) across a suite of
functions and representative inputs.

Run from repository root:
  python sandbox/compare_helperroutines_impls.py

Exit code 0 if all comparisons within tolerance, non-zero otherwise.
"""
from __future__ import annotations

import math
import sys
import traceback
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

# Ensure reproducibility for any randomized values (if added later)
np.random.seed(42)

# Import the active package namespace (will be Cython-backed if available)
try:
    import ModularCirc.HelperRoutines as cy  # package __init__ re-exports functions
except Exception as e:
    print("Could not import ModularCirc.HelperRoutines package:", e)
    sys.exit(2)

# Import the pure-Python submodule explicitly (Numba-backed implementations)
import importlib
try:
    py = importlib.import_module('ModularCirc.HelperRoutines.HelperRoutines')
except Exception as e:
    print("Could not import pure-Python HelperRoutines submodule:", e)
    sys.exit(2)

# Numeric comparison tolerances
ATOL = 1e-10
RTOL = 1e-10


def almost_equal(a: float, b: float, atol: float = ATOL, rtol: float = RTOL) -> bool:
    if math.isfinite(a) and math.isfinite(b):
        return abs(a - b) <= atol + rtol * max(abs(a), abs(b))
    return a == b


def compare_scalar(func_name: str, cy_val: float, py_val: float) -> Tuple[bool, str]:
    ok = almost_equal(cy_val, py_val)
    msg = "" if ok else f"Mismatch {func_name}: cy={cy_val} py={py_val}"
    return ok, msg


def run_case(
    name: str,
    cy_func: Callable[..., Any],
    py_func: Callable[..., Any],
    args: Tuple[Any, ...] = (),
    kwargs: Dict[str, Any] | None = None,
    is_string: bool = False,
) -> Tuple[bool, str]:
    kwargs = kwargs or {}
    try:
        cy_out = cy_func(*args, **kwargs)
        py_out = py_func(*args, **kwargs)
        print(f"Outputs for {name}:\n Cython: {cy_out}\n Python: {py_out}")
        print(type(cy_func), type(py_func))
        if is_string:
            ok = cy_out == py_out
            return ok, ("" if ok else f"Mismatch {name}: cy={cy_out!r} py={py_out!r}")
        else:
            return compare_scalar(name, float(cy_out), float(py_out))
    except Exception:
        return False, f"Exception in {name}:\n" + traceback.format_exc()


def main() -> int:
    failures: List[str] = []
    total = 0

    def check(name: str, cy_f: Callable[..., Any], py_f: Callable[..., Any], *args, **kwargs):
        nonlocal total
        total += 1
        print(f"Comparing function: {name}")
        ok, msg = run_case(name, cy_f, py_f, args=args, kwargs=kwargs)
        if not ok:
            failures.append(msg)

    # Basic resistor/impedance/capacitor models
    check("resistor_model_flow", cy.resistor_model_flow, py.resistor_model_flow, 0.0, np.array([100.0, 95.0], dtype=np.float64), 2.0)
    check("resistor_upstream_pressure", cy.resistor_upstream_pressure, py.resistor_upstream_pressure, 0.0, np.array([5.0, 90.0], dtype=np.float64), 2.0)
    check("resistor_impedance_flux_rate", cy.resistor_impedance_flux_rate, py.resistor_impedance_flux_rate, 0.0, np.array([100.0, 90.0, 2.0], dtype=np.float64), 2.0, 0.5)

    check("grounded_capacitor_model_pressure", cy.grounded_capacitor_model_pressure, py.grounded_capacitor_model_pressure, 0.0, np.array([12.0], dtype=np.float64), 10.0, 2.0)
    check("grounded_capacitor_model_volume", cy.grounded_capacitor_model_volume, py.grounded_capacitor_model_volume, 0.0, np.array([1.0], dtype=np.float64), 10.0, 2.0)
    check("grounded_capacitor_model_dpdt", cy.grounded_capacitor_model_dpdt, py.grounded_capacitor_model_dpdt, 0.0, np.array([5.0, 3.0], dtype=np.float64), 2.0)
    check("chamber_volume_rate_change", cy.chamber_volume_rate_change, py.chamber_volume_rate_change, 0.0, np.array([5.0, 3.0], dtype=np.float64))

    # Diodes and valves
    check("non_ideal_diode_flow_pos", cy.non_ideal_diode_flow, py.non_ideal_diode_flow, 0.0, np.array([2.0], dtype=np.float64), 3.0)
    check("non_ideal_diode_flow_neg", cy.non_ideal_diode_flow, py.non_ideal_diode_flow, 0.0, np.array([-2.0], dtype=np.float64), 3.0)

    check("simple_bernoulli_diode_flow_fwd", cy.simple_bernoulli_diode_flow, py.simple_bernoulli_diode_flow, 0.0, np.array([100.0, 90.0], dtype=np.float64), 1.3, 0.1)
    check("simple_bernoulli_diode_flow_bwd", cy.simple_bernoulli_diode_flow, py.simple_bernoulli_diode_flow, 0.0, np.array([90.0, 100.0], dtype=np.float64), 1.3, 0.1)

    check("maynard_valve_flow", cy.maynard_valve_flow, py.maynard_valve_flow, 0.0, np.array([100.0, 90.0, 0.5], dtype=np.float64), 1.3, 0.1)
    check("maynard_phi_law_open", cy.maynard_phi_law, py.maynard_phi_law, 0.0, np.array([100.0, 90.0, 0.5], dtype=np.float64), 0.01, 0.02)
    check("maynard_phi_law_close", cy.maynard_phi_law, py.maynard_phi_law, 0.0, np.array([90.0, 100.0, 0.5], dtype=np.float64), 0.01, 0.02)
    check("maynard_impedance_dqdt", cy.maynard_impedance_dqdt, py.maynard_impedance_dqdt, 0.0, np.array([100.0, 90.0, 5.0, 0.5], dtype=np.float64), 1.3, 0.2, 0.05, 0.1)

    # Leaky diode (direct scalar signature)
    check("leaky_diode_flow_fwd", cy.leaky_diode_flow, py.leaky_diode_flow, 100.0, 90.0, 1.5, 5.0)
    check("leaky_diode_flow_bwd", cy.leaky_diode_flow, py.leaky_diode_flow, 90.0, 100.0, 1.5, 5.0)

    # Activations
    check("activation_function_1_val", cy.activation_function_1, py.activation_function_1, 100.0, 300.0, 200.0, 120.0, False)
    check("activation_function_1_dt", cy.activation_function_1, py.activation_function_1, 100.0, 300.0, 200.0, 120.0, True)

    check("activation_function_2_val", cy.activation_function_2, py.activation_function_2, 100.0, 150.0, 300.0, False)
    check("activation_function_2_dt", cy.activation_function_2, py.activation_function_2, 100.0, 150.0, 300.0, True)

    check("activation_function_3_val", cy.activation_function_3, py.activation_function_3, 100.0, 80.0, 120.0, False)
    check("activation_function_3_dt", cy.activation_function_3, py.activation_function_3, 100.0, 80.0, 120.0, True)

    # Pressure laws and their derivatives
    check("active_pressure_law", cy.active_pressure_law, py.active_pressure_law, 0.0, np.array([120.0], dtype=np.float64), 2.0, 100.0)
    check("passive_pressure_law", cy.passive_pressure_law, py.passive_pressure_law, 0.0, np.array([120.0], dtype=np.float64), 2.0, 0.01, 100.0)

    check("active_dpdt_law", cy.active_dpdt_law, py.active_dpdt_law, 0.0, np.array([120.0, 8.0, 5.0], dtype=np.float64), 2.0)
    check("passive_dpdt_law", cy.passive_dpdt_law, py.passive_dpdt_law, 0.0, np.array([120.0, 8.0, 5.0], dtype=np.float64), 2.0, 0.01, 100.0)

    # Nonlinear volume from pressure
    check("volume_from_pressure_nonlinear", cy.volume_from_pressure_nonlinear, py.volume_from_pressure_nonlinear, 0.0, np.array([15.0], dtype=np.float64), 2.0, 100.0, 0.01)

    # Time shift
    check("time_shift_basic", cy.time_shift, py.time_shift, 950.0, 80.0, 1000.0)

    # Relu / softplus
    check("relu_max_pos", cy.relu_max, py.relu_max, 3.0)
    check("relu_max_neg", cy.relu_max, py.relu_max, -3.0)
    check("softplus_basic", cy.softplus, py.softplus, -3.5, 0.2)

    # get_softplus_max returns a callable; compare outputs for a few values
    try:
        cy_sp = cy.get_softplus_max(0.2)
        py_sp = py.get_softplus_max(0.2)
        for val in [-5.0, -1.0, 0.0, 1.0, 5.0]:
            ok, msg = compare_scalar(f"get_softplus_max({val})", float(cy_sp(val)), float(py_sp(val)))
            total += 1
            if not ok:
                failures.append(msg)
    except Exception:
        failures.append("Exception creating/testing get_softplus_max:\n" + traceback.format_exc())
        total += 1

    # bold_text (string)
    ok, msg = run_case("bold_text", cy.bold_text, py.bold_text, args=("hello",), is_string=True)
    total += 1
    if not ok:
        failures.append(msg)

    # Summary
    print("Compared functions:", total)
    if failures:
        print("\nFailures (", len(failures), "):", sep="")
        for f in failures:
            print(" -", f)
        return 1
    else:
        print("All comparisons within tolerance (atol=", ATOL, ", rtol=", RTOL, ").", sep="")
        return 0


if __name__ == "__main__":
    sys.exit(main())
