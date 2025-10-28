"""
Optimized ComponentFactories without Numba JIT on closures.

Since HelperRoutines are now Cython-compiled, we don't need Numba JIT
on the thin wrapper closures. This eliminates JIT compilation overhead
during setup while maintaining performance since the actual work is done
in the pre-compiled Cython functions.
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple
from functools import partial
from ..HelperRoutines import (
    resistor_upstream_pressure, grounded_capacitor_model_dpdt,
    grounded_capacitor_model_pressure, grounded_capacitor_model_volume,
    resistor_model_flow, chamber_volume_rate_change, resistor_impedance_flux_rate,
    simple_bernoulli_diode_flow, non_ideal_diode_flow, maynard_valve_flow,
    maynard_impedance_dqdt, maynard_phi_law, time_shift,
    active_pressure_law, passive_pressure_law, active_dpdt_law, passive_dpdt_law,
    volume_from_pressure_nonlinear,
    activation_function_1, activation_function_2, activation_function_3,
    GenTimeShifter
)

# Cache compiled time shifter closures to avoid recompiling identical (delay, T)
_TIME_SHIFTER_CACHE: Dict[Tuple[float, float], object] = {}


class ComponentFunctionFactory:
    """Factory class for generating commonly used component functions."""
    
    @staticmethod
    def gen_resistor_upstream_pressure(r: float):
        """Generate resistor upstream pressure function."""
        def resistor_upstream_pressure_func(t, y):    
            return resistor_upstream_pressure(t, y, r=r)
        return resistor_upstream_pressure_func
    
    @staticmethod
    def gen_resistor_flow(r: float):
        """Generate resistor flow function."""
        return partial(resistor_model_flow, r=r)
    
    @staticmethod
    def gen_capacitor_dpdt(c: float):
        """Generate capacitor pressure derivative function."""
        return partial(grounded_capacitor_model_dpdt, c=c)
    
    @staticmethod
    def gen_capacitor_pressure(v_ref: float, c: float):
        """Generate capacitor pressure initialization function."""
        return partial(grounded_capacitor_model_pressure, v_ref=v_ref, c=c)
    
    @staticmethod
    def gen_capacitor_volume(v_ref: float, c: float):
        """Generate capacitor volume function."""
        return partial(grounded_capacitor_model_volume, v_ref=v_ref, c=c)

    @staticmethod
    def gen_impedance_flow_rate(r: float, l: float):
        """Generate resistor-impedance flow rate function."""
        return partial(resistor_impedance_flux_rate, r=r, l=l)

    @staticmethod
    def gen_simple_bernoulli_flow(CQ: float, RRA: float = 0.0):
        """Generate simple Bernoulli diode flow function."""
        return partial(simple_bernoulli_diode_flow, CQ=CQ, RRA=RRA)
    
    @staticmethod
    def gen_non_ideal_diode_flow(r: float, max_func):
        """Generate non-ideal diode flow function."""
        def func(t, y):
            dp = y[0] - y[1]
            dp = max_func(dp)
            return non_ideal_diode_flow(t, y=np.array([dp]), r=r)
        return func

    @staticmethod
    def gen_maynard_valve_flow(CQ: float, RRA: float = 0.0):
        """Generate Maynard valve flow function."""
        return partial(maynard_valve_flow, CQ=CQ, RRA=RRA)

    @staticmethod
    def gen_maynard_impedance_dqdt(CQ: float, RRA: float, L: float, R: float):
        """Generate Maynard impedance derivative function."""
        return partial(maynard_impedance_dqdt, CQ=CQ, RRA=RRA, L=L, R=R)    

    @staticmethod
    def gen_maynard_phi_law(Ko: float, Kc: float):
        """Generate Maynard phi law function."""
        return partial(maynard_phi_law, Ko=Ko, Kc=Kc)

    @staticmethod
    def gen_time_shifter(delay: float, T: float):
        """Generate time shifter function.
        
        GenTimeShifter is used instead of partial(time_shift, ...) because it provides
        better performance and avoids closure overhead for repeated (delay, T) pairs.
        """
        delay = 0.0 if np.isnan(delay) else delay
        return GenTimeShifter(shift=delay, tcycle=T)
        
    
    @staticmethod
    def gen_activation_function(af, time_shifter, **kwargs):
        """Generate activation function with time shifting."""
        # Pre-defined optimized functions for each activation function type
        if af is activation_function_1:
            # Extract parameters for activation_function_1
            t_max = kwargs.get('t_max')
            t_tr = kwargs.get('t_tr') 
            tau = kwargs.get('tau')

            def func(t, dt=False):
                shifted_t = time_shifter(t)
                return activation_function_1(shifted_t, t_max=t_max, t_tr=t_tr, tau=tau, dt=dt)
            return func
            
        elif af is activation_function_2:
            # Extract parameters for activation_function_2
            tr = kwargs.get('tr')
            td = kwargs.get('td')
            
            def func(t, dt=False):
                shifted_t = time_shifter(t)
                return activation_function_2(shifted_t, tr=tr, td=td, dt=dt)
            return func
            
        elif af is activation_function_3:
            # Extract parameters for activation_function_3
            tpwb = kwargs.get('tpwb')
            tpww = kwargs.get('tpww')
            
            def func(t, dt=False):
                shifted_t = time_shifter(t)
                return activation_function_3(shifted_t, tpwb=tpwb, tpww=tpww, dt=dt)
            return func
            
        else:
            # Fallback to original dynamic approach for unknown activation functions
            excluded_names = {'coeff', 't'}
            af_varnames = af.__code__.co_varnames
            kwargs2 = {k: v for k, v in kwargs.items()
                    if k in af_varnames and k not in excluded_names}

            def func(t, dt=False):
                return af(time_shifter(t), dt=dt, **kwargs2)
            return func


class ElastanceFactory:
    """Factory for heart chamber elastance functions."""
    
    @staticmethod
    def gen_constant_elastance(E_act: float, E_pas: float, af, v_ref: float):
        """Generate constant elastance functions."""
        # Pre-compute the difference for better performance
        E_diff = E_act - E_pas
        def comp_E(t):
            af_t = af(t)
            return af_t * E_diff + E_pas
        return comp_E
    
    @staticmethod
    def gen_constant_elastance_derivative(comp_E, eps: float = 1e-3):
        """Generate elastance derivative function."""
        # Pre-compute the division constant for better performance
        inv_2eps = 1.0 / (2.0 * eps)
        def comp_dEdt(t):
            return (comp_E(t + eps) - comp_E(t - eps)) * inv_2eps
        return comp_dEdt
    
    @staticmethod
    def gen_pressure_from_volume(comp_E, v_ref: float):
        """Generate pressure calculation from volume."""
        def func(t, y):
            return comp_E(t) * (y - v_ref)
        return func
    
    @staticmethod
    def gen_volume_from_pressure(comp_E, v_ref: float):
        """Generate volume calculation from pressure."""
        def func(t, y):
            return y / comp_E(t) + v_ref
        return func
    
    @staticmethod
    def gen_pressure_derivative(comp_E, comp_dEdt, v_ref: float):
        """Generate pressure time derivative."""
        def func(t, y):
            return comp_dEdt(t) * (y[0] - v_ref) + comp_E(t) * (y[1] - y[2])
        return func
    
    # Mixed elastance functions
    @staticmethod
    def gen_active_pressure(E_act: float, v_ref: float):
        """Generate active pressure function."""
        def func(t, y):
            return active_pressure_law(t, y, E_act=E_act, v_ref=v_ref)
        return func

    @staticmethod
    def gen_active_dpdt(E_act: float):
        """Generate active pressure derivative."""
        def func(t, y):
            return active_dpdt_law(t, y, E_act=E_act)
        return func

    @staticmethod
    def gen_passive_pressure(E_pas: float, k_pas: float, v_ref: float):
        """Generate passive pressure function."""
        def func(t, y):
            return passive_pressure_law(t, y, E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
        return func

    @staticmethod
    def gen_passive_dpdt(E_pas: float, k_pas: float, v_ref: float):
        """Generate passive pressure derivative."""
        def func(t, y):
            return passive_dpdt_law(t, y, E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
        return func
        
    @staticmethod
    def gen_total_pressure(_af, active_p, passive_p):
        """Generate total pressure function."""
        def func(t, y):
            return _af(t) * active_p(t, y) + (1.0 - _af(t)) * passive_p(t, y)
        return func
    
    @staticmethod
    def gen_total_dpdt(active_p, passive_p, _af, active_dpdt, passive_dpdt):
        """Generate total pressure derivative."""
        def func(t, y):
            dtact = _af(t, dt=True)
            act = _af(t)
            return (dtact * (active_p(t, y[0:1]) - passive_p(t, y[0:1])) +
                   act * active_dpdt(t, y) + (1.0 - act) * passive_dpdt(t, y))
        return func
    
    @staticmethod
    def gen_volume_from_pressure_nonlinear(E_pas: float, v_ref: float, k_pas: float):
        """Generate volume from pressure for nonlinear case."""
        def func(t, y):
            return volume_from_pressure_nonlinear(t, y, E_pas=E_pas, v_ref=v_ref, k_pas=k_pas)
        return func

    # Consolidated mixed elastance functions - eliminates redundant gen_*_fixed methods
    @staticmethod
    def gen_total_pressure_fixed(_af, E_act: float, v_ref: float, E_pas: float, k_pas: float):
        """Generate total pressure function directly using law functions."""
        def func(t, y):
            _af_t = _af(t, dt=False)
            # Use law functions directly - they extract y[0] internally
            active_val = active_pressure_law(t=0.0, y=y, E_act=E_act, v_ref=v_ref)
            passive_val = passive_pressure_law(t=0.0, y=y, E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
            return _af_t * active_val + (1.0 - _af_t) * passive_val
        return func
    
    @staticmethod
    def gen_total_dpdt_fixed(_af, E_act: float, v_ref: float, E_pas: float, k_pas: float):
        """Generate total pressure derivative function directly using law functions."""
        def func(t, y):
            _af_t = _af(t,dt=False)
            _d_af_dt = _af(t, dt=True)
            
            # Use law functions directly - they extract needed values internally
            active_p_val = active_pressure_law(t=0.0, y=y, E_act=E_act, v_ref=v_ref)
            passive_p_val = passive_pressure_law(t=0.0, y=y, E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
            
            # For derivatives, use the full y array (functions extract y[0], y[1], y[2] as needed)
            active_dpdt_val = active_dpdt_law(t=0.0, y=y, E_act=E_act)
            passive_dpdt_val = passive_dpdt_law(t=0.0, y=y, E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
            
            return (_d_af_dt * (active_p_val - passive_p_val) +
                   _af_t * active_dpdt_val +
                   (1. - _af_t) * passive_dpdt_val)
        return func
    
    # PP variants (pure passive component always included) - simplified
    @staticmethod
    def gen_total_pressure_pp(_af, E_act: float, v_ref: float, E_pas: float, k_pas: float):
        """Generate total pressure function for PP variant (passive always included)."""
        def func(t, y):
            _af_t = _af(t, dt=False)
            # Use law functions directly - they extract y[0] internally
            active_val = active_pressure_law(t=0.0, y=y, E_act=E_act, v_ref=v_ref)
            passive_val = passive_pressure_law(t=0.0, y=y, E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
            return _af_t * active_val + passive_val  # PP: passive always included
        return func
    
    @staticmethod
    def gen_total_dpdt_pp(_af, E_act: float, v_ref: float, E_pas: float, k_pas: float):
        """Generate total pressure derivative for PP variant."""
        def func(t, y):
            _af_t = _af(t, dt=False)
            _d_af_dt = _af(t, dt=True)
            
            # Use law functions directly - they extract needed values internally
            active_p_val = active_pressure_law(t=0.0, y=y, E_act=E_act, v_ref=v_ref)
            
            # For derivatives, use the full y array
            active_dpdt_val = active_dpdt_law(t=0.0, y=y, E_act=E_act)
            passive_dpdt_val = passive_dpdt_law(t=0.0, y=y, E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
            
            return (_d_af_dt * active_p_val + 
                   _af_t * active_dpdt_val + 
                   passive_dpdt_val)  # PP: passive dpdt always included
        return func


class ComponentSetupMixin:
    """Mixin providing common setup functionality."""
    
    def _validate_initial_conditions(self):
        """Validate that at least one initial condition is provided."""
        # More efficient validation - short-circuit evaluation
        has_v0 = (hasattr(self, 'v0') and self.v0 is not None and 
                  not (isinstance(self.v0, float) and np.isnan(self.v0)))
        has_p0 = (hasattr(self, 'p0') and self.p0 is not None and 
                  not (isinstance(self.p0, float) and np.isnan(self.p0)))
        
        if not (has_v0 or has_p0):
            raise ValueError("Solver needs at least the initial volume or pressure to be defined!")
    
    def _setup_volume_state_variable(self):
        """Standard volume state variable setup."""
        self._V.set_dudt_func(chamber_volume_rate_change,
                              function_name='chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in': self._Q_i.name,
                                      'q_out': self._Q_o.name}))
    
    def _setup_initial_conditions(self, p_init_func=None, v_init_func=None, 
                                  p_inputs=None, v_inputs=None):
        """Setup initial conditions with standard patterns."""
        # Pressure initialization
        if hasattr(self, 'p0') and (self.p0 is None or np.isnan(self.p0)):
            if p_init_func is not None:
                self._P_i.set_i_func(p_init_func, function_name=p_init_func.__name__)
                if p_inputs:
                    self._P_i.set_i_inputs(p_inputs)
        elif hasattr(self, 'p0') and self.p0 is not None:
            self._P_i._u.loc[0] = self.p0
        
        # Volume initialization  
        if hasattr(self, 'v0') and (self.v0 is None or np.isnan(self.v0)):
            if v_init_func is not None:
                self._V.set_i_func(v_init_func, function_name=v_init_func.__name__)
                if v_inputs:
                    self._V.set_i_inputs(v_inputs)
        elif hasattr(self, 'v0') and self.v0 is not None:
            self._V._u.loc[0] = self.v0
