"""
Common function factories for component generation.
This module eliminates code duplication by providing reusable function generators.
"""

import numpy as np
import pandas as pd
from ..HelperRoutines import (
    resistor_upstream_pressure, grounded_capacitor_model_dpdt,
    grounded_capacitor_model_pressure, grounded_capacitor_model_volume,
    resistor_model_flow, chamber_volume_rate_change, resistor_impedance_flux_rate,
    simple_bernoulli_diode_flow, non_ideal_diode_flow, maynard_valve_flow,
    maynard_impedance_dqdt, maynard_phi_law, time_shift, activation_function_1
)


class ComponentFunctionFactory:
    """Factory class for generating commonly used component functions."""
    
    @staticmethod
    def gen_resistor_upstream_pressure(r: float):
        """Generate resistor upstream pressure function."""
        def func(t, y):
            return resistor_upstream_pressure(t, y=y, r=r)
        return func
    
    @staticmethod
    def gen_resistor_flow(r: float):
        """Generate resistor flow function."""
        def func(t, y):
            return resistor_model_flow(t=t, y=y, r=r)
        return func
    
    @staticmethod
    def gen_capacitor_dpdt(c: float):
        """Generate capacitor pressure derivative function."""
        def func(t, y):
            return grounded_capacitor_model_dpdt(t, y=y, c=c)
        return func
    
    @staticmethod
    def gen_capacitor_pressure(v_ref: float, c: float):
        """Generate capacitor pressure initialization function."""
        def func(t, y):
            return grounded_capacitor_model_pressure(t, y=y, v_ref=v_ref, c=c)
        return func
    
    @staticmethod
    def gen_capacitor_volume(v_ref: float, c: float):
        """Generate capacitor volume function."""
        def func(t, y):
            return grounded_capacitor_model_volume(t, y=y, v_ref=v_ref, c=c)
        return func
    
    @staticmethod
    def gen_impedance_flow_rate(r: float, l: float):
        """Generate resistor-impedance flow rate function."""
        def func(t, y):
            return resistor_impedance_flux_rate(t, y=y, r=r, l=l)
        return func
    
    @staticmethod
    def gen_simple_bernoulli_flow(CQ: float, RRA: float = 0.0):
        """Generate simple Bernoulli diode flow function."""
        def func(t, y):
            return simple_bernoulli_diode_flow(t, y=y, CQ=CQ, RRA=RRA)
        return func
    
    @staticmethod
    def gen_non_ideal_diode_flow(r: float, max_func):
        """Generate non-ideal diode flow function."""
        def func(t, y):
            return non_ideal_diode_flow(t, y=y, r=r, max_func=max_func)
        return func
    
    @staticmethod
    def gen_maynard_valve_flow(CQ: float, RRA: float = 0.0):
        """Generate Maynard valve flow function."""
        def func(t, y):
            return maynard_valve_flow(t, y=y, CQ=CQ, RRA=RRA)
        return func
    
    @staticmethod
    def gen_maynard_impedance_dqdt(CQ: float, RRA: float, L: float, R: float):
        """Generate Maynard impedance derivative function."""
        def func(t, y):
            return maynard_impedance_dqdt(t, y=y, CQ=CQ, RRA=RRA, L=L, R=R)
        return func
    
    @staticmethod
    def gen_maynard_phi_law(Ko: float, Kc: float):
        """Generate Maynard phi law function."""
        def func(t, y):
            return maynard_phi_law(t, y=y, Ko=Ko, Kc=Kc)
        return func
    
    @staticmethod
    def gen_time_shifter(delay: float, T: float):
        """Generate time shifter function."""
        def func(t):
            return time_shift(t, delay, T)
        return func
    
    @staticmethod
    def gen_activation_function(af, time_shifter, **kwargs):
        """Generate activation function with parameters."""
        varnames = [name for name in af.__code__.co_varnames if name not in ['coeff', 't']]
        kwargs2 = {key: val for key, val in kwargs.items() if key in varnames}
        
        def func(t, dt=False):
            return af(time_shifter(t), dt=dt, **kwargs2)
        return func


class ElastanceFactory:
    """Factory for heart chamber elastance functions."""
    
    @staticmethod
    def gen_constant_elastance(E_act: float, E_pas: float, af, v_ref: float):
        """Generate constant elastance functions."""
        comp_E = lambda t: af(t) * E_act + (1.0 - af(t)) * E_pas
        return comp_E
    
    @staticmethod
    def gen_constant_elastance_derivative(comp_E, eps: float = 1e-3):
        """Generate elastance derivative function."""
        comp_dEdt = lambda t: (comp_E(t + eps) - comp_E(t - eps)) / (2.0 * eps)
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
            return E_act * (y - v_ref)
        return func
    
    @staticmethod
    def gen_active_dpdt(E_act: float):
        """Generate active pressure derivative."""
        def func(t, y):
            return E_act * (y[1] - y[2])
        return func
    
    @staticmethod
    def gen_passive_pressure(E_pas: float, k_pas: float, v_ref: float):
        """Generate passive pressure function."""
        def func(t, y):
            return E_pas * (np.exp(k_pas * (y - v_ref)) - 1.0)
        return func
    
    @staticmethod
    def gen_passive_dpdt(E_pas: float, k_pas: float, v_ref: float):
        """Generate passive pressure derivative."""
        def func(t, y):
            return E_pas * k_pas * np.exp(k_pas * (y[0] - v_ref)) * (y[1] - y[2])
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
            return v_ref + np.log(y[0] / E_pas + 1.0) / k_pas
        return func
    
    # Fixed interface versions of mixed elastance functions
    @staticmethod
    def gen_active_pressure_fixed(E_act: float, v_ref: float):
        """Generate active pressure function with fixed interface."""
        def func(v):
            return E_act * (v - v_ref)
        return func
    
    @staticmethod
    def gen_active_dpdt_fixed(E_act: float):
        """Generate active pressure derivative with fixed interface."""
        def func(q_i, q_o):
            return E_act * (q_i - q_o)
        return func
    
    @staticmethod
    def gen_passive_pressure_fixed(E_pas: float, k_pas: float, v_ref: float):
        """Generate passive pressure function with fixed interface."""
        def func(v):
            return E_pas * (np.exp(k_pas * (v - v_ref)) - 1.0)
        return func
    
    @staticmethod
    def gen_passive_dpdt_fixed(E_pas: float, k_pas: float, v_ref: float):
        """Generate passive pressure derivative with fixed interface."""
        def func(v, q_i, q_o):
            return E_pas * k_pas * np.exp(k_pas * (v - v_ref)) * (q_i - q_o)
        return func
    
    @staticmethod
    def gen_total_pressure_fixed(_af, active_p, passive_p):
        """Generate total pressure function with fixed interface."""
        def func(t, y):
            _af_t = _af(t)
            return _af_t * active_p(y) + (1.0 - _af_t) * passive_p(y)
        return func
    
    @staticmethod
    def gen_total_dpdt_fixed(active_p, passive_p, _af, active_dpdt, passive_dpdt):
        """Generate total pressure derivative with fixed interface."""
        def func(t, y):
            _af_t = _af(t)
            _d_af_dt = _af(t, dt=True)
            return (_d_af_dt * (active_p(y[0]) - passive_p(y[0])) +
                   _af_t * active_dpdt(y[1], y[2]) +
                   (1. - _af_t) * passive_dpdt(y[0], y[1], y[2]))
        return func
    
    # PP variants (pure passive component always included)
    @staticmethod
    def gen_total_pressure_pp(_af, active_p, passive_p):
        """Generate total pressure function for PP variant (passive always included)."""
        def func(t, y):
            _af_t = _af(t)
            return _af_t * active_p(y) + passive_p(y)
        return func
    
    @staticmethod
    def gen_total_dpdt_pp(active_p, passive_p, _af, active_dpdt, passive_dpdt):
        """Generate total pressure derivative for PP variant."""
        def func(t, y):
            _af_t = _af(t)
            _d_af_dt = _af(t, dt=True)
            return (_d_af_dt * active_p(y[0]) + 
                   _af_t * active_dpdt(y[1], y[2]) + 
                   passive_dpdt(y[0], y[1], y[2]))
        return func


class ComponentSetupMixin:
    """Mixin providing common setup functionality."""
    
    def _validate_initial_conditions(self):
        """Validate that at least one initial condition is provided."""
        has_v0 = hasattr(self, 'v0') and self.v0 is not None and not np.isnan(self.v0)
        has_p0 = hasattr(self, 'p0') and self.p0 is not None and not np.isnan(self.p0)
        
        if not has_v0 and not has_p0:
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