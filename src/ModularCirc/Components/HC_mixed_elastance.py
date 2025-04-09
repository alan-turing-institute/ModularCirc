from .ComponentBase import ComponentBase
from ..HelperRoutines import activation_function_1, \
    chamber_volume_rate_change, \
                time_shift
from ..Time import TimeClass

import pandas as pd
import numpy as np


def dvdt(t, q_in=None, q_out=None, v=None, v_ref=0.0, y=None):
    if y is not None:
        q_in, q_out, v = y[:3]
    if q_in - q_out > 0.0 or v > v_ref:
        return q_in - q_out
    else:
        return 0.0

def gen_active_p(E_act, v_ref):
    def active_p(v):
        return E_act * (v - v_ref)
    return active_p

def gen_active_dpdt(E_act):
    def active_dpdt(q_i, q_o):
        return E_act * (q_i - q_o)
    return active_dpdt

def gen_passive_p(E_pas, k_pas, v_ref):
    def passive_p(v):
        return E_pas * (np.exp(k_pas * (v - v_ref)) - 1.0)
    return passive_p

def gen_passive_dpdt(E_pas, k_pas, v_ref):
    def passive_dpdt(v, q_i, q_o):
        return  E_pas * k_pas * np.exp(k_pas * (v - v_ref)) * (q_i - q_o)
    return passive_dpdt

def gen_total_p(_af, active_p, passive_p):
    def total_p(t, y):
        _af_t = _af(t)
        return _af_t * active_p(y) + (1.0 - _af_t) * passive_p(y)
    return total_p

def gen_total_dpdt(active_p, passive_p, _af, active_dpdt, passive_dpdt):
    def total_dpdt(t, y):
        _af_t = _af(t)
        _d_af_dt = _af(t, dt=True)
        return (_d_af_dt *(active_p(y[0]) - passive_p(y[0])) + 
                _af_t      * active_dpdt(       y[1], y[2]) + 
               (1. -_af_t) * passive_dpdt(y[0], y[1], y[2]))
    return total_dpdt

def gen_comp_v(E_pas, v_ref, k_pas):
    def comp_v(t, y): 
        return v_ref + np.log(y[0] / E_pas + 1.0) / k_pas
    return comp_v

def gen_time_shifter(delay_, T):
    def time_shifter(t):
        return  time_shift(t, delay_, T)
    return time_shifter

def gen__af(af, time_shifter, kwargs):
    varnames = [name for name in af.__code__.co_varnames if name != 'coeff' and name != 't']
    kwargs2  = {key: val for key,val in kwargs.items() if key in varnames}    
    def _af(t, dt=False):
        return af(time_shifter(t), dt=dt, **kwargs2)
    return _af

class HC_mixed_elastance(ComponentBase):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 E_pas: float, 
                 E_act: float,
                 k_pas: float,
                 v_ref: float,
                 v    : float = None,
                 p    : float = None,
                 af = activation_function_1,
                 *args, **kwargs
                 ) -> None:
        super().__init__(name=name, time_object=time_object, v=v, p=p)
        self.E_pas = E_pas
        self.k_pas = k_pas
        self.E_act = E_act
        self.v_ref = v_ref
        self.eps = 1.0e-3
        self.kwargs = kwargs
        self.af = af
        
        self.make_unique_io_state_variable(p_flag=True, q_flag=False)
        
    @property    
    def P(self):
        return self._P_i._u
    
    def setup(self) -> None:
        E_pas = self.E_pas
        k_pas = self.k_pas
        E_act = self.E_act
        v_ref = self.v_ref
        eps   = self.eps
        kwargs= self.kwargs
        T     = self._to.tcycle
        af    = self.af
        
        time_shifter = gen_time_shifter(delay_=kwargs['delay'], T=T)
        _af          = gen__af(af=af, time_shifter=time_shifter, kwargs=kwargs)
        
        active_p     = gen_active_p(E_act=E_act, v_ref=v_ref)
        active_dpdt  = gen_active_dpdt(E_act=E_act)
        passive_p    = gen_passive_p(E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
        passive_dpdt = gen_passive_dpdt(E_pas=E_pas, k_pas=k_pas, v_ref=v_ref)
        total_p      = gen_total_p(_af=_af, active_p=active_p, passive_p=passive_p)
        total_dpdt   = gen_total_dpdt(active_p=active_p, passive_p=passive_p, 
                                      _af=_af, active_dpdt=active_dpdt, passive_dpdt=passive_dpdt)
        comp_v       = gen_comp_v(E_pas=E_pas, v_ref=v_ref, k_pas=k_pas)
        
        self._V.set_dudt_func(chamber_volume_rate_change,
                              function_name='chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in' :self._Q_i.name, 
                                      'q_out':self._Q_o.name}))
        
        self._P_i.set_dudt_func(total_dpdt, function_name='total_dpdt') 
        self._P_i.set_inputs(pd.Series({'v'  :self._V.name, 
                                        'q_i':self._Q_i.name, 
                                        'q_o':self._Q_o.name}))
        if self.p0 is None or self.p0 is np.NaN:
            self._P_i.set_i_func(total_p, function_name='total_p')
            self._P_i.set_i_inputs(pd.Series({'v':self._V.name}))
        else:
            self.P_i.loc[0] = self.p0
        if self.v0 is None or self.v0 is np.NaN:
            self._V.set_i_func(comp_v, function_name='comp_v')
            self._V.set_i_inputs(pd.Series({'p':self._P_i.name}))
        if (self.v0 is None or self.v0 is np.NaN) and (self.p0 is None or self.p0 is np.NaN):
            raise Exception("Solver needs at least the initial volume or pressure to be defined!")                                    