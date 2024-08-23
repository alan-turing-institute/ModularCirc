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
        
        self._af = lambda t : af(time_shift(t, kwargs['delay'], time_object.tcycle) , **kwargs)
        
        self.make_unique_io_state_variable(p_flag=True, q_flag=False)
        
    @property    
    def P(self):
        return self._P_i._u
        
    def active_p(self, v):
        return self.E_act * (v - self.v_ref)
    
    def active_dpdt(self, q_i, q_o):
        return self.E_act * (q_i - q_o)
    
    def passive_p(self, v):
        return self.E_pas * (np.exp(self.k_pas * (v - self.v_ref)) - 1.0)
    
    def passive_dpdt(self, v, q_i, q_o):
        return self.E_pas * self.k_pas * np.exp(self.k_pas * (v - self.v_ref)) * (q_i - q_o)
    
    def total_p(self, t, v=None, y=None):
        if y is not None:
            v = y
        return self._af(t) * self.active_p(v) + (1.0 - self._af(t)) * self.passive_p(v)
    
    def d_af_dt(self, t):
        return (self._af(t+self.eps) - self._af(t-self.eps)) / 2.0 / self.eps
    
    def total_dpdt(self, t, v=None, q_i=None, q_o=None, y=None):
        if y is not None:
            v, q_i, q_o = y[:3]
        return (self.d_af_dt(t)   *(self.active_p(v) - self.passive_p(v)) + 
                     self._af(t)  * self.active_dpdt(    q_i, q_o) + 
                (1. -self._af(t)) * self.passive_dpdt(v, q_i, q_o))
        
    def comp_v(self, t:float=None, p:float=None, y:np.ndarray[float]=None)->float:
        if y is not None:
            p = y[:1]
        return self.v_ref + np.log(p / self.E_pas + 1.0) / self.k_pas
    
    def comp_dvdt(self, t, q_in=None, q_out=None, v=None, y=None):
        if y is not None:
            q_in, q_out, v = y[:3]
        return dvdt(t, q_in=q_in, q_out=q_out, v=v, v_ref=self.v_ref)
        
    def setup(self) -> None:
        E_pas = self.E_pas
        k_pas = self.k_pas
        E_act = self.E_act
        v_ref = self.v_ref
        eps   = self.eps
        _af   = self._af
        
        active_p     = lambda v : E_act * (v - v_ref)
        active_dpdt  = lambda q_i, q_o : E_act * (q_i - q_o)
        passive_p    = lambda v : E_pas * (np.exp(k_pas * (v - v_ref)) - 1.0)
        passive_dpdt = lambda v, q_i, q_o : E_pas * k_pas * np.exp(k_pas * (v - v_ref)) * (q_i - q_o)
        total_p      = lambda t, y : _af(t) * active_p(y) + (1.0 - _af(t)) * passive_p(y)
        d_af_dt      = lambda t : (_af(t+eps) - _af(t-eps)) / 2.0 / eps
        total_dpdt   = lambda t, y : (d_af_dt(t)  * (active_p(y[0]) - passive_p(y[0])) + 
                                      _af(t)      * active_dpdt(       y[1], y[2]) + 
                                     (1. -_af(t)) * passive_dpdt(y[0], y[1], y[2]))
        comp_v       = lambda t, y : v_ref + np.log(y[0] / E_pas + 1.0) / k_pas
        
        self._V.set_dudt_func(chamber_volume_rate_change,
                              function_name='chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in' :self._Q_i.name, 
                                      'q_out':self._Q_o.name}))
        
        # self._P_i.set_dudt_func(self.total_dpdt, function_name='self.total_dpdt') 
        self._P_i.set_dudt_func(total_dpdt, function_name='lambda.total_dpdt') 
        self._P_i.set_inputs(pd.Series({'v'  :self._V.name, 
                                        'q_i':self._Q_i.name, 
                                        'q_o':self._Q_o.name}))
        if self.p0 is None or self.p0 is np.NaN:
            # self._P_i.set_i_func(self.total_p, function_name='self.total_p')
            self._P_i.set_i_func(total_p, function_name='lambda.total_p')
            self._P_i.set_i_inputs(pd.Series({'v':self._V.name}))
        else:
            self.P_i.loc[0] = self.p0
        if self.v0 is None or self.v0 is np.NaN:
            # self._V.set_i_func(self.comp_v, function_name='self.comp_v')
            self._V.set_i_func(comp_v, function_name='lambda.comp_v')
            self._V.set_i_inputs(pd.Series({'p':self._P_i.name}))
        if (self.v0 is None or self.v0 is np.NaN) and (self.p0 is None or self.p0 is np.NaN):
            raise Exception("Solver needs at least the initial volume or pressure to be defined!")                                    