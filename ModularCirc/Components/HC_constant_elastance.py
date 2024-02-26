from .ComponentBase import ComponentBase
from ..HelperRoutines import activation_function_1, \
    chamber_volume_rate_change, \
                time_shift
from ..Time import TimeClass

import pandas as pd
import numpy as np

class HC_constant_elastance(ComponentBase):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 E_pas: float, 
                 E_act: float,
                 v_ref: float,
                 v    : float = None,
                 p    : float = None,
                 af = activation_function_1,
                 *args, **kwargs
                 ) -> None:
        super().__init__(name=name, time_object=time_object, v=v, p=p)
        self.E_pas = E_pas
        self.E_act = E_act
        self.v_ref = v_ref
        self.eps = 1.0e-3
        
        def af_parameterised(t):
            return af(time_shift(t, kwargs['delay'], time_object.tcycle) , **kwargs)
        self._af = af_parameterised
        
        self.make_unique_io_state_variable(p_flag=True, q_flag=False)
        
    def comp_E(self, t:float) -> float:
        return self._af(t) * self.E_act + (1.0 - self._af(t)) * self.E_pas
    
    def comp_dEdt(self, t:float) -> float:
        return (self.comp_E(t + self.eps) - self.comp_E(t - self.eps)) / 2.0 / self.eps
    
    def comp_p(self, t:float, v:float=None, y:np.ndarray[float]=None) ->float:
        e = self.comp_E(t)
        if y is not None:
            v = y
        return e * (v - self.v_ref)
    
    def comp_v(self, t:float=None, p:float=None, y:np.ndarray[float]=None)->float:
        e = self.comp_E(t)
        if y is not None:
            p = y
        return p / e + self.v_ref
    
    def comp_dpdt(self, t:float=None, V:float=None, q_i:float=None, q_o:float=None, y:np.ndarray[float]=None) -> float:
        if y is not None:
            V, q_i, q_o, = y[:3]
        dEdt = self.comp_dEdt(t)
        e    = self.comp_E(t)
        return dEdt * (V - self.v_ref) + e * (q_i - q_o)
        
    def setup(self) -> None:
        self._V.set_dudt_func(chamber_volume_rate_change,
                              function_name='chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in':self._Q_i.name, 
                                      'q_out':self._Q_o.name}))
        self._P_i.set_dudt_func(self.comp_dpdt, function_name='self.comp_dpdt') 
        self._P_i.set_inputs(pd.Series({'V':self._V.name, 
                                        'q_i':self._Q_i.name, 
                                        'q_o':self._Q_o.name}))
        if self.p0 is None or self.p0 is np.NaN:
            self._P_i.set_i_func(self.comp_p, function_name='self.comp_p')
            self._P_i.set_i_inputs(pd.Series({'V':self._V.name}))
        else:
            self.P_i.loc[0] = self.p0
        if self.v0 is None or self.v0 is np.NaN:
            self._V.set_i_func(self.comp_v, function_name='self.comp_v')
            self._V.set_i_inputs(pd.Series({'p':self._P_i.name}))
        if (self.v0 is None or self.v0 is np.NaN) and (self.p0 is None or self.p0 is np.NaN):
            raise Exception("Solver needs at least the initial volume or pressure to be defined!")
            