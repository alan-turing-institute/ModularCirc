from .ComponentBase import ComponentBase
from ..HelperRoutines import activation_function_1, \
    chamber_volume_rate_change, \
        chamber_pressure_function, \
            chamber_linear_elastic_law
from ..Time import TimeClass

import pandas as pd
import numpy as np

class HC_constant_elastance(ComponentBase):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 E_pas: float, 
                 E_act: float,
                 V_ref: float,
                 v    : float = None,
                 af = activation_function_1,
                 ) -> None:
        super().__init__(name=name, time_object=time_object, v=v)
        self._af = af
        self.E_pas = E_pas
        self.E_act = E_act
        self.V_ref = V_ref
        self.eps = 1.0e-3
        
        self.make_unique_io_state_variable(p_flag=True, q_flag=False)
        
    def comp_E(self, t:float) -> float:
        return self._af(t) * self.E_act + (1.0 - self._af(t)) * self.E_pas
    
    def comp_dEdt(self, t:float) -> float:
        return (self.comp_E(t + self.eps) - self.comp_E(t - self.eps)) / 2.0 / self.eps
    
    def comp_p(self, t:float, v:float=None, y:np.ndarray[float]=None) ->float:
        e = self.comp_E(t)
        if y is not None:
            v = y
        return e * (v - self.V_ref)
    
    def comp_dpdt(self, t:float=None, V:float=None, q_i:float=None, q_o:float=None, y:np.ndarray[float]=None) -> float:
        if y is not None:
            V, q_i, q_o, = y[:3]
        dEdt = self.comp_dEdt(t)
        e    = self.comp_E(t)
        return dEdt * (V - self.V_ref) + e * (q_i - q_o)
        
    def setup(self) -> None:
        self._V.set_dudt_func(chamber_volume_rate_change,
                              function_name='chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in':self._Q_i.name, 
                                      'q_out':self._Q_o.name}))
        self._P_i.set_dudt_func(self.comp_dpdt, function_name='self.comp_dpdt') 
        self._P_i.set_inputs(pd.Series({'V':self._V.name, 
                                        'q_i':self._Q_i.name, 
                                        'q_o':self._Q_o.name}))
        self._P_i.set_i_func(self.comp_p,
                             function_name='lamda chamber_pressure_function + activation_function_1 + 2xchamber_linear_elastic_law')
        self._P_i.set_i_inputs(pd.Series({'V':self._V.name}))