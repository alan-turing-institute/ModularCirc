from .ComponentBase import ComponentBase
from ._ComponentFactories import ComponentFunctionFactory
from ..HelperRoutines import chamber_volume_rate_change
from ..Time import TimeClass

import pandas as pd
import numpy as np

class Rc_component(ComponentBase):
    def __init__(self,
                 name: str,
                 time_object: TimeClass,
                 r: float,
                 c: float,
                 v_ref: float,
                 v: float = None,
                 p: float = None,
                 ) -> None:
        super().__init__(time_object=time_object, name=name, v=v)
        self.R = r
        self.C = c
        self.V_ref = v_ref
        self.p0 = p

        if p is not None:
            self._P_i._u.loc[0] = p

    def setup(self) -> None:
        # Use factory methods for all functions
        p_i_dudt_func = ComponentFunctionFactory.gen_capacitor_dpdt(self.C)
        p_i_init_func = ComponentFunctionFactory.gen_capacitor_pressure(self.V_ref, self.C)
        q_o_func = ComponentFunctionFactory.gen_resistor_flow(self.R)
        v_i_func = ComponentFunctionFactory.gen_capacitor_volume(self.V_ref, self.C)
        
        # Set the dudt function for the input pressure state variable
        self._P_i.set_dudt_func(p_i_dudt_func, function_name='grounded_capacitor_model_dpdt')
        self._P_i.set_inputs(pd.Series({'q_in': self._Q_i.name,
                                        'q_out': self._Q_o.name}))
        
        # Pressure initialization
        if self._is_none_or_nan(self.p0):
            self._P_i.set_i_func(p_i_init_func, function_name='grounded_capacitor_model_pressure')
            self._P_i.set_i_inputs(pd.Series({'v': self._V.name}))
        else:
            self._P_i._u.loc[0] = self.p0
            
        # Flow computation
        self._Q_o.set_u_func(q_o_func, function_name='resistor_model_flow')
        self._Q_o.set_inputs(pd.Series({'p_in': self._P_i.name,
                                        'p_out': self._P_o.name}))
        
        # Volume state variable
        self._V.set_dudt_func(chamber_volume_rate_change, function_name='chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in': self._Q_i.name,
                                      'q_out': self._Q_o.name}))
        
        # Volume initialization
        if self._is_none_or_nan(self.v0):
            self._V.set_i_func(v_i_func, function_name='grounded_capacitor_model_volume')
            self._V.set_i_inputs(pd.Series({'p': self._P_i.name}))

    def __del__(self):
        super().__del__()
