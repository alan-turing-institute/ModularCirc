from .ComponentBase import ComponentBase
from ._ComponentFactoriesAuto import ComponentFunctionFactory
from ..HelperRoutines import chamber_volume_rate_change
from ..Time import TimeClass

import pandas as pd
import numpy as np

class Rc_nonlinear_component(ComponentBase):
    def __init__(
        self, 
        name:str, 
        time_object:TimeClass, 
        r:float,
        c_ref:float,
        p_ref:float,
        v_ref:float,
        v:float = None, 
        p:float = None,
    
    ) -> None:
        super().__init__(name=name, time_object=time_object, v=v)
        self.R = r
        self.C_ref = c_ref
        self.P_ref = p_ref
        self.V_ref = v_ref
        self.p0 = p
        
        if p is not None:
            self._P_i._u.loc[0] = p
        return
    
    def setup(self) -> None:
        # Use factory methods for all functions
        p_i_dudt_func = ComponentFunctionFactory.gen_nonlinear_capacitor_dpdt(c0 = self.C_ref, p0 = self.P_ref)
        p_i_init_func = ComponentFunctionFactory.gen_nonlinear_capacitor_pressure(v_ref=self.V_ref, c0=self.C_ref, p0=self.P_ref)
        v_i_func = ComponentFunctionFactory.gen_nonlinear_capacitor_volume(v_ref=self.V_ref, c0=self.C_ref, p0=self.P_ref)
        q_o_func = ComponentFunctionFactory.gen_resistor_flow(self.R)
        
        # Set the dudt function for the input pressure state variable
        self._P_i.set_dudt_func(p_i_dudt_func, function_name='grounded_nonlinear_capacitor_model_dpdt')
        self._P_i.set_inputs(pd.Series({'q_in': self._Q_i.name,
                                        'q_out': self._Q_o.name,
                                        'p_in': self._P_i.name}))
        
        # Pressure initialization
        if self._is_none_or_nan(self.p0):
            self._P_i.set_i_func(p_i_init_func, function_name='grounded_nonlinear_capacitor_model_pressure')
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
            self._V.set_i_func(v_i_func, function_name='grounded_nonlinear_capacitor_model_volume')
            self._V.set_i_inputs(pd.Series({'p': self._P_i.name}))
        return 
    
    def __del__(self):
        return super().__del__()
    
    