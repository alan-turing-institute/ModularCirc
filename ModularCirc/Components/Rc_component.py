from .ComponentBase import ComponentBase
from ..HelperRoutines import grounded_capacitor_model_dpdt, \
    grounded_capacitor_model_pressure, \
        grounded_capacitor_model_volume, \
            resistor_model_flow, \
                chamber_volume_rate_change
from ..Time import TimeClass

import pandas as pd

class Rc_component(ComponentBase):
    def __init__(self, 
                 name:str, 
                 time_object:TimeClass,
                 r:float, 
                 c:float, 
                 v_ref:float,
                 v:float=None, 
                 p:float=None,
                 ) -> None:
        # super().__init__(time_object, main_var)
        super().__init__(time_object=time_object, name=name, v=v)
        self.R = r
        self.C = c
        self.V_ref = v_ref
        
        if p is not None:
            self.p0 = p
            self.P_i.loc[0] = p
        else:
            self.p0 = None
            
    def q_o_u_func(self, t, y):
        return resistor_model_flow(t=t, y=y, r=self.R)
    
    def p_i_dudt_func(self, t, y):
        return grounded_capacitor_model_dpdt(t, y=y, c=self.C)
    
    def p_i_i_func(self, t, y):
        return grounded_capacitor_model_pressure(t, y=y, v_ref=self.V_ref, c=self.C)
    
    def v_i_func(self, t, y):
        return grounded_capacitor_model_volume(t, y=y, v_ref=self.V_ref, c=self.C)
    
    def setup(self) -> None:
        # Set the dudt function for the input pressure state variable 
        self._P_i.set_dudt_func(self.p_i_dudt_func, function_name='grounded_capacitor_model_dpdt')
        # Set the mapping betwen the local input names and the global names of the state variables
        self._P_i.set_inputs(pd.Series({'q_in' :self._Q_i.name, 
                                        'q_out':self._Q_o.name}))
        if self.p0 is None:
            # Set the initialization function for the input pressure state variable
            self._P_i.set_i_func(self.p_i_i_func, function_name='grounded_capacitor_model_pressure')
            self._P_i.set_i_inputs(pd.Series({'v':self._V.name}))
        else:
            self.P_i.loc[0] = self.p0
        # Set the function for computing the flows based on the current pressure values at the nodes of the componet
        self._Q_o.set_u_func(self.q_o_u_func, function_name='resistor_model_flow')
        self._Q_o.set_inputs(pd.Series({'p_in':self._P_i.name, 
                                        'p_out':self._P_o.name}))
        # Set the dudt function for the compartment volume
        self._V.set_dudt_func(chamber_volume_rate_change, function_name='chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in':self._Q_i.name, 
                                      'q_out':self._Q_o.name}))
        if self.v0 is None:
            # Set the initialization function for the input volume state variable  
            self._V.set_i_func(self.v_i_func, function='grounded_capacitor_model_volume')     
            self._V.set_i_inputs(pd.Series({'p':self._P_i.name}))  
        