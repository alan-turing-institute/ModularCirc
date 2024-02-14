from .ComponentBase import ComponentBase
from ..HelperRoutines import grounded_capacitor_model_dpdt, \
    grounded_capacitor_model_pressure, \
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
            
    def define_functions(self):
        def q_o_u_func(t, y):
            return resistor_model_flow(t, p_in=y[0], p_out=y[1], r=self.R)
        self.q_o_u_func = q_o_u_func
    
    def setup(self) -> None:
        self.define_functions()
        # Set the dudt function for the input pressure state variable 
        self._P_i.set_dudt_func(lambda t, q_in, q_out: grounded_capacitor_model_dpdt(t=t, q_in=q_in, q_out=q_out, c=self.C),
                                function_name='lambda grounded_capacitor_model_dpdt')
        # Set the mapping betwen the local input names and the global names of the state variables
        self._P_i.set_inputs(pd.Series({'q_in' :self._Q_i.name, 
                                        'q_out':self._Q_o.name}))
        # Set the initialization function for the input pressure state variable
        self._P_i.set_i_func(lambda V: grounded_capacitor_model_pressure(t=0.0, v=V, v_ref=self.V_ref, c=self.C),
                             function_name= 'lambda grounded_capacitor_model_pressure')
        self._P_i.set_i_inputs(pd.Series({'V':self._V.name}))
        # Set the function for computing the flows based on the current pressure values at the nodes of the componet
        self._Q_o.set_u_func(self.q_o_u_func, function_name='resistor_model_flow')
        self._Q_o.set_inputs(pd.Series({'p_in':self._P_i.name, 
                                        'p_out':self._P_o.name}))
        # Set the dudt function for the compartment volume
        self._V.set_dudt_func(chamber_volume_rate_change, function_name='chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in':self._Q_i.name, 
                                      'q_out':self._Q_o.name}))         
        