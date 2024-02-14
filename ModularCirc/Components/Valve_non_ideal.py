from .ComponentBase import ComponentBase
from ..Time import TimeClass
from ..HelperRoutines import non_ideal_diode_flow 

import pandas as pd

class Valve_non_ideal(ComponentBase):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 r:float, 
                 max_func
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False) 
        # setting the resistance value
        self.R = r
        self.max_func = max_func
        
    def define_functions(self):
        def q_i_u_func(t, y):
            return non_ideal_diode_flow(t, p_in=y[0], p_out=y[1], r=self.R, max_func=self.max_func)
        self.q_i_u_func = q_i_u_func
        
    def setup(self) -> None:
        self.define_functions()
        self._Q_i.set_u_func(self.q_i_u_func, function_name='non_ideal_diode_flow + max_func')
        self._Q_i.set_inputs(pd.Series({'p_in':self._P_i.name, 
                                        'p_out':self._P_o.name}))
        