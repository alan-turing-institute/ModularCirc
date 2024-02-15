from .ComponentBase import ComponentBase
from ..Time import TimeClass
from ..HelperRoutines import simple_bernoulli_diode_flow 

import pandas as pd

class Valve_simple_bernoulli(ComponentBase):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 CQ:float, 
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False) 
        # setting the resistance value
        self.max_func = CQ
        
    def q_i_u_func(self, t, y):
        return simple_bernoulli_diode_flow(t, y=y, CQ=self.CQ)
        
    def setup(self) -> None:
        self._Q_i.set_u_func(self.q_i_u_func, function_name='simple_bernoulli_diode_flow')
        self._Q_i.set_inputs(pd.Series({'p_in':self._P_i.name, 
                                        'p_out':self._P_o.name}))
        