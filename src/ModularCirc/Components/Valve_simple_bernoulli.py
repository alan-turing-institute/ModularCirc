from .ComponentBase import ComponentBase
from ..Time import TimeClass
from ..HelperRoutines import simple_bernoulli_diode_flow

import pandas as pd

class Valve_simple_bernoulli(ComponentBase):
    def __init__(self,
                 name:str,
                 time_object: TimeClass,
                 CQ:float,
                 RRA:float=0.0,
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False)
        # setting the resistance value
        self.CQ = CQ
        # setting the relative regurgitant area
        self.RRA = RRA

    def q_i_u_func(self, t, y):
        return simple_bernoulli_diode_flow(t, y=y, CQ=self.CQ, RRA=self.RRA)

    def setup(self) -> None:
        CQ  = self.CQ
        RRA = self.RRA
        q_i_u_func = lambda t, y: simple_bernoulli_diode_flow(t, y=y, CQ=CQ, RRA=RRA)
        self._Q_i.set_u_func(q_i_u_func, function_name='simple_bernoulli_diode_flow')
        self._Q_i.set_inputs(pd.Series({'p_in':self._P_i.name,
                                        'p_out':self._P_o.name}))
