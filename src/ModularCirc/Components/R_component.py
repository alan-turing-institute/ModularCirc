from .ComponentBase import ComponentBase
from ..Time import TimeClass
from ..HelperRoutines import resistor_upstream_pressure

import pandas as pd

class R_component(ComponentBase):
    def __init__(self,
                 name:str,
                 time_object: TimeClass,
                 r:float,
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False)
        # setting the resistance value
        self.R = r

    @property
    def P(self):
        return self._P_i._u

    def p_i_u_func(self, t, y):
        return resistor_upstream_pressure(t, y=y, r=self.R)

    def setup(self) -> None:
        r=self.R
        self._P_i.set_u_func(lambda t, y: resistor_upstream_pressure(t, y, r=r), function_name='resistor_upstream_pressure')
        self._P_i.set_inputs(pd.Series({'q_in' :self._Q_i.name,
                                        'p_out':self._P_o.name}))
