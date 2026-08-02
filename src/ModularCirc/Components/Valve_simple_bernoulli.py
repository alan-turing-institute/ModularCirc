from .ComponentBase import ComponentBase
from ._ComponentFactoriesAuto import ComponentFunctionFactory
from ..Time import TimeClass

import pandas as pd

class Valve_simple_bernoulli(ComponentBase):
    def __init__(self,
                 name: str,
                 time_object: TimeClass,
                 CQ: float,
                 RRA: float = 0.0,
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False)
        self.CQ = CQ
        self.RRA = RRA

    @property
    def Q(self):
        return self._Q_i._u

    def setup(self) -> None:
        q_i_func = ComponentFunctionFactory.gen_simple_bernoulli_flow(self.CQ, self.RRA)
        self._Q_i.set_u_func(q_i_func, function_name='simple_bernoulli_diode_flow')
        self._Q_i.set_inputs(pd.Series({'p_in': self._P_i.name,
                                        'p_out': self._P_o.name}))
