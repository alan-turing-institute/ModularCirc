from .ComponentBase import ComponentBase
from ._ComponentFactories import ComponentFunctionFactory
from ..Time import TimeClass

import pandas as pd

class Valve_non_ideal(ComponentBase):
    def __init__(self,
                 name: str,
                 time_object: TimeClass,
                 r: float,
                 max_func
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False)
        self.R = r
        self.max_func = max_func

    def setup(self) -> None:
        q_i_func = ComponentFunctionFactory.gen_non_ideal_diode_flow(self.R, self.max_func)
        self._Q_i.set_u_func(q_i_func, function_name='non_ideal_diode_flow + max_func')
        self._Q_i.set_inputs(pd.Series({'p_in': self._P_i.name,
                                        'p_out': self._P_o.name}))
