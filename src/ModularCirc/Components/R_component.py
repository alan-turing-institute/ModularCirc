from .ComponentBase import ComponentBase
from ._ComponentFactoriesAuto import ComponentFunctionFactory
from ..Time import TimeClass

import pandas as pd

class R_component(ComponentBase):
    def __init__(self,
                 name: str,
                 time_object: TimeClass,
                 r: float,
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False)
        # setting the resistance value
        self.R = r

    def setup(self) -> None:
        # Use factory method instead of lambda
        p_i_func = ComponentFunctionFactory.gen_resistor_upstream_pressure(self.R)
        self._P_i.set_u_func(p_i_func, function_name='resistor_upstream_pressure')
        self._P_i.set_inputs(pd.Series({'q_in': self._Q_i.name,
                                        'p_out': self._P_o.name}))
