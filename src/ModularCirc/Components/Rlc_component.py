from .Rc_component import Rc_component
from ._ComponentFactoriesAuto import ComponentFunctionFactory
from ..Time import TimeClass

import pandas as pd
import numpy as np

class Rlc_component(Rc_component):
    def __init__(self,
                 name: str,
                 time_object: TimeClass,
                 r: float,
                 c: float,
                 l: float,
                 v_ref: float,
                 v: float = None,
                 p: float = None,
                 ) -> None:
        super().__init__(time_object=time_object, name=name, v=v, p=p, r=r, c=c, v_ref=v_ref)
        self.L = l

    def setup(self) -> None:
        # Setup base RC component first
        super().setup()
        
        # Add inductance behavior if significant
        if np.abs(self.L) > 1e-11:
            q_o_dudt_func = ComponentFunctionFactory.gen_impedance_flow_rate(self.R, self.L)
            self._Q_o.set_dudt_func(q_o_dudt_func, function_name='resistor_impedance_flux_rate')
            self._Q_o.set_inputs(pd.Series({'p_in': self._P_i.name,
                                            'p_out': self._P_o.name,
                                            'q_out': self._Q_o.name}))
            # Remove u_func since we now have dudt_func
            self._Q_o.set_u_func(None, None)

    def __del__(self):
        super().__del__()
