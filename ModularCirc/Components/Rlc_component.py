from .Rc_component import Rc_component
from ..HelperRoutines import resistor_impedance_flux_rate
from ..Time import TimeClass

import pandas as pd
import numpy as np

class Rlc_component(Rc_component):
    def __init__(self, 
                 name:str, 
                 time_object:TimeClass,
                 r:float, 
                 c:float, 
                 l:float,
                 v_ref:float,
                 v:float=None,
                 p:float=None, 
                 ) -> None:
        super().__init__(time_object=time_object, name=name, v=v, p=p, r=r, c=c, v_ref=v_ref)
        self.L = l
    
    def setup(self) -> None:
        Rc_component.setup(self)
        if (np.abs(self.L) > 1e-11):
            self._Q_o.set_dudt_func(lambda t, p_in, p_out, q_out : resistor_impedance_flux_rate(t, p_in=p_in, p_out=p_out, q_out=q_out, r=self.R, l=self.L),
                                function_name='lambda resistor_impedance_flux_rate')
            self._Q_o.set_inputs(pd.Series({'p_in':self._P_i.name, 
                                            'p_out':self._P_o.name,
                                            'q_out':self._Q_o.name}))
            self._Q_o.set_u_func(None,None)