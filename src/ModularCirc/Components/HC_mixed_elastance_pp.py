from .ComponentBase import ComponentBase
from ._ComponentFactories import ComponentFunctionFactory, ElastanceFactory
from ..HelperRoutines import activation_function_1, chamber_volume_rate_change
from ..Time import TimeClass

import pandas as pd
import numpy as np

class HC_mixed_elastance_pp(ComponentBase):
    def __init__(self,
                 name:str,
                 time_object: TimeClass,
                 E_pas: float,
                 E_act: float,
                 k_pas: float,
                 v_ref: float,
                 v    : float = None,
                 p    : float = None,
                 af = activation_function_1,
                 *args, **kwargs
                 ) -> None:
        super().__init__(name=name, time_object=time_object, v=v, p=p)
        self.E_pas = E_pas
        self.k_pas = k_pas
        self.E_act = E_act
        self.v_ref = v_ref
        self.eps = 1.0e-3
        self.kwargs = kwargs
        self.af = af

        self.make_unique_io_state_variable(p_flag=True, q_flag=False)

    @property
    def P(self):
        return self._P_i._u

    def setup(self) -> None:
        # Use factory methods for all function generation
        time_shifter = ComponentFunctionFactory.gen_time_shifter(
            self.kwargs['delay'], self._to.tcycle)
        _af = ComponentFunctionFactory.gen_activation_function(
            self.af, time_shifter, **self.kwargs)

        # Use simplified PP variant functions (passive always included)
        total_p = ElastanceFactory.gen_total_pressure_pp(
            _af, self.E_act, self.v_ref, self.E_pas, self.k_pas)
        total_dpdt = ElastanceFactory.gen_total_dpdt_pp(
            _af, self.E_act, self.v_ref, self.E_pas, self.k_pas)
        comp_v = ElastanceFactory.gen_volume_from_pressure_nonlinear(
            self.E_pas, self.v_ref, self.k_pas)

        # Volume dynamics
        self._V.set_dudt_func(chamber_volume_rate_change, function_name='chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in': self._Q_i.name,
                                      'q_out': self._Q_o.name}))

        # Pressure dynamics
        self._P_i.set_dudt_func(total_dpdt, function_name='total_dpdt')
        self._P_i.set_inputs(pd.Series({'v': self._V.name,
                                        'q_i': self._Q_i.name,
                                        'q_o': self._Q_o.name}))
        
        # Initial conditions using base class helper
        self._validate_initial_conditions()
        
        if self._is_none_or_nan(self.p0):
            self._P_i.set_i_func(total_p, function_name='total_p')
            self._P_i.set_i_inputs(pd.Series({'v': self._V.name}))
        else:
            self._P_i._u.loc[0] = self.p0
            
        if self._is_none_or_nan(self.v0):
            self._V.set_i_func(comp_v, function_name='comp_v')
            self._V.set_i_inputs(pd.Series({'p': self._P_i.name}))
