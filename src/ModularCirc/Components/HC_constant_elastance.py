from .ComponentBase import ComponentBase
# from ._ComponentFactories import ComponentFunctionFactory, ElastanceFactory
from ._ComponentFactoriesAuto import ComponentFunctionFactory, ElastanceFactory
from ..HelperRoutines import (
    activation_function_1, chamber_volume_rate_change
)
from ..Time import TimeClass

import pandas as pd
import numpy as np

class HC_constant_elastance(ComponentBase):
    def __init__(self,
                 name: str,
                 time_object: TimeClass,
                 E_pas: float,
                 E_act: float,
                 v_ref: float,
                 v: float = None,
                 p: float = None,
                 af=activation_function_1,
                 *args, **kwargs
                 ) -> None:
        super().__init__(name=name, time_object=time_object, v=v, p=p)
        self.E_pas = E_pas
        self.E_act = E_act
        self.v_ref = v_ref
        self.eps = 1.0e-3
        self.p0 = p
        self.af = af
        self.kwargs = kwargs

        # Create parameterized activation function
        time_shifter = ComponentFunctionFactory.gen_time_shifter(
            kwargs['delay'], time_object.tcycle)
        self._af = ComponentFunctionFactory.gen_activation_function(
            af, time_shifter, **kwargs)

        self.make_unique_io_state_variable(p_flag=True, q_flag=False)

    def setup(self) -> None:
        # Generate elastance functions using factory
        comp_E = ElastanceFactory.gen_constant_elastance(
            self.E_act, self.E_pas, self._af, self.v_ref)
        comp_dEdt = ElastanceFactory.gen_constant_elastance_derivative(comp_E, self.eps)
        comp_p = ElastanceFactory.gen_pressure_from_volume(comp_E, self.v_ref)
        comp_v = ElastanceFactory.gen_volume_from_pressure(comp_E, self.v_ref)
        comp_dpdt = ElastanceFactory.gen_pressure_derivative(comp_E, comp_dEdt, self.v_ref)

        # Volume dynamics
        self._V.set_dudt_func(chamber_volume_rate_change, function_name='chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in': self._Q_i.name,
                                      'q_out': self._Q_o.name}))
        
        # Pressure dynamics
        self._P_i.set_dudt_func(comp_dpdt, function_name='comp_dpdt')
        self._P_i.set_inputs(pd.Series({'V': self._V.name,
                                        'q_i': self._Q_i.name,
                                        'q_o': self._Q_o.name}))
        
        # Initial conditions using base class helper
        self._validate_initial_conditions()
        
        if self._is_none_or_nan(self.p0):
            self._P_i.set_i_func(comp_p, function_name='comp_p')
            self._P_i.set_i_inputs(pd.Series({'V': self._V.name}))
        else:
            self._P_i._u.loc[0] = self.p0
            
        if self._is_none_or_nan(self.v0):
            self._V.set_i_func(comp_v, function_name='comp_v')
            self._V.set_i_inputs(pd.Series({'p': self._P_i.name}))
