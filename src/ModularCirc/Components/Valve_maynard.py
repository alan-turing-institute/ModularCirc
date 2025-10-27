from .ComponentBase import ComponentBase
from ._ComponentFactoriesAuto import ComponentFunctionFactory 
from ..Time import TimeClass
from ..StateVariable import StateVariable

import pandas as pd

class Valve_maynard(ComponentBase):
    def __init__(self,
                 name: str,
                 time_object: TimeClass,
                 Kc: float,
                 Ko: float,
                 CQ: float,
                 R: float = 0.0,
                 L: float = 0.0,
                 RRA: float = 0.0,
                 *args, **kwargs
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False)
        self.CQ = CQ
        self.R = R
        self.L = L
        self.RRA = RRA
        self.Kc, self.Ko = Kc, Ko
        # defining the valve opening factor state variable
        self._PHI = StateVariable(name=name+'_PHI', timeobj=time_object)

    @property
    def PHI(self):
        return self._PHI._u

    def setup(self) -> None:
        if self.L < 1.0e-6:
            # Low inductance: use algebraic flow function
            q_i_func = ComponentFunctionFactory.gen_maynard_valve_flow(self.CQ, self.RRA)
            self._Q_i.set_u_func(q_i_func, function_name='maynard_valve_flow')
            self._Q_i.set_inputs(pd.Series({'p_in': self._P_i.name,
                                            'p_out': self._P_o.name,
                                            'phi': self._PHI.name}))
        else:
            # High inductance: use differential flow function
            q_i_dudt_func = ComponentFunctionFactory.gen_maynard_impedance_dqdt(
                self.CQ, self.RRA, self.L, self.R)
            self._Q_i.set_dudt_func(q_i_dudt_func, function_name='maynard_impedance_dqdt')
            self._Q_i.set_inputs(pd.Series({'p_in': self._P_i.name,
                                            'p_out': self._P_o.name,
                                            'q_in': self._Q_i.name,
                                            'phi': self._PHI.name}))

        # Phi (valve opening) dynamics
        phi_dudt_func = ComponentFunctionFactory.gen_maynard_phi_law(self.Ko, self.Kc)
        self._PHI.set_dudt_func(phi_dudt_func, function_name='maynard_phi_law')
        self._PHI.set_inputs(pd.Series({'p_in': self._P_i.name,
                                        'p_out': self._P_o.name,
                                        'phi': self._PHI.name}))
