from .ComponentBase import ComponentBase
from ..Time import TimeClass
from ..HelperRoutines import maynard_valve_flow, maynard_phi_law, maynard_impedance_dqdt 
from ..StateVariable import StateVariable

import pandas as pd

class Valve_maynard(ComponentBase):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 Kc:float,
                 Ko:float,
                 CQ:float,
                 R :float=0.0,
                 L :float=0.0,
                 RRA:float=0.0, 
                 *args, **kwargs
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False) 
        # setting the bernoulli resistance value
        self.CQ = CQ
        # setting the resistance value
        self.R = R
        # setting the valve impedance value
        self.L  = L
        # setting the relative regurgitant area
        self.RRA = RRA
        # setting the rate of valve opening and closing
        self.Kc, self.Ko = Kc, Ko
        # defining the valve opening factor state variable
        self._PHI = StateVariable(name=name+'_PHI', timeobj=time_object)
        
    @property
    def PHI(self):
        return self._PHI._u
        
    def q_i_u_func(self, t, y):
        return maynard_valve_flow(t, y=y, CQ=self.CQ, RRA=self.RRA)
    
    def q_i_dudt_func(self, t, y):
        return maynard_impedance_dqdt(t, y=y, CQ=self.CQ, RRA=self.RRA, L=self.L, R=self.R)
    
    def phi_dudt_func(self, t, y):
        return maynard_phi_law(t, y=y, Ko=self.Ko, Kc=self.Kc)
        
    def setup(self) -> None:
        if self.L < 1.0e-6:
            self._Q_i.set_u_func(self.q_i_u_func, function_name='maynard_valve_flow')
            self._Q_i.set_inputs(pd.Series({'p_in' : self._P_i.name, 
                                            'p_out': self._P_o.name,
                                            'phi'  : self._PHI.name}))
        else:
            self._Q_i.set_dudt_func(self.q_i_dudt_func, function_name='maynard_impedance_dqdt')
            self._Q_i.set_inputs(pd.Series({'p_in' : self._P_i.name, 
                                            'p_out': self._P_o.name,
                                            'q_in' : self._Q_i.name,
                                            'phi'  : self._PHI.name}))
        
        self._PHI.set_dudt_func(self.phi_dudt_func, function_name='maynard_phi_law')
        self._PHI.set_inputs(pd.Series({'p_in' : self._P_i.name,
                                        'p_out': self._P_o.name,
                                        'phi'  : self._PHI.name}))