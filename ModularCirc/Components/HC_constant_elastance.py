from .ComponentBase import ComponentBase
from ..HelperRoutines import activation_function_1, \
    chamber_volume_rate_change, \
        chamber_pressure_function, \
            chamber_linear_elastic_law
from ..Time import TimeClass

import pandas as pd

class HC_constant_elastance(ComponentBase):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 E_pas: float, 
                 E_act: float,
                 V_ref: float,
                 v    : float = None,
                 activation_function_template = activation_function_1,
                 *args, **kwargs
                 ) -> None:
        super().__init__(name=name, time_object=time_object, v=v)
        self._af = lambda t : activation_function_template(t)
        self.E_pas = E_pas
        self.E_act = E_act
        self.V_ref = V_ref
        self.eps = 1.0e-3
        
        self.make_unique_io_state_variable(p_flag=True, q_flag=False)
        
    def comp_E(self, t:float) -> float:
        return self._af(t) * self.E_act + (1.0 - self._af(t)) * self.E_pas
    
    def comp_dEdt(self, t:float) -> float:
        return (self.comp_E(t + self.eps) - self.comp_E(t - self.eps)) / 2.0 / self.eps
    
    def comp_p(self, intt:int, intv:int):
        e = self.comp_E(self._to.time['real_t'].iloc[intt])
        v = self.V[intv]
        return e * (v - self.V_ref)
    
    def comp_dpdt(self, intt:int=None, intq:int=None, t:float=None, V:float=None, q_i:float=None, q_o:float=None) -> float:
        if intt is not None:
            dEdt = self.comp_dEdt(self._to.time['real_t'].iloc[intt])
            e    = self.comp_E(self._to.time['real_t'].iloc[intt])
        elif t is not None:
            dEdt = self.comp_dEdt(t)
            e    = self.comp_E(t)
        else:
            raise Exception("Input case not covered.")
        if intq is not None:
            dvdt = self.comp_dvdt(intq=intq)
            v    = self.V[intq]
            return dEdt * (v - self.V_ref) + e * dvdt
        elif V is not None and q_i is not None and q_o is not None:
            return dEdt * (V - self.V_ref) + e * (q_i - q_o)
        else: 
            raise Exception("Input case not covered.")
        return
    
    def setup(self) -> None:
        self._V.set_dudt_func(lambda t, q_in, q_out : chamber_volume_rate_change(t, q_in=q_in, q_out=q_out),
                              function_name='lambda chamber_volume_rate_change')
        self._V.set_inputs(pd.Series({'q_in':self._Q_i.name, 
                                      'q_out':self._Q_o.name}))
        self._P_i.set_dudt_func(lambda t, V, q_i, q_o: self.comp_dpdt(t=t, V=V, q_i=q_i, q_o=q_o),
                                function_name='lamda constant elastance dpdt') # setup to be reviewed
        self._P_i.set_inputs(pd.Series({'V':self._V.name, 
                                        'q_i':self._Q_i.name, 
                                        'q_o':self._Q_o.name}))
        self._P_i.set_i_func(lambda V: chamber_pressure_function(t=0, v=V, v_ref=self.V_ref, 
                                                                 E_pas=self.E_pas, E_act=self.E_act,
                                                                 activation_function=self._af,
                                                                 active_law = chamber_linear_elastic_law,
                                                                 passive_law= chamber_linear_elastic_law),
                             function_name='lamda chamber_pressure_function + activation_function_1 + 2xchamber_linear_elastic_law')
        self._P_i.set_i_inputs(pd.Series({'V':self._V.name}))