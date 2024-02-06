from .Time import TimeClass
from .HelperRoutines import *
from .StateVariable import StateVariable

class Component():
    def __init__(self,
                 name,
                 time_object:TimeClass, 
                 ) -> None:
        self._name     = name
        self._to      = time_object
        
        self._P_i = StateVariable(name=name+'_P_i', timeobj=time_object)
        self._Q_i = StateVariable(name=name+'_Q_i', timeobj=time_object)
        self._P_o = StateVariable(name=name+'_P_o', timeobj=time_object)
        self._Q_o = StateVariable(name=name+'_Q_o', timeobj=time_object)
        self._V   = StateVariable(name=name+'_V' , timeobj=time_object)
        
        # self._V.set_dudt_func(function=chamber_volume_rate_change, 
        #                       function_name='chamber_volume_rate_change')
        return
        
    def __repr__(self) -> str:
        var = (f" Component {self._name}" + '\n' +
               f" - P_i " + str(self._P_i) + '\n' +
               f" - Q_i " + str(self._Q_i) + '\n' +
               f" - P_o " + str(self._P_o) + '\n' +
               f" - Q_o " + str(self._Q_o) + '\n'
               )
        return var
                
    @property
    def P_i(self):
        return self._P_i._u
    
    @property
    def P_o(self):
        return self._P_o._u
    
    @property
    def Q_i(self):
        return self._Q_i._u
    
    @property
    def Q_o(self):
        return self._Q_o._u
    
    @property
    def V(self):
        return self._V._u
    
    @property
    def global_state_variable_names(self):
        return {
            'P_i' : self._P_i.name,
            'P_o' : self._P_o.name,
            'Q_i' : self._Q_i.name,
            'Q_o' : self._Q_o.name
            }
    
    def comp_dvdt(self, intq:int=None, q_i:float=None, q_o:float=None):
        if intq is not None:
            return self.Q_i[intq] - self.Q_o[intq]
        elif q_i is not None and q_o is not None:
            return q_i - q_o
        
    def make_unique_io_state_variable(self, q_flag:bool=False, p_flag:bool=True) -> None:
        if q_flag: 
            self._Q_o = self._Q_i
            self._Q_o.set_name(self._name + '_Q')
        if p_flag: 
            self._P_o = self._P_i
            self._P_o.set_name(self._name + '_P')
        return 
    
    def setup(self) -> None:
        raise Exception("This is a template class only.")
            
class Rc_component(Component):
    def __init__(self, 
                 name:str, 
                 time_object:TimeClass, 
                #  main_var:str, 
                 r:float, 
                 c:float, 
                 v_ref:float
                 ) -> None:
        # super().__init__(time_object, main_var)
        super().__init__(time_object=time_object, name=name)
        self.R = r
        self.C = c
        self.V_ref = v_ref
        
    def comp_dpdt(self, intq:int=None, q_in:float=None, q_out:float=None):
        if intq is None and q_in is None and q_out is None:
            raise Exception("You have to either asign a value to intq or to q_in and q_out.")
        elif intq is None and q_in is None and q_out is not None:
            raise Exception("If you define dpdt in terms of q_in and q_out both have to be assigned.")
        elif intq is None and q_in is not None and q_out is None:
            raise Exception("If you define dpdt in terms of q_in and q_out both have to be assigned.")
        
        if intq is not None:    
            return 1.0 /  self.C * (self.Q_i[intq] - self.Q_o[intq])
        elif q_in is not None and q_out is not None:
            return 1.0 / self.C * (q_in - q_out)
        else: 
            raise Exception("Input case not covered.")
    
    def comp_p_i(self, intv:int):
        return (self.V[intv] - self.V_ref) / self.C
    
    def comp_q_o(self, intp:int=None, p_i:float=None, p_o:float=None):
        if intp is not None:
            return (self.P_i[intp] - self.P_o[intp]) / self.R
        elif p_i is not None and p_o is not None:
            return (p_i - p_o) / self.R
        else:
            raise Exception("Input case not covered")
    
    def setup(self) -> None:
        self._P_i.set_dudt_func(lambda t, q_in, q_out: grounded_capacitor_model_pressure(t=t, q_in=q_in, q_out=q_out, c=self.C),
                                function_name='lambda grounded_capacitor_model_pressure')
        self._P_i.set_inputs({'q_in' :self._Q_i.name, 
                              'q_out':self._Q_o.name})
        self._Q_o.set_u_func(lambda t, p_in, p_out : resistor_model_flow(t, p_in=p_in, p_out=p_out, r=self.R),
                             function_name='lambda resistor_model_flow')
        self._Q_o.set_inputs({'p_in':self._P_i.name, 
                              'p_out':self._P_o.name})
        self._V.set_dudt_func(lambda t, q_in, q_out : chamber_volume_rate_change(t=t, q_in=q_in, q_out=q_out),
                              function_name='lambda chamber_volume_rate_change')
        self._V.set_inputs({'q_in':self._Q_i.name, 
                            'q_out':self._Q_o.name}) 
    
    
class Valve_non_ideal(Component):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 r:float, 
                 max_func
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        # allow for pressure gradient but not for flow
        self.make_unique_io_state_variable(q_flag=True, p_flag=False) 
        # setting the resistance value
        self.R = r
        self.max_func = max_func
        
    def comp_q(self, intp:int=None, p_in:float=None, p_out:float=None):
        if intp is not None:
            return self.max_func(self.P_i[intp] - self.P_o[intp]) / self.R
        elif p_in is not None and p_out is not None:
            return (p_in - p_out) / self.R
        
    def setup(self) -> None:
        self._Q_i.set_u_func(lambda t, p_in, p_out : resistor_model_flow(t=t ,p_in=p_in, p_out=p_out, r=self.R),
                             function_name='lambda resistor_model_flow')
        self._Q_i.set_inputs({'p_in':self._P_i.name, 
                              'p_out':self._P_o.name})
        
    
class HC_constant_elastance(Component):
    def __init__(self, 
                 name:str,
                 time_object: TimeClass,
                 E_pas: float, 
                 E_act: float,
                 V_ref: float,
                 activation_function_template = activation_function_1,
                 *args, **kwargs
                 ) -> None:
        super().__init__(name=name, time_object=time_object)
        self._af = lambda t : activation_function_template(t, *args, **kwargs)
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
        e = self.comp_E(self._to._sym_t_norm[intt])
        v = self.V[intv]
        return e * (v - self.V_ref)
    
    def comp_dpdt(self, intt:int=None, intq:int=None, t:float=None, V:float=None, q_i:float=None, q_o:float=None) -> float:
        if intt is not None:
            dEdt = self.comp_dEdt(self._to._sym_t_norm[intt])
            e    = self.comp_E(self._to._sym_t_norm[intt])
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
            return dEdt(V - self.V_ref) + e * (q_i - q_o)
        else: 
            raise Exception("Input case not covered.")
        return
    
    def setup(self) -> None:
        self._V.set_dudt_func(lambda t, q_in, q_out : chamber_volume_rate_change(t, q_in=q_in, q_out=q_out),
                              function_name='lambda chamber_volume_rate_change')
        self._V.set_inputs({'q_in':self._Q_i.name, 
                            'q_out':self._Q_o.name})
        self._P_i.set_dudt_func(lambda t, V, q_i, q_o: self.comp_dpdt(t=t, V=V, q_i=q_i, q_o=q_o),
                                function_name='lamda constant elastance dpdt') # setup to be reviewed
        self._P_i.set_inputs({'V':self._V.name, 
                              'q_i':self._Q_i.name, 
                              'q_o':self._Q_o.name})