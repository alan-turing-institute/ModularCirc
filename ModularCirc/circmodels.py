import numpy as np
import sympy as sp
# from .timeclass import TimeSeries
from .HelperRoutines import *
from .Time import *
from .StateVariable import StateVariable
        

class Chamber():
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
        
        self._V.set_dudt_func(function=chamber_volume_rate_change)
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
            
class Rc_component(Chamber):
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
        self._P_i.set_dudt_func(lambda q_in, q_out: grounded_capacitor_model_pressure(q_in=q_in, q_out=q_out, c=self.C))
        self._P_i.set_inputs([self._Q_i.name, self._Q_o.name])
        self._Q_o.set_u_func(lambda p_in, p_out : resistor_model_flow(p_in=p_in, p_out=p_out, r=self.R))
        self._Q_o.set_inputs([self._P_i.name, self._P_o.name])
        self._V.set_dudt_func(chamber_volume_rate_change)
        self._V.set_inputs([self._Q_i.name, self._Q_o.name])
    
    
class Valve_non_ideal(Chamber):
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
        self._Q_i.set_u_func(lambda p_in, p_out : resistor_model_flow(p_in=p_in, p_out=p_out, r=self.R))
        self._Q_i.set_inputs([self._P_i.name, self._P_o.name])
        
    
class HC_constant_elastance(Chamber):
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
        self._V.set_dudt_func(chamber_volume_rate_change)
        self._V.set_inputs([self._Q_i.name, self._Q_o.name])
        self._P_i.set_dudt_func(lambda t, V, q_i, q_o: self.comp_dpdt(V=V, q_i=q_i, q_o=q_o)) # setup to be reviewed
        self._P_i.set_inputs(['Time', self._V.name, self._Q_i.name, self._Q_o.name])
        
class Solver():
    def __init__(self, type:str=None, time_object:TimeClass=None) -> None:
        self.type = None
        self._pvk = []
        self._svk = []
        
    def setup(self, state_variable_dictionary:dict)->None:
        print('Blah')
        for key, component in state_variable_dictionary.items():
            print(f' * {key}')
            
        
        
class OdeModel():
    def __init__(self, time_setup_dict) -> None:
        self.time_object = TimeClass(time_setup_dict=time_setup_dict)
        self._state_variable_dict = dict()
        self.commponents = dict()
        self.name = 'Template'
        
        self.solver = Solver(time_object=self.time_object)
    
    def connect_modules(self, 
                        module1:Chamber, 
                        module2:Chamber,
                        pvariable:StateVariable = None,
                        qvariable:StateVariable = None,
                        plabel:str=None, 
                        qlabel:str=None
                        ) ->None:   
        """
        Method for connecting chamber modules.
        
        Inputs
        ------
        module1 (Chamber) : upstream chamber
        module2 (Chamber) : downstream chamber
        plabel (str) : new name for the shared pressure state variable
        qlabel (str) : new name for the shared flow state variable
        """
        module2._Q_i = module1._Q_o
        module2._P_i = module1._P_o
        if plabel is not None:
            module1._P_o.set_name(plabel)
            self._state_variable_dict[plabel] = module1._P_o
        if qlabel is not None:
            module1._Q_o.set_name(qlabel)
            self._state_variable_dict[qlabel] = module1._Q_o
        return
        
            
    @property
    def state_variable_dict(self):
        return self._state_variable_dict
    
    def __str__(self) -> str:
        out = f'Model {self.name} \n\n'
        for component in self.commponents.values():
            out += (str(component) + '\n')
        out += '\n'
        out += 'Main State Variable Dictionary \n'
        for key in self._state_variable_dict.keys():
            out += f" - {key} \n"
        return out

    
class NaghaviModel(OdeModel):
    def __init__(self, time_setup_dict,) -> None:
        super().__init__(time_setup_dict)
        self.name = 'NaghaviModel'
                
        # Defining the aorta object
        self.commponents['ao'] = Rc_component( name='Aorta',
                                time_object=self.time_object, 
                                r = 1.0,
                                c = 1.0, 
                                v_ref = 1.0,
                                  )
        
        # Defining the arterial system object
        self.commponents['art'] = Rc_component(name='Arteries',
                                time_object=self.time_object,
                                r = 1.0,
                                c = 1.0,
                                v_ref= 1.0,
                                )
        
        # Defining the venous system object
        self.commponents['ven'] = Rc_component(name='VenaCava',
                                time_object=self.time_object,
                                r = 1.0,
                                c = 1.0,
                                v_ref= 1.0,
                                )
        
        # Defining the aortic valve object
        self.commponents['av']  = Valve_non_ideal(  name='AorticValve',
                                     time_object=self.time_object,
                                     r=1.0,
                                     max_func=relu_max
                                     )
        
        # Defining the mitral valve object
        self.commponents['mv'] = Valve_non_ideal(  name='MitralValve',
                                    time_object=self.time_object,
                                    r=1.0,
                                    max_func=relu_max
                                    )
        
        # Defining the left atrium activation function
        la_af = lambda t: activation_function_1(t=t, 
                                                t_max=1.0, 
                                                t_tr=1.0, 
                                                tau=1.0
                                                )
        # Defining the left atrium class
        self.commponents['la'] = HC_constant_elastance(name='LeftAtrium',
                                        time_object=self.time_object,
                                        E_pas=1.0,
                                        E_act=10.0,
                                        V_ref=1.0,
                                        activation_function_template=la_af
                                        )
        
        # Defining the left ventricle activation function
        lv_af = lambda t: activation_function_1(t=t,
                                                t_max=1.0,
                                                t_tr=1.0,
                                                tau=1.0
                                                )
        self.commponents['lv'] = HC_constant_elastance(name='LeftVentricle', 
                                        time_object=self.time_object,
                                        E_pas=1.0,
                                        E_act=10.0,
                                        V_ref=1.0,
                                        activation_function_template=lv_af
                                        )
        
        
        # connect the left ventricle class to the aortic valve
        self.connect_modules(self.commponents['lv'],  
                             self.commponents['av'],  
                             plabel='p_lv',   
                             qlabel='q_av')
        # connect the aortic valve to the aorta
        self.connect_modules(self.commponents['av'],  
                             self.commponents['ao'],  
                             plabel='p_ao',   
                             qlabel='q_av')
        # connect the aorta to the arteries
        self.connect_modules(self.commponents['ao'],  
                             self.commponents['art'], 
                             plabel= 'p_art', 
                             qlabel= 'q_ao')
        # connect the arteries to the veins
        self.connect_modules(self.commponents['art'], 
                             self.commponents['ven'], 
                             plabel= 'p_ven', 
                             qlabel= 'q_art')
        # connect the veins to the left atrium
        self.connect_modules(self.commponents['ven'], 
                             self.commponents['la'],  
                             plabel= 'p_la',  
                             qlabel='q_ven')
        # connect the left atrium to the mitral valve
        self.connect_modules(self.commponents['la'],  
                             self.commponents['mv'],  
                             plabel= 'p_la', 
                             qlabel='q_mv')
        # connect the mitral valve to the left ventricle
        self.connect_modules(self.commponents['mv'],  
                             self.commponents['lv'],  
                             plabel='p_lv',  
                             qlabel='q_mv')
        
        for component in self.commponents.values():
            component.setup()
            
        self.solver.setup(state_variable_dictionary=self.state_variable_dict,)
        
        
        