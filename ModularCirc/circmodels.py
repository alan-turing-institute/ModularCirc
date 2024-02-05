import numpy as np
import sympy as sp
# from .timeclass import TimeSeries
from .HelperRoutines import *
from .Time import *
from .StateVariable import StateVariable
from .Component import *
from .Solver import *

        
        
class OdeModel():
    def __init__(self, time_setup_dict) -> None:
        self.time_object = TimeClass(time_setup_dict=time_setup_dict)
        self._state_variable_dict = dict()
        self.commponents = dict()
        self.name = 'Template'
        
        self.solver = Solver(time_object=self.time_object, state_variable_dictionary= self._state_variable_dict)
    
    def connect_modules(self, 
                        module1:Component, 
                        module2:Component,
                        pvariable:StateVariable = None,
                        qvariable:StateVariable = None,
                        plabel:str=None, 
                        qlabel:str=None
                        ) ->None:   
        """
        Method for connecting Component modules.
        
        Inputs
        ------
        module1 (Component) : upstream Component
        module2 (Component) : downstream Component
        plabel (str) : new name for the shared pressure state variable
        qlabel (str) : new name for the shared flow state variable
        """
        if qvariable is None:
            print(f" {module1._Q_o._ode_sys_mapping['u_func']} : {module1._Q_o._ode_sys_mapping['dudt_func']} : {module2._Q_i._ode_sys_mapping['u_func']} : {module2._Q_i._ode_sys_mapping['dudt_func']}")
            if module1._Q_o._ode_sys_mapping['u_func'] is not None or module1._Q_o._ode_sys_mapping['dudt_func'] is not None:
                module2._Q_i = module1._Q_o
            elif module2._Q_i._ode_sys_mapping['u_func'] is not None or module2._Q_i._ode_sys_mapping['dudt_func'] is not None:
                module1._Q_o = module2._Q_i
            else:
                raise Exception(f'Definition of flow between modules {module1._name} and {module2._name} is ambiguous.')
        else:
            module2._Q_i = qvariable
            module1._Q_o = qvariable
            
        if pvariable is None:
            if module1._P_o._ode_sys_mapping['u_func'] is not None or module1._P_o._ode_sys_mapping['dudt_func'] is not None:
                module2._P_i = module1._P_o
            elif module2._P_i._ode_sys_mapping['u_func'] is not None or module2._P_i._ode_sys_mapping['dudt_func'] is not None:
                module1._P_o = module2._P_i
            else:
                raise Exception(f'Definition of pressure between modules {module1._name} and {module2._name} is ambiguous')
        else:
            module2._P_i = pvariable
            module1._P_o = pvariable
             
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
        self._state_variable_dict['v_la'] = self.commponents['la']._V
        self._state_variable_dict['v_la'].set_name('v_la')
        
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
        self._state_variable_dict['v_lv'] = self.commponents['lv']._V
        self._state_variable_dict['v_lv'].set_name('v_lv')
        
        for component in self.commponents.values():
            component.setup()
        
        # connect the left ventricle class to the aortic valve
        self.connect_modules(self.commponents['lv'],  
                             self.commponents['av'],  
                             plabel='p_lv',   
                             qlabel='q_av',
                            #  pvariable=self.commponents['lv']._P_o,
                            #  qvariable=self.commponents['av']._Q_i
                             )
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
                             qlabel='q_ven',
                            #  qvariable=self.commponents['ven']._Q_o,
                            #  pvariable=self.commponents['la']._P_i
                             )
        # connect the left atrium to the mitral valve
        self.connect_modules(self.commponents['la'],  
                             self.commponents['mv'],  
                             plabel= 'p_la', 
                             qlabel='q_mv')
        # connect the mitral valve to the left ventricle
        self.connect_modules(self.commponents['mv'],  
                             self.commponents['lv'],  
                             plabel='p_lv',  
                             qlabel='q_mv',
                            #  pvariable=self.commponents['lv']._P_o,
                            #  qvariable=self.commponents['mv']._Q_i
                             )
            
        self.solver.setup()
        
        
        