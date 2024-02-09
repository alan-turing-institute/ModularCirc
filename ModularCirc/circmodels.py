import numpy as np
import sympy as sp
from .HelperRoutines import *
from .Time import *
from .StateVariable import StateVariable
from .Component import *
        
        
class OdeModel():
    def __init__(self, time_setup_dict) -> None:
        self.time_object = TimeClass(time_setup_dict=time_setup_dict)
        self._state_variable_dict = pd.Series()
        self.all_sv_data = pd.DataFrame(index=self.time_object.time.index, dtype='float64')
        self.commponents = dict()
        self.name = 'Template'
        
    
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
            self.all_sv_data[plabel] = module1.P_o
            module1._P_o._u = self.all_sv_data[plabel]
        if qlabel is not None:
            module1._Q_o.set_name(qlabel)
            self._state_variable_dict[qlabel] = module1._Q_o
            self.all_sv_data[qlabel] = module1.Q_o
            module1._Q_o._u = self.all_sv_data[qlabel]
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
    
    
class NaghaviModelParameters():
    def __init__(self) -> None:
        components = ['ao', 'art', 'ven', 'av', 'mv', 'la', 'lv']
        self.components = {key : None for key in components}
        for key in ['ao', 'art', 'ven']:
            self.components[key] = pd.Series(index=['r', 'c', 'v_ref', 'v'], dtype='float64')
        for key in ['av', 'mv']:
            self.components[key] = pd.Series(index=['r', 'max_func'], dtype=object)
        for key in ['la', 'lv']:
            self.components[key] = pd.Series(index=['E_pas', 'E_act', 'V_ref', 't_max', 't_tr', 'tau', 'V'], dtype='float64')
                        
        # self.components['ao'].loc[:] =  [32000., 0.0025, 100., 0.025*5200.0]
        self.components['ao'].loc[:] =  [32000., 0.0025, 100., 0.025*5200.0]
        self.components['art'].loc[:] = [150000., 0.025, 50., 0.21*5200.0]
        self.components['ven'].loc[:] = [1200., 1., 2800., 0.727*5200.0]
        
        self.components['av'].loc[:] = [800., relu_max]
        self.components['mv'].loc[:] = [550., relu_max]        
        
        # self.components['av'].loc[:] = [800., get_softplus_max(0.2)]
        # self.components['mv'].loc[:] = [550., get_softplus_max(0.2)]
        
        # original
        # self.components['la'].loc[:] = [60., 0.44/0.0075, 10., 150., 1.5*150., 25., 0.018*5200.0]
        # self.components['lv'].loc[:] = [400, 1./0.0075, 10., 280., 1.5*280., 25., 0.02*5200.0]
        
        self.components['la'].loc[:] = [60., 0.44/0.0075, 10., 150., 1.5*150., 50., 0.018*5200.0]
        self.components['lv'].loc[:] = [400, 1./0.0075  , 10., 280., 1.5*280., 50., 0.02*5200.0]
        
    def __repr__(self) -> str:
        out = 'Naghavi Model parameter set: \n'
        for comp in self.components: 
            out += f" * Component - {comp}" + '\n'
            for key, item in self.components[comp].items():
                out += (f"  - {key:<10} : {item}") + '\n'
            out += '\n'
        return out
    
    def __getitem__(self, key):
        return self.components[key]
    
    def set_rc_comp(self, key:str, r:float=None, c:float=None, v_ref:float=None, v:float=None)->None:
        if key not in ['ao','art', 'ven']:
            raise Exception('Wrong key!')
        if r is not None:
            self.components[key].loc['r'] = r
        if c is not None:
            self.components[key].loc['c'] = c
        if v_ref is not None:
            self.components[key].loc['v_ref'] = v_ref
        if v is not None:
            self.components[key].loc['v'] = v
        return
            
    def set_valve_comp(self, key:str, r:float=None, max_func=None)->None:
        if key not in ['av', 'mv']:
            raise Exception('Wrong key!')
        if r is not None:
            self.components[key].loc['r'] = r
        if max_func is not None:
            self.components[key].loc['max_func'] = max_func
        return
    
    def set_chamber_comp(self, 
                         key:str, 
                         E_pas:float=None, 
                         E_act:float=None, 
                         V_ref:float=None, 
                         t_max:float=None, 
                         t_tr:float=None, 
                         tau:float=None
                         )->None:
        if key not in ['lv', 'la']:
            raise Exception('Wrong key!')
        if E_pas is not None:
            self.components[key].loc['E_pas'] = E_pas
        if E_act is not None:
            self.components[key].loc['E_act'] = E_act
        if V_ref is not None:
            self.components[key].loc['V_ref'] = V_ref
        if t_max is not None:
            self.components[key].loc['t_max'] = t_max
        if t_tr is not None:
            self.components[key].loc['t_tr'] = t_tr
        if tau is not None:
            self.components[key].loc['tau'] = tau
        
    
class NaghaviModel(OdeModel):
    def __init__(self, time_setup_dict, parobj:NaghaviModelParameters=NaghaviModelParameters()) -> None:
        super().__init__(time_setup_dict)
        self.name = 'NaghaviModel'
        
        base = NaghaviModelParameters()
        print(base)
                
        # Defining the aorta object
        self.commponents['ao'] = Rc_component( name='Aorta',
                                time_object=self.time_object, 
                                r     = parobj['ao']['r'],
                                c     = parobj['ao']['c'], 
                                v_ref = parobj['ao']['v_ref'],
                                v     = parobj['ao']['v']
                                  )
        # self.commponents['ao']._V.set_name('v_ao')     ##### test
        self._state_variable_dict['v_ao'] = self.commponents['ao']._V
        self._state_variable_dict['v_ao'].set_name('v_ao')
        self.all_sv_data['v_ao'] = self.commponents['ao'].V     ##### test
        self.commponents['ao']._V._u = self.all_sv_data['v_ao']
        
        # Defining the arterial system object
        self.commponents['art'] = Rc_component(name='Arteries',
                                time_object=self.time_object,
                                r     = parobj['art']['r'],
                                c     = parobj['art']['c'], 
                                v_ref = parobj['art']['v_ref'],
                                v     = parobj['art']['v']
                                )
        self._state_variable_dict['v_art'] = self.commponents['art']._V
        self._state_variable_dict['v_art'].set_name('v_art')
        self.all_sv_data['v_art'] = self.commponents['art'].V
        self.commponents['art']._V._u = self.all_sv_data['v_art']
        
        # Defining the venous system object
        self.commponents['ven'] = Rc_component(name='VenaCava',
                                time_object=self.time_object,
                                r     = parobj['ven']['r'],
                                c     = parobj['ven']['c'], 
                                v_ref = parobj['ven']['v_ref'],
                                v     = parobj['ven']['v']
                                )
        self._state_variable_dict['v_ven'] = self.commponents['ven']._V
        self._state_variable_dict['v_ven'].set_name('v_ven')
        self.all_sv_data['v_ven'] = self.commponents['ven'].V
        self.commponents['ven']._V._u = self.all_sv_data['v_ven']
                
        # Defining the aortic valve object
        self.commponents['av']  = Valve_non_ideal(name='AorticValve',
                                     time_object=self.time_object,
                                     r=parobj['av']['r'],
                                     max_func=parobj['av']['max_func']
                                     )
        
        # Defining the mitral valve object
        self.commponents['mv'] = Valve_non_ideal(name='MitralValve',
                                    time_object=self.time_object,
                                    r=parobj['mv']['r'],
                                    max_func=parobj['mv']['max_func']
                                    )
        
        # Defining the left atrium activation function
        la_af = lambda t: activation_function_2(t=time_shift(t, 100., self.time_object), 
                                                t_max=parobj['la']['t_max'], 
                                                t_tr=parobj['la']['t_tr'], 
                                                tau=parobj['la']['tau']
                                                )
        # Defining the left atrium class
        self.commponents['la'] = HC_constant_elastance(name='LeftAtrium',
                                        time_object=self.time_object,
                                        E_pas=parobj['la']['E_pas'],
                                        E_act=parobj['la']['E_act'],
                                        V_ref=parobj['la']['V_ref'],
                                        v    =parobj['la']['V'],
                                        activation_function_template=la_af
                                        )
        self._state_variable_dict['v_la'] = self.commponents['la']._V
        self._state_variable_dict['v_la'].set_name('v_la')
        self.all_sv_data['v_la'] = self.commponents['la'].V
        self.commponents['la']._V._u = self.all_sv_data['v_la']
        
        # Defining the left ventricle activation function
        lv_af = lambda t: activation_function_2(t=t,
                                                t_max=parobj['lv']['t_max'], 
                                                t_tr=parobj['lv']['t_tr'], 
                                                tau=parobj['lv']['tau']
                                                )
        self.commponents['lv'] = HC_constant_elastance(name='LeftVentricle', 
                                        time_object=self.time_object,
                                        E_pas=parobj['lv']['E_pas'],
                                        E_act=parobj['lv']['E_act'],
                                        V_ref=parobj['lv']['V_ref'],
                                        v    =parobj['lv']['V'],
                                        activation_function_template=lv_af
                                        )
        self._state_variable_dict['v_lv'] = self.commponents['lv']._V
        self._state_variable_dict['v_lv'].set_name('v_lv')
        self.all_sv_data['v_lv'] = self.commponents['lv'].V
        self.commponents['lv']._V._u = self.all_sv_data['v_lv']
        
        for component in self.commponents.values():
            component.setup()
        
        # connect the left ventricle class to the aortic valve
        self.connect_modules(self.commponents['lv'],  
                             self.commponents['av'],  
                             plabel='p_lv',   
                             qlabel='q_av',
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
                             )
        
        for component in self.commponents.values():
            component.setup()
                    
        
        