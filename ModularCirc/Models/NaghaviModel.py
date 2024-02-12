from .OdeModel import OdeModel
from .NaghaviModelParameters import NaghaviModelParameters
from ..Components import Rc_component, Valve_non_ideal, HC_constant_elastance
from ..HelperRoutines import *

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