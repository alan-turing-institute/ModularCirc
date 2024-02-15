from ..HelperRoutines import *
from .ParametersObject import ParametersObject
import pandas as pd

class NaghaviModelParameters(ParametersObject):
    def __init__(self, name='Naghavi Model') -> None:
        super().__init__(name=name)
        components = ['ao', 'art', 'ven', 'av', 'mv', 'la', 'lv']
        self.components = {key : None for key in components}
        for key in ['ao', 'art', 'ven']:
            self.components[key] = pd.Series(index=['r', 'c', 'l', 'v_ref', 'v', 'p'], dtype=object)
        for key in ['av', 'mv']:
            self.components[key] = pd.Series(index=['r', 'max_func'], dtype=object)
        for key in ['la', 'lv']:
            self.components[key] = pd.Series(index=['E_pas', 
                                                    'E_act', 
                                                    'V_ref', 
                                                    'activation_function', 
                                                    't_max', 
                                                    't_tr', 
                                                    'tau',
                                                    'delay',
                                                    'V',
                                                    'P'], dtype=object)
                        
        self.set_rlc_comp(key='ao', r=32000., c=0.0025, l=0.0, v_ref=100., v=0.025*5200.0, p=None)
        self.set_rlc_comp(key='art', r=150000., c=0.025, l=0.0, v_ref=50. , v=0.025*5200.0, p=None)
        self.set_rlc_comp(key='ven', r=1200., c=1.000, l=0.0, v_ref=2800. , v=0.727*5200.0, p=None)
        
        self.set_valve_comp(key='av', r=800., max_func=relu_max)
        self.set_valve_comp(key='mv', r=550., max_func=relu_max)     
        
        
        # original
        self.set_chamber_comp('la', E_pas=60., E_act=0.44/0.0075, V_ref=10.,
                              activation_function=activation_function_2,
                              t_max=150., t_tr=1.5*150., tau=175., delay=100., V=0.018*5200.0, P=None)

        self.set_chamber_comp('lv', E_pas=400., E_act=1./0.0075, V_ref=10.,
                              activation_function=activation_function_2,
                              t_max=280., t_tr=1.5*280., tau=305., V=0.028*5200.0, P=None)
    
    def set_rc_comp(self, key:str, **kwargs):
        self._set_comp(key=key, set=['ao','art', 'ven'], **kwargs)
        
    def set_rlc_comp(self, key:str, **kwargs):
        self._set_comp(key=key, set=['ao','art', 'ven'], **kwargs)
        
    def set_valve_comp(self, key:str, **kwargs):
        self._set_comp(key=key, set=['av', 'mv'], **kwargs)
            
    
    def set_chamber_comp(self, key:str, **kwargs):
        self._set_comp(key=key, set=['lv', 'la'], **kwargs)
         
    def set_activation_function(self, key:str, activation_func=activation_function_2):
        self._set_comp(key=key, set=['lv', 'la'], activation_func=activation_function_2)