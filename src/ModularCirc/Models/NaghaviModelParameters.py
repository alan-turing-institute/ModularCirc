from ..HelperRoutines import *
from .ParametersObject import ParametersObject
import pandas as pd

TEMPLATE_TIME_SETUP_DICT = {
        "name": "TimeTest",
        "ncycles": 30,
        "tcycle": 1000.,
        "dt": 1.,
        "export_min": 1,
    }

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
                                                    'v_ref',
                                                    'k_pas',
                                                    'activation_function',
                                                    't_tr',
                                                    't_max',
                                                    'tau',
                                                    'delay',
                                                    'v',
                                                    'p',
                                                    'activation_func'], dtype=object)

        self.set_rlc_comp(key='ao',  r=240.,  c=0.3,  l=0.0, v_ref=100.,  v=0.025*5200.0, p=None)
        self.set_rlc_comp(key='art', r=1125., c=3.0,  l=0.0, v_ref=900.,  v=0.21 *5200.0, p=None)
        self.set_rlc_comp(key='ven', r=9.0 ,  c=133.3,l=0.0, v_ref=2800., v=0.727*5200.0, p=None)

        self.set_valve_comp(key='av', r=6.  , max_func=relu_max)
        self.set_valve_comp(key='mv', r=4.1,  max_func=relu_max)


        # original
        self.set_chamber_comp('la', E_pas=0.44, E_act=0.45, v_ref=10., k_pas=0.05, # 0.05
                              activation_function=activation_function_1,
                              t_tr=225., t_max=150., tau=25., delay=100., v=0.018*5200.0, p=None)

        self.set_chamber_comp('lv', E_pas=1.0, E_act=3.0, v_ref=10., k_pas=0.027, # 0.027
                              activation_function=activation_function_1,
                              t_tr=420., t_max=280., tau=25., delay=None, v=0.02*5200.0, p=None)

    def set_rc_comp(self, key:str, **kwargs):
        self._set_comp(key=key, set=['ao','art', 'ven'], **kwargs)

    def set_rlc_comp(self, key:str, **kwargs):
        self._set_comp(key=key, set=['ao','art', 'ven'], **kwargs)

    def set_valve_comp(self, key:str, **kwargs):
        self._set_comp(key=key, set=['av', 'mv'], **kwargs)


    def set_chamber_comp(self, key:str, **kwargs):
        self._set_comp(key=key, set=['lv', 'la'], **kwargs)

    def set_activation_function(self, key:str, activation_func=activation_function_2):
        self._set_comp(key=key, set=['lv', 'la'], activation_func=activation_func)
