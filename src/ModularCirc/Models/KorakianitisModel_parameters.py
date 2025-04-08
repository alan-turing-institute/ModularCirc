from ..HelperRoutines import activation_function_2, activation_function_3
from .ParametersObject import ParametersObject
import pandas as pd


KORAKIANITIS_2006_COMPONENTS = [
              'la', # left atrium
              'mi', # mitral valve
              'lv', # left ventricle
              'ao', # aortic valve
              'sas', # systemic aortic sinus
              'sat', # systemic artery
              'svn', # systemic vein
              'ra',  # right atrium
              'ti',  # tricuspid vale
              'rv',  # right ventricle
              'po',  # pulmonary valve
              'pas', # pulmonary artery sinus
              'pat', # pulmonary artery 
              'pvn'  # pulmonary vein
              ]
VESSELS = ['sas', 'sat', 'svn', 'pas', 'pat', 'pvn']
VESSELS_PAR = ['r', 'c', 'l', 'v_ref', 'v', 'p']

VALVES  = ['mi', 'ao', 'ti', 'po']
VALVES_PAR = ['CQ', 'RRA']

CHAMBERS = ['la', 'lv', 'ra', 'rv']
CHAMBERS_PAR = ['E_pas', 'E_act', 'v_ref', 'af',  'v', 'p', 'tr', 'td', 'delay', 'tpww', 'tpwb']

# TIMINGS = []

class KorakianitisModel_parameters(ParametersObject):
    """
    Intro
    -----
   Model Parameters based on Korakianitis and Shi (2006)
    """
    def __init__(self, name='Korakianitis 2006') -> None:
        super().__init__(name=name)
        self.components = {key : None for key in KORAKIANITIS_2006_COMPONENTS}
        for type_, type_var in [[VESSELS, VESSELS_PAR], [VALVES, VALVES_PAR], [CHAMBERS, CHAMBERS_PAR]]:
            for key in type_:
                self[key] = pd.Series(index=type_var, dtype=object)
                
        # self.timings = {key : pd.Series(index=TIMINGS) for key in CHAMBERS} 
                
        self._vessels = VESSELS
        self._valves  = VALVES
        self._chambers= CHAMBERS
                
        self.set_chamber_comp('lv', E_pas= 0.1, E_act= 2.5, v_ref=5.0, tr = 0.30, td = 0.450, v=500.)
        self.set_chamber_comp('la', E_pas= 0.15, E_act= 0.25, v_ref=4.0, tpwb = 0.0, tpww = 0.09, delay=0.08, v=0.0)
        self.set_chamber_comp('rv', E_pas= 0.1, E_act= 1.15, v_ref=10., tr=0.30, td=0.45, v=400.)
        self.set_chamber_comp('ra', E_pas= 0.15, E_act= 0.25, v_ref=4., tpwb=0.0, tpww=0.09, delay=0.08, v=0.0)

        self.set_activation_function('lv', af=activation_function_2)
        self.set_activation_function('rv', af=activation_function_2)
        
        self.set_activation_function('la', af=activation_function_3)
        self.set_activation_function('ra', af=activation_function_3)


        # systemic circulation
        self.set_rlc_comp('sas', r=0.003,               c=0.08, l=0.000062, v=0.0, v_ref=0.0)
        self.set_rlc_comp('sat', r=(0.05 + 0.5 + 0.52), c=1.6 , l=0.0017  , v=0.0, v_ref=0.0)
        self.set_rlc_comp('svn', r=0.075,               c=20.5,             v=0.0, v_ref=0.0)
        
        # pulmonary circulation
        self.set_rlc_comp('pas', r=0.002           , c=0.18, l=0.000052, v=0.0, v_ref=0.0)
        self.set_rlc_comp('pat', r=(0.01+0.05+0.25), c=3.8 , l=0.0017  , v=0.0, v_ref=0.0)
        self.set_rlc_comp('pvn', r=0.006           , c=20.5            , v=0.0, v_ref=0.0)
        
        # valves
        self.set_valve_comp('ao', CQ=350., RRA=0.0)
        self.set_valve_comp('mi', CQ=400., RRA=0.0)
        self.set_valve_comp('po', CQ=350., RRA=0.0)
        self.set_valve_comp('ti', CQ=400., RRA=0.0)
        
    def set_chamber_comp(self, key, **kwargs):
        self._set_comp(key=key, set=CHAMBERS, **kwargs)
        
    def set_activation_function(self, key, af):
        self._set_comp(key, set=CHAMBERS, af=af)
        
    def set_rlc_comp(self, key, **kwargs):
        self._set_comp(key=key, set=VESSELS, **kwargs)
        
    def set_valve_comp(self, key, **kwargs):
        self._set_comp(key=key, set=VALVES, **kwargs)
        
    def set(self, key, **kwargs):
        if key in CHAMBERS:
            self._set_comp(key=key, set=CHAMBERS, **kwargs)
        if key in VESSELS:
            self._set_comp(key=key, set=VESSELS, **kwargs)
        if key in VALVES:
            self._set_comp(key=key, set=VALVES, **kwargs)
