from ..HelperRoutines import relu_max, activation_function_2, bold_text
from .ParametersObject import ParametersObject
import pandas as pd


KORAKIANITIS_2006_COMPONENTS = ['la', # left atrium
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
VALVES_PAR = ['CQ']

CHAMBERS = ['la', 'lv', 'ra', 'rv']
CHAMBERS_PAR = ['E_pas', 'E_act', 'V_ref', 'activation_function', 
                't_max', 't_tr', 'tau','delay', 'V', 'P']

class Korakianitis_2006(ParametersObject):
    """
    Intro
    -----
   Model Parameters based on Korakianitis and Shi (2006)
    """
    def __init__(self, name='Korakianitis 2006') -> None:
        super().__init__(name=name)
        self.components = {key : None for key in KORAKIANITIS_2006_COMPONENTS}
        for type_, type_var in [[VESSELS, VALVES_PAR], [VALVES, VALVES_PAR], [CHAMBERS, CHAMBERS_PAR]]:
            for key in type_:
                self[key] = pd.Series(index=type_var, dtype=object)
        
        mmhg = 133.32
        
        for chamber in CHAMBERS:
            self.set_activation_function(chamber, activation_function=activation_function_2)
        self.set_chamber_comp('lv',
                              E_pas=mmhg * 0.1,
                              E_act=mmhg * 2.5,
                              V_ref=5.0,
                              t_max=30.,
                              tau=45.)
        self.set_chamber_comp('la',
                              E_pas=mmhg * 0.15,
                              E_act=mmhg * 0.25,
                              V_ref=4.0,
                              t_max=9.0,
                              tau=18.0,
                              delay=17.0)
        self.set_chamber_comp('rv',
                              E_pas=mmhg * 0.1,
                              E_act=mmhg * 1.15,
                              V_ref=10.,
                              t_max=30.,
                              tau=45.)
        self.set_chamber_comp('ra',
                              E_pas=mmhg * 0.15,
                              E_act=mmhg * 0.15,
                              V_ref=4.,
                              t_max=9.0,
                              tau=18.0,
                              delay=17.0)
        
        # systemic circulation
        self.set_rlc_comp('sas', r=0.003*mmhg, c=1.6/mmhg, l=0.0017*mmhg)
        self.set_rlc_comp('sat', r=(0.05 + 0.5 + 0.52)*mmhg, c=1.6/mmhg, l=0.0017*mmhg)
        self.set_rlc_comp('svn', r=0.075*mmhg, c=20.5/mmhg)
        
        # pulmonary circulation
        self.set_rlc_comp('pas', r=0.002*mmhg, c=0.18/mmhg, l=0.000052/mmhg)
        self.set_rlc_comp('pat', r=(0.01+0.05+0.25)*mmhg, c=3.8/mmhg, l=0.0017)
        self.set_rlc_comp('pvn', r=0.006*mmhg, c=20.5/mmhg)
        
        # valves
        self.set_valve_comp('ao', CQ=350. / mmhg**0.5)
        self.set_valve_comp('mi', CQ=400. / mmhg**0.5)
        self.set_valve_comp('po', CQ=350. / mmhg**0.5)
        self.set_valve_comp('ti', CQ=400. / mmhg**0.5)
        
    def set_chamber_comp(self, key, **kwargs):
        self._set_comp(key=key, set=CHAMBERS, **kwargs)
        
    def set_activation_function(self, key, activation_function):
        self._set_comp(key, set=CHAMBERS, activation_function=activation_function)
        
    def set_rlc_comp(self, key, **kwargs):
        self._set_comp(key=key, set=VESSELS, **kwargs)
        
    def set_valve_comp(self, key, **kwargs):
        self._set_comp(key=key, set=VALVES, **kwargs)
        
