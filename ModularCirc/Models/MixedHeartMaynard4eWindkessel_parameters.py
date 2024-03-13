from ..HelperRoutines import activation_function_2, activation_function_3, relu_max, softplus
from .ParametersObject import ParametersObject
import pandas as pd


MHM4WK_COMPONENTS = [
              'la', # left atrium
              'mi', # mitral valve
              'lv', # left ventricle
              'ao', # aortic valve
              'sai', # systemic aortic impedance
              'sa', # systemic artery
              'sc', # systemic capilary bed
              'sv', # systemic vein
              'ra',  # right atrium
              'ti',  # tricuspid vale
              'rv',  # right ventricle
              'po',  # pulmonary valve
              'pai', # pulmonary artery impedance
              'pa', # pulmonary artery 
              'pc', # pulmonary capilary
              'pv',  # pulmonary vein
              ]
VESSELS = ['sa',  'sv', 'pa', 'pv']
VESSELS_PAR = ['r', 'c', 'l', 'v_ref', 'v', 'p']

VALVES  = ['mi', 'ao', 'ti', 'po']
# VALVES_PAR = ['CQ', 'RRA', 'Ko', 'Kc', 'R', 'L']
# VALVES_PAR = ['r', 'max_func']
VALVES_PAR = ['CQ', 'RRA']


CHAMBERS = ['la', 'lv', 'ra', 'rv']
CHAMBERS_PAR = ['E_pas', 'E_act', 'v_ref', 'k_pas', 'af',  'v', 'p', 'tr', 'td', 'delay', 'tpww', 'tpwb']

IMPEDANCES = ['sai', 'pai']
IMPEDANCES_PAR = ['r']

CAPILARIES = ['sc', 'pc']
CAPILARIES_PAR = ['r']

OBJ_PAR_PAIRS = [[VESSELS,    VESSELS_PAR],
                 [VALVES,     VALVES_PAR],
                 [CHAMBERS,   CHAMBERS_PAR],
                 [IMPEDANCES, IMPEDANCES_PAR],
                 [CAPILARIES, CAPILARIES_PAR]]

class MixedHeartMaynard4eWindkessel_parameters(ParametersObject):
    """
    Intro
    -----
   Model Parameters for MixedHeartMaynard4eWindkessel models
    """
    def __init__(self, name='MixedHeartMaynard4eWindkessel_parameters') -> None:
        super().__init__(name=name)
        self.components = {key : None for key in MHM4WK_COMPONENTS}
        for type_, type_var in OBJ_PAR_PAIRS:
            for key in type_:
                self[key] = pd.Series(index=type_var, dtype=object)
                                
        self._vessels = VESSELS
        self._valves  = VALVES
        self._chambers= CHAMBERS
        self._imp     = IMPEDANCES
        self._cap     = CAPILARIES
                
        self.set_chamber_comp('lv', E_pas= 0.1,  E_act= 2.5, k_pas=0.03,  v_ref=5.0, tr = 0.30,  td = 0.450,              v=50.)
        self.set_chamber_comp('la', E_pas= 0.15, E_act= 0.25, k_pas=0.03, v_ref=4.0, tpwb = 0.0, tpww = 0.09, delay=0.08, v=0.0)
        self.set_chamber_comp('rv', E_pas= 0.1,  E_act= 1.15, k_pas=0.03, v_ref=10., tr =0.30,   td=0.45,                 v=50.)
        self.set_chamber_comp('ra', E_pas= 0.15, E_act= 0.25, k_pas=0.03, v_ref=4.,  tpwb=0.0,   tpww=0.09,   delay=0.08, v=0.0)

        self.set_activation_function('lv', af=activation_function_2)
        self.set_activation_function('rv', af=activation_function_2)
        
        self.set_activation_function('la', af=activation_function_3)
        self.set_activation_function('ra', af=activation_function_3)


        # systemic circulation
        self.set_rlc_comp('sa', r=0.05,   c=1.6 , l=0.0017  , v=450.0, v_ref=0.0)
        self.set_rlc_comp('sv', r=0.075,  c=20.5,             v=0.0,   v_ref=0.0)
        
        # set impedances
        self.set_resistance('sai', r = 0.003)
        self.set_resistance('pai', r = 0.002)
        
        # pulmonary circulation
        self.set_rlc_comp('pa', r=0.01, c=3.8 , l=0.0017   , v=250.0, v_ref=0.0)
        self.set_rlc_comp('pv', r=0.006, c=20.5            , v=0.0,   v_ref=0.0)
        
        # set capilary resistances
        self.set_resistance('sc', r = 0.5  + 0.52)
        self.set_resistance('pc', r = 0.05 + 0.25)
        
        # valves
        #####################################################
        # self.set_valve_comp('ao', r=0.01, max_func=relu_max)
        # self.set_valve_comp('mi', r=0.01, max_func=relu_max)
        # self.set_valve_comp('po', r=0.01, max_func=relu_max)
        # self.set_valve_comp('ti', r=0.01, max_func=relu_max)
        #####################################################
        # self.set_valve_comp('ao', r=0.01, max_func=softplus)
        # self.set_valve_comp('mi', r=0.01, max_func=softplus)
        # self.set_valve_comp('po', r=0.01, max_func=softplus)
        # self.set_valve_comp('ti', r=0.01, max_func=softplus)
        #####################################################
        # self.set_valve_comp('ao', CQ=350., RRA=0.0, Ko = 26., Kc = 2e3,  L=0.0, R=0.0)
        # self.set_valve_comp('mi', CQ=400., RRA=0.0, Ko = 40.,  Kc = 2e3, L=0.0, R=0.0 )
        # self.set_valve_comp('po', CQ=350., RRA=0.0, Ko = 40.,  Kc = 18e3,L=0.0, R=0.0 )
        # self.set_valve_comp('ti', CQ=400., RRA=0.0, Ko = 40.,  Kc = 2e3, L=0.0, R=0.0 )
        #####################################################
        self.set_valve_comp('ao', CQ=350., RRA=0.0)
        self.set_valve_comp('mi', CQ=400., RRA=0.0)
        self.set_valve_comp('po', CQ=350., RRA=0.0)
        self.set_valve_comp('ti', CQ=400., RRA=0.0)
        #####################################################
        
        
    def set_chamber_comp(self, key, **kwargs):
        self._set_comp(key=key, set=CHAMBERS, **kwargs)
        
    def set_activation_function(self, key, af):
        self._set_comp(key, set=CHAMBERS, af=af)
        
    def set_rlc_comp(self, key, **kwargs):
        self._set_comp(key=key, set=VESSELS, **kwargs)
        
    def set_valve_comp(self, key, **kwargs):
        self._set_comp(key=key, set=VALVES, **kwargs)
        
    def set_resistance(self, key, **kwargs):
        self._set_comp(key=key, set=IMPEDANCES + CAPILARIES, **kwargs)
