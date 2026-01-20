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
VESSELS = ['sas', 'sat', 'svn', 'pas', 'pvn']
VESSELS_PAR = ['r', 'c', 'l', 'v_ref', 'v', 'p']

NONLINEAR_VESSELS = ['pat',]
NONLINEAR_VESSELS_PAR = ['r', 'c_ref', 'p1', 'p2', 'v_ref', 'v', 'p']

VALVES  = ['mi', 'ao', 'ti', 'po']
VALVES_PAR = ['CQ', 'RRA']

CHAMBERS = ['la', 'lv', 'ra', 'rv']
CHAMBERS_PAR = ['E_pas', 'E_act', 'v_ref', 'k_pas', 'af',  'v', 'p', 'tr', 'td', 'delay', 'tpww', 'tpwb']

TIME_SETUP_DICT = {
        'name'       :  'TimeTest', # the name asssociated with the temporal discretization (not super important.. names internat variable)
        'ncycles'    :  30,         # the maximum number of cycles for which we run the simulation
        'tcycle'     :  1.0,      # the duration of a heart beat (here in ms)
        'dt'         :  0.001,        # the duration of a discrete time step
        'export_min' :  2           # number of time steps for which we export the simulation (can also be used as a way to impose a minimum number of pulses)
    }


class KMM_PP_NRC_parameters(ParametersObject):
    """
    Intro
    -----
    Model Parameters based on Korakianitis and Shi (2006) with Maynard (2012) valves
    """
    def __init__(self, name='Korakianitis 2006') -> None:
        super().__init__(name=name)
        self.components = {key : None for key in KORAKIANITIS_2006_COMPONENTS}
        for type_, type_var in [
                                    [VESSELS, VESSELS_PAR], 
                                    [NONLINEAR_VESSELS, NONLINEAR_VESSELS_PAR],
                                    [VALVES, VALVES_PAR], 
                                    [CHAMBERS, CHAMBERS_PAR]
                                ]:
            for key in type_:
                self[key] = pd.Series(index=type_var, dtype=object)

        self._vessels = VESSELS
        self._nonlinear_vessels = NONLINEAR_VESSELS
        self._valves  = VALVES
        self._chambers= CHAMBERS

        self.set_chamber_comp('lv', E_pas= 1.7,  E_act= 3.0,  v_ref=15.0, k_pas=0.015, tr = 0.30,  td = 0.450,              v=50.)
        self.set_chamber_comp('la', E_pas= 0.5, E_act= 0.25, v_ref=4.0, k_pas=0.07,  tpwb = 0.0, tpww = 0.09, delay=0.08, v=0.0)
        self.set_chamber_comp('rv', E_pas= 0.67,  E_act= 1.15, v_ref=15., k_pas=0.015, tr=0.30,    td=0.45,                 v=100.)
        self.set_chamber_comp('ra', E_pas= 0.5, E_act= 0.25, v_ref=4.0, k_pas=0.07,  tpwb = 0.0, tpww = 0.09, delay=0.08, v=0.0)

        self.set_activation_function('lv', af=activation_function_2)
        self.set_activation_function('rv', af=activation_function_2)

        self.set_activation_function('la', af=activation_function_3)
        self.set_activation_function('ra', af=activation_function_3)


        # systemic circulation
        self.set_rlc_comp('sas', r=0.003,               c=0.08, l=0.000062, v=450.0, v_ref=0.0)
        self.set_rlc_comp('sat', r=(0.05 + 0.5 + 0.52), c=1.6 , l=0.0017  , v=0.0, v_ref=0.0)
        self.set_rlc_comp('svn', r=0.075,               c=20.5,             v=0.0, v_ref=0.0)

        # pulmonary circulation
        self.set_rlc_comp('pas', r=0.002           , c=0.18, l=0.000052, v=200.0, v_ref=0.0)
        self.set_rlc_comp('pvn', r=0.006           , c=20.5            , v=9.1, v_ref=0.0)

        # nonlinear vessel
        self.set_nonlinear_rc_comp('pat', r=(0.01+0.05+0.25), c_ref=1., p1=29.5, p2=22.9, v_ref=0.0)

        # valves
        dyn = 1333.22
        self.set_valve_comp('ao', CQ=350., RRA=0.0)
        self.set_valve_comp('mi', CQ=400., RRA=0.0)
        self.set_valve_comp('po', CQ=350., RRA=0.0)
        self.set_valve_comp('ti', CQ=400., RRA=0.0)

    def set_chamber_comp(self, key, **kwargs):
        self._set_comp(key=key, set=self._chambers, **kwargs)

    def set_activation_function(self, key, af):
        self._set_comp(key, set=self._chambers, af=af)

    def set_rlc_comp(self, key, **kwargs):
        self._set_comp(key=key, set=self._vessels, **kwargs)
    
    def set_nonlinear_rc_comp(self, key, **kwargs):
        self._set_comp(key=key, set=self._nonlinear_vessels, **kwargs)

    def set_valve_comp(self, key, **kwargs):
        self._set_comp(key=key, set=self._valves, **kwargs)

    def set(self, key, **kwargs):
        if key in self._chambers:
            self._set_comp(key=key, set=self._chambers, **kwargs)
        if key in self._nonlinear_vessels:
            self._set_comp(key=key, set=self._nonlinear_vessels, **kwargs)
        if key in self._vessels:
            self._set_comp(key=key, set=self._vessels, **kwargs)
        if key in self._valves:
            self._set_comp(key=key, set=self._valves, **kwargs) 