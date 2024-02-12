from ..HelperRoutines import *

import pandas as pd

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
                        
        self.components['ao'].loc[:] =  [32000., 0.0025, 100., 0.025*5200.0]
        self.components['art'].loc[:] = [150000., 0.025, 50., 0.21*5200.0]
        self.components['ven'].loc[:] = [1200., 1., 2800., 0.727*5200.0]
        
        self.components['av'].loc[:] = [800., relu_max]
        self.components['mv'].loc[:] = [550., relu_max]        
        
        
        # original
        self.components['la'].loc[:] = [60., 0.44/0.0075, 10., 150., 1.5*150., 25., 0.018*5200.0]
        self.components['lv'].loc[:] = [400, 1./0.0075, 10., 280., 1.5*280., 25., 0.02*5200.0]
        
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
        