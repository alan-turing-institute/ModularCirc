from ..HelperRoutines import *

import pandas as pd

class NaghaviModelParameters():
    def __init__(self) -> None:
        components = ['ao', 'art', 'ven', 'av', 'mv', 'la', 'lv']
        self.components = {key : None for key in components}
        for key in ['ao', 'art', 'ven']:
            self.components[key] = pd.Series(index=['r', 'c', 'l', 'v_ref', 'v'], dtype='float64')
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
                                                    'V'], dtype=object)
                        
        self.components['ao'].loc[:] =  [32000., 0.0025, 0.0, 100. , 0.025*5200.0]
        self.components['art'].loc[:] = [150000., 0.025, 0.0, 50.  , 0.21*5200.0]
        self.components['ven'].loc[:] = [1200.,   1.   , 0.0, 2800., 0.727*5200.0]
        
        self.components['av'].loc[:] = [800., relu_max]
        self.components['mv'].loc[:] = [550., relu_max]        
        
        
        # original
        self.components['la'].loc[:] = [60.,           # E_pas
                                        0.44/0.0075,   # E_act
                                        10.,           # V_ref
                                        activation_function_2,
                                        150.,          # t_max
                                        1.5*150.,      # t_tr
                                        25.,           # tau
                                        0.018*5200.0   # V
                                        ]
        self.components['lv'].loc[:] = [400,           # E_pas
                                        1./0.0075,     # E_act
                                        10.,           # V_ref 
                                        activation_function_2,
                                        280.,          # t_max 
                                        1.5*280.,      # t_tr 
                                        25.,           # tau 
                                        0.02*5200.0    # V
                                        ]
        
    def __repr__(self) -> str:
        out = 'Naghavi Model parameters set: \n'
        for comp in self.components: 
            out += f" * Component - {bold_text(str(comp))}" + '\n'
            for key, item in self.components[comp].items():
                if isinstance(item, float):
                    out += (f"  - {bold_text(str(key)):<20} : {item:.3e}") + '\n'
                else:
                    out += (f"  - {bold_text(str(key)):<20} : {item}") + '\n'
            out += '\n'
        return out
    
    def __getitem__(self, key):
        return self.components[key]
    
    def set_rc_comp(self, key:str, 
                    r:float=None, 
                    c:float=None,
                    v_ref:float=None, 
                    v:float=None)->None:
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
    
    def set_rlc_comp(self, key:str, 
                    r:float=None, 
                    c:float=None,
                    l:float=None, 
                    v_ref:float=None, 
                    v:float=None)->None:
        if key not in ['ao','art', 'ven']:
            raise Exception('Wrong key!')
        if l is not None:
            self.components[key].loc['l'] = l
        self.set_rc_comp(key=key, r=r,c=c, v_ref=v_ref, v=v)
            
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
        return
            
    def set_activation_function(self, key:str, activation_func=activation_function_2) -> None:
        if key not in ['lv', 'la']:
            raise Exception('Wrong key!')
        if activation_func is not None:
            self.components[key].loc['activation_function'] = activation_func
        