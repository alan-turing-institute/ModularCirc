from .Time import TimeClass
from .StateVariable import StateVariable
from .Models.OdeModel import OdeModel
from .HelperRoutines import bold_text
from pandera.typing import DataFrame, Series
from .Models.OdeModel import OdeModel

import pandas as pd
import numpy as np

from scipy.integrate import solve_ivp

class Solver():
    def __init__(self, 
                model:OdeModel=None,
                 ) -> None:
        
        self.model = model
        
        self._asd = model.all_sv_data
        self._vd  = model._state_variable_dict
        self._to  = model.time_object
        
        self._global_psv_update_fun  = {}
        self._global_ssv_update_fun  = {}
        self._global_psv_update_fun_n  = {}
        self._global_ssv_update_fun_n  = {}
        self._global_psv_update_ind  = {}
        self._global_ssv_update_ind  = {}
        self._global_psv_names      = []
        self._global_sv_init_fun    = {}
        self._global_sv_init_ind    = {}
        self._global_sv_id          = {key: id   for id, key in enumerate(model.all_sv_data.columns.to_list())}
        self._global_sv_id_rev      = {id: key   for id, key in enumerate(model.all_sv_data.columns.to_list())}
                 
        self._initialize_by_function = pd.Series()
        
    def setup(self)->None:
        """
        Method for detecting which are the principal variables and which are the secondary ones.
        """
        for key, component in self._vd.items():
            mkey = self._global_sv_id[key]
            if component.i_func is not None:
                self._initialize_by_function[key] = component
                self._global_sv_init_fun[mkey] = component.i_func
                self._global_sv_init_ind[mkey] = [self._global_sv_id[key2] for key2 in component.i_inputs.to_list()]
                
            if component.dudt_func is not None:
                print(f" -- Variable {bold_text(key)} added to the principal variable key list.")
                print(f'    - name of update function: {bold_text(component.dudt_name)}')
                self._global_psv_update_fun[mkey]   = component.dudt_func
                self._global_psv_update_fun_n[mkey] = component.dudt_name
                self._global_psv_update_ind[mkey]   = [self._global_sv_id[key2] for key2 in component.inputs.to_list()]
                self._global_psv_names.append(key)
                
            elif component.u_func is not None:
                print(f" -- Variable {bold_text(key)} added to the secondary variable key list.")
                print(f'    - name of update function: {bold_text(component.u_name)}')
                self._global_ssv_update_fun[mkey]   = component.u_func
                self._global_ssv_update_fun_n[mkey] = component.u_name
                self._global_ssv_update_ind[mkey]   = [self._global_sv_id[key2] for key2 in component.inputs.to_list()]
            else:
                continue
        print(' ')
        self.generate_dfdt_functions()
        return
    
    @property
    def vd(self) -> Series[StateVariable]:
        return self._vd
    
    @property
    def dt(self) -> float:
        return self._to.dt
        
    def generate_dfdt_functions(self):
        
        funcs1 = self._global_sv_init_fun.values()
        ids1   = self._global_sv_init_ind.values()

        def initialize_by_function(y:np.ndarray[float]) -> np.ndarray[float]:
            return np.array([fun(t=0.0, y=y[inds]) for fun, inds in zip(funcs1, ids1)])
        
        funcs2 = self._global_ssv_update_fun.values()
        ids2   = self._global_ssv_update_ind.values()
            
        def s_u_update(t, y:np.ndarray[float,float]) -> np.ndarray[float]:
            return np.array([fun(t=t, y=y[inds]) for fun, inds in zip(funcs2, ids2)])
        
        keys3  = np.array(list(self._global_psv_update_fun.keys()))
        keys4  = np.array(list(self._global_ssv_update_fun.keys()))
        funcs3 = self._global_psv_update_fun.values()
        ids3   = self._global_psv_update_ind.values()
            
        def pv_dfdt_update(t, y:np.ndarray[float]) -> np.ndarray[float]:
            ht = t%self._to.tcycle
            y_temp = np.zeros( (len(self._global_sv_id),y.shape[1]) if len(y.shape) == 2 else (len(self._global_sv_id),)) 
            y_temp[keys3] = y
            y_temp[keys4] = s_u_update(t, y_temp)
            return np.array([func(t=ht, y=y_temp[inds]) for func, inds in zip(funcs3, ids3)])
           
        self.initialize_by_function = initialize_by_function    
        self.pv_dfdt_global = pv_dfdt_update
        self.s_u_update     = s_u_update
    
    def solve(self):
        # initialize the solution fields
        self._asd.loc[0, self._initialize_by_function.index] = \
            self.initialize_by_function(y=self._asd.loc[0].to_numpy()).T
        t = self._to._sym_t.values 
        # Solve the main system of ODEs..   
        res = solve_ivp(fun=self.pv_dfdt_global, 
                        t_span=(t[0], t[-1]), 
                        y0=self._asd.iloc[0, list(self._global_psv_update_fun.keys())].to_list(), 
                        t_eval=t,
                        max_step=self._to.dt,
                        method='BDF',
                        vectorized=True,
                        )
        for ind, id in enumerate(self._global_psv_update_fun.keys()):
            self._asd.iloc[:,id] = res.y[ind,:]
        temp = self.s_u_update(t=0.0, y=self._asd.values.T)
        self._asd.iloc[:,self._global_ssv_update_fun.keys()] = temp.T
            
           
        

            
        
        
        
        
    
    
    
    
    