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

        def initialize_by_function_rountine_2(y:np.ndarray[float]) -> np.ndarray[float]:
            funcs = self._global_sv_init_fun.values()
            ids   = self._global_sv_init_ind.values()
            return [fun(*y[inds]) for fun, inds in zip(funcs, ids)]
            
        def s_u_update_2(t, y:np.ndarray[float]) -> np.ndarray[float]:
            funcs = self._global_ssv_update_fun.values()
            ids   = self._global_ssv_update_ind.values()
            return [fun(t, y[inds]) for fun, inds in zip(funcs, ids)]
            
        def pv_dfdt_function_2(t, y:np.ndarray[float]) -> np.ndarray[float]:
            ht = t%self._to.tcycle
            y_temp = np.zeros((len(self._global_sv_id),))
            y_temp[np.array(list(self._global_psv_update_fun.keys()))] = y
            y_temp[np.array(list(self._global_ssv_update_fun.keys()))] = s_u_update_2(t, y_temp)
            
            funcs = self._global_psv_update_fun.values()
            ids   = self._global_psv_update_ind.values()
            return [func(ht,*y_temp[inds]) for func, inds in zip(funcs, ids)]
           
        self.initialize_by_function_rountine = initialize_by_function_rountine_2    
        self.pv_dfdt_global = pv_dfdt_function_2
        self.s_u_update     = s_u_update_2
    
    def solve(self):
        # initialize the solution fields
        self._asd.loc[0, self._initialize_by_function.index] = \
            self.initialize_by_function_rountine(y=self._asd.loc[0].to_numpy())
                    
        t = self._to._sym_t.values 
        
        # Solve the main system of ODEs..   
        res = solve_ivp(fun=self.pv_dfdt_global, 
                        t_span=(t[0], t[-1]), 
                        y0=self._asd.iloc[0, list(self._global_psv_update_fun.keys())].to_list(), 
                        t_eval=t,
                        max_step=self._to.dt,
                        method='BDF'
                        )

        for ind, id in enumerate(self._global_psv_update_fun.keys()):
            self._asd.iloc[:,id] = res.y[ind,:]
        
        # update the secondary variables...   
        def temp_func(y:Series)->Series:
            return pd.Series(self.s_u_update(t=0, y=y.values))
        temp = self._asd.apply(temp_func, axis=1)
        for ind, id in enumerate(self._global_ssv_update_fun.keys()):
            self._asd.iloc[:,id] = temp.loc[:,ind]
            
           
        

            
        
        
        
        
    
    
    
    
    