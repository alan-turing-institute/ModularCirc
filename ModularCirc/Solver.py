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
        
        self._N_sv = len(self._global_sv_id)
        
        self._Nconv = None
        
                
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
                print(f'    - inputs: {component.inputs.to_list()}')
                self._global_psv_update_fun[mkey]   = component.dudt_func
                self._global_psv_update_fun_n[mkey] = component.dudt_name
                self._global_psv_update_ind[mkey]   = [self._global_sv_id[key2] for key2 in component.inputs.to_list()]
                self._global_psv_update_ind[mkey]   = np.pad(self._global_psv_update_ind[mkey], (0, self._N_sv-len(self._global_psv_update_ind[mkey])), mode='constant')
                self._global_psv_names.append(key)
                
            elif component.u_func is not None:
                print(f" -- Variable {bold_text(key)} added to the secondary variable key list.")
                print(f'    - name of update function: {bold_text(component.u_name)}')
                self._global_ssv_update_fun[mkey]   = component.u_func
                self._global_ssv_update_fun_n[mkey] = component.u_name
                self._global_ssv_update_ind[mkey]   = [self._global_sv_id[key2] for key2 in component.inputs.to_list()]
                self._global_ssv_update_ind[mkey]   = np.pad(self._global_ssv_update_ind[mkey], (0, self._N_sv-len(self._global_ssv_update_ind[mkey])), mode='constant')
            else:
                continue
        self._N_psv= len(self._global_psv_update_fun)
        self._N_ssv= len(self._global_ssv_update_fun)
        print(' ')
        self.generate_dfdt_functions()
        return
    
    @property
    def vd(self) -> Series[StateVariable]:
        return self._vd
    
    @property
    def dt(self) -> float:
        return self._to.dt
    
    @property
    def Nconv(self) -> float:
        return self._Nconv
        
    def generate_dfdt_functions(self):
        
        funcs1 = self._global_sv_init_fun.values()
        ids1   = self._global_sv_init_ind.values()

        def initialize_by_function(y:np.ndarray[float]) -> np.ndarray[float]:
            return np.array([fun(t=0.0, y=y[inds]) for fun, inds in zip(funcs1, ids1)])
            
        funcs2 = np.array(list(self._global_ssv_update_fun.values()))
        ids2   = np.stack(list(self._global_ssv_update_ind.values()))
            
        def s_u_update(t, y:np.ndarray[float,float]) -> np.ndarray[float]:
            return np.array([fi(t=y, y=yi) for fi, yi in zip(funcs2, y[ids2])], dtype=np.float64)
        
        keys3  = np.array(list(self._global_psv_update_fun.keys()))
        keys4  = np.array(list(self._global_ssv_update_fun.keys()))
        funcs3 = np.array(list(self._global_psv_update_fun.values()))
        ids3   = np.stack(list(self._global_psv_update_ind.values()))
                            
        def pv_dfdt_update(t, y:np.ndarray[float]) -> np.ndarray[float]:
            ht = t%self._to.tcycle
            y_temp = np.zeros( (len(self._global_sv_id),y.shape[1]) if len(y.shape) == 2 else (len(self._global_sv_id),)) 
            y_temp[keys3] = y
            y_temp[keys4] = s_u_update(t, y_temp)
            return np.fromiter([fi(t=ht, y=yi) for fi, yi in zip(funcs3, y_temp[ids3])], dtype=np.float64)
        
        self.initialize_by_function = initialize_by_function    
        self.pv_dfdt_global = pv_dfdt_update
        self.s_u_update     = s_u_update
        
    
    def advance_cycle(self, y0, cycleID):
        n_t = self._to.n_c - 1
        t = self._to._sym_t.values[cycleID*n_t:(cycleID+1)*n_t+1] 
        res = solve_ivp(fun=self.pv_dfdt_global, 
                        t_span=(t[0], t[-1]), 
                        y0=y0, 
                        t_eval=t,
                        max_step=self._to.dt,
                        method='BDF',
                        )
        for ind, id in enumerate(self._global_psv_update_fun.keys()):
            self._asd.iloc[cycleID*n_t:(cycleID+1)*n_t+1, id] = res.y[ind, 0:n_t+1]
            
        if cycleID == 0: return False
        
        ind = list(self._global_psv_update_fun.keys())
        cycleP = cycleID - 1
        cols = [col for col in self._asd.columns if 'v_' in col or 'p_' in col]
        cs   = self._asd[cols].iloc[cycleID*n_t:(cycleID+1)*n_t, :].values
        cp   = self._asd[cols].iloc[cycleP *n_t:(cycleP +1)*n_t, :].values
        
        norm = np.where(np.ptp(cp) > 1e-10 , np.abs(cs - cp) / np.ptp(cp), np.abs(cs - cp))
        norm = np.sum(norm.std(axis=0)**2.0) ** 0.5
        
        return norm < 0.001
    
    
    def solve(self):
        # initialize the solution fields
        self._asd.loc[0, self._initialize_by_function.index] = \
            self.initialize_by_function(y=self._asd.loc[0].to_numpy()).T
        # Solve the main system of ODEs.. 
        for i in range(self._to.ncycles):
            y0 = self._asd.iloc[i * (self._to.n_c-1), list(self._global_psv_update_fun.keys())].to_list()
            flag = self.advance_cycle(y0=y0, cycleID=i)
            if flag and i > self._to.export_min:
                self._Nconv = i
                break
        self._asd = self._asd.iloc[:self.Nconv*(self._to.n_c-1)+1]
        self._to._sym_t   = self._to._sym_t.head(self.Nconv*(self._to.n_c-1)+1)
        self._to._cycle_t = self._to._cycle_t.head(self.Nconv*(self._to.n_c-1)+1)
        
        self._to.n_t = self.Nconv*(self._to.n_c-1)+1
                
        for key in self._vd.keys():
            self._vd[key]._u = self._asd[key] 
        
        temp = self.s_u_update(t=0.0, y=self._asd.values.T)
        self._asd.iloc[:,self._global_ssv_update_fun.keys()] = temp.T
            
           
        

            
        
        
        
        
    
    
    
    
    