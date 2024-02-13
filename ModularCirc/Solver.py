from .Time import TimeClass
from .StateVariable import StateVariable
from .Models.OdeModel import OdeModel
from .HelperRoutines import bold_text
from pandera.typing import DataFrame, Series
from .Models.OdeModel import OdeModel

import pandas as pd
import numpy as np

import time
from scipy.integrate import odeint, solve_ivp

class Solver():
    def __init__(self, 
                type_:str=None, 
                model:OdeModel=None,
                theta:float=0.0
                 ) -> None:
        
        self._Solver_types = {
            None : 'ForwarddEuler',
            'ForwarddEuler' : 'ForwarddEuler',
            'BackwardEuler' : 'BackwardEuler',
            'ThetaScheme'   : 'ThetaScheme'
        }
        
        self.type = self._Solver_types[type_]
        self.model = model
        if self.type == 'BackwardEuler' or self.type == 'ThetaScheme' : 
            self.use_back_component = True
        else:
            self.use_back_component = False
        #####
        self._psv = pd.Series()
        self._ssv = pd.Series()
        #####
        self._asd = model.all_sv_data
        #####
        self._vd  = model._state_variable_dict
        self._to  = model.time_object
        
        print(self.model.all_sv_data.columns)
        print(self._vd)
        N  = len(self.model.all_sv_data.index)
        self._global_psv_update_fun  = {}
        self._global_ssv_update_fun  = {}
        self._global_psv_update_fun_n  = {}
        self._global_ssv_update_fun_n  = {}
        self._global_psv_update_ind  = {}
        self._global_ssv_update_ind  = {}
        self._global_sv_init_fun    = {}
        self._global_sv_init_ind    = {}
        self._global_sv_id          = {key: id   for id, key in enumerate(model.all_sv_data.columns.to_list())}
        self._global_sv_id_rev      = {id: key   for id, key in enumerate(model.all_sv_data.columns.to_list())}
        
        print('_global_sv_id')
        print(self._global_sv_id)
        
        self._type = type_
        self._eps = 0.001
        
        self._initialize_by_function = pd.Series()
        
    def setup(self)->None:
        """
        Method for detecting which are the principal variables and which are the secondary ones.
        """
        global_key_list = self._asd.columns.to_series().to_list()
        print(self._global_sv_id)
        for key, component in self._vd.items():
            mkey = self._global_sv_id[key]
            if component.i_func is not None:
                self._initialize_by_function[key] = component
                self._global_sv_init_fun[mkey] = component.i_func
                self._global_sv_init_ind[mkey] = [self._global_sv_id[key2] for key2 in component.i_inputs.to_list()]
                
            if component.dudt_func is not None:
                print(f" -- Variable {bold_text(key)} added to the principal variable key list.")
                print(f'    - name of update function: {bold_text(component.dudt_name)}')
                self._psv[key] = component
                self._global_psv_update_fun[mkey]   = component.dudt_func
                self._global_psv_update_fun_n[mkey] = component.dudt_name
                self._global_psv_update_ind[mkey]   = [self._global_sv_id[key2] for key2 in component.inputs.to_list()]
                
            elif component.u_func is not None:
                print(f" -- Variable {bold_text(key)} added to the secondary variable key list.")
                print(f'    - name of update function: {bold_text(component.u_name)}')
                self._ssv[key] = component
                self._global_ssv_update_fun[mkey]   = component.u_func
                self._global_ssv_update_fun_n[mkey] = component.u_name
                self._global_ssv_update_ind[mkey]   = [self._global_sv_id[key2] for key2 in component.inputs.to_list()]
                # print(mkey, key, component.inputs.to_list(), self._global_ssv_update_ind[mkey])
            else:
                continue
        # raise Exception
        print(' ')
        self.generate_dfdt_functions()
        return
            
    @property
    def psv(self) -> Series[StateVariable]:
        return self._psv
    
    @property
    def ssv(self) -> Series[StateVariable]:
        return self._ssv
    
    @property
    def vd(self) -> Series[StateVariable]:
        return self._vd
    
    @property
    def dt(self) -> float:
        return self._to.dt
        
    def generate_dfdt_functions(self):
        
        def initialize_by_function_rountine(y:Series[float], *args) -> Series[float]:
            return self._initialize_by_function.apply(
                lambda sv : sv.i_func(**sv.i_inputs.apply(lambda key : y[key])))
            
        def initialize_by_function_rountine_2(y:np.ndarray[float]) -> np.ndarray[float]:
            funcs = self._global_sv_init_fun.values()
            ids   = self._global_sv_init_ind.values()
            return [fun(*y[inds]) for fun, inds in zip(funcs, ids)]
        
        def s_u_update(t:float, y:Series[float], *args) -> Series[float]:
            return self._ssv.apply(
                 lambda sv : sv.u_func(t=t, **sv.inputs.apply(lambda key : y[key])))
            
        def s_u_update_2(t, y, *args):
            funcs = self._global_ssv_update_fun.values()
            ids   = self._global_ssv_update_ind.values()
            fname = self._global_ssv_update_fun_n
            # for key in self._global_ssv_update_fun_n.keys():
            #     print(f' - var: {self._global_sv_id_rev[key]}')
            #     print(f' - u update {fname[key]}')
            #     print(f' - inputs: {[self._global_sv_id_rev[k] for k in self._global_ssv_update_ind[key]]}')
            #     print(f' - input val: {y[self._global_ssv_update_ind[key]]}')
            #     print(f' - val: {self._global_ssv_update_fun[key](t, *y[self._global_ssv_update_ind[key]])}')
            #     print(' ')
            return [fun(t, *y[inds]) for fun, inds in zip(funcs, ids)]
        
        def pv_dfdt_function(t:float, y:Series[float]) -> Series[float]:
            """
            Function for computing dfdt for the principal state variables of the simulations.

            Args:
            -----
                ti (int): time index of dfdt
                y (Series[float]): current prinicipal state variable values

            Returns:
            --------
                Series[float]: dfdt value
            """
            # print(self._global_sv_update_ind)
            y_secondary = s_u_update(t=t, y=y)
            y_new = pd.concat([y, y_secondary])
            return self._psv.apply(
                lambda sv : sv.dudt_func(t=t, **sv.inputs.apply(lambda key : y_new[key])))
            
        def pv_dfdt_function_2(t, y):
            ht = t%self._to.tcycle
            # print(f"sim time {t:.1f} heart time {ht:.1f}")
            y_temp = np.zeros((len(self._global_sv_id),))
            y_temp[np.array(list(self._global_psv_update_fun.keys()))] = y
            y_temp[np.array(list(self._global_ssv_update_fun.keys()))] = s_u_update_2(t, y_temp)
            
            funcs = self._global_psv_update_fun.values()
            ids   = self._global_psv_update_ind.values()
            # for func, inds in zip(funcs, ids):
            #     print(func)
            return [func(ht,*y_temp[inds]) for func, inds in zip(funcs, ids)]
           
        self.initialize_by_function_rountine = initialize_by_function_rountine_2    
        self.pv_dfdt_global = pv_dfdt_function_2
        self.s_u_update     = s_u_update_2
    
    def solve(self):
        self._asd.loc[0, self._initialize_by_function.index] = \
            self.initialize_by_function_rountine(y=self._asd.loc[0])
                    
        t = self._to._sym_t.values
        # res = odeint(func=self.pv_dfdt_global, y0=self._asd.loc[0, self._psv.index].to_list(), t=t, tfirst=True)
        
        # print(f" T span {t[0]} {t[-1]}")
        res = solve_ivp(fun=self.pv_dfdt_global, 
                        t_span=(t[0], t[-1]), 
                        y0=self._asd.loc[0, self._psv.index].to_list(), 
                        t_eval=t,
                        max_step=self._to.dt,
                        method='BDF'
                        )
                                
        # print(res.y.shape)
        # print(len(self._asd))

        for ind, id in enumerate(self._global_psv_update_fun.keys()):
            self._asd.iloc[:,id] = res.y[ind,:]
            
        # print(' Set Q')    
        temp = self._asd.apply(lambda y: pd.Series(self.s_u_update(t=0, y=y.values)), axis=1)
        # print(temp)
        
        for ind, id in enumerate(self._global_ssv_update_fun.keys()):
            self._asd.iloc[:,id] = temp.loc[:,ind]
            
           
        

            
        
        
        
        
    
    
    
    
    