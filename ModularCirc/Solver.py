from .Time import TimeClass
from .StateVariable import StateVariable, StateVariableDictionary
from .HelperRoutines import bold_text
from pandera.typing import DataFrame
from .circmodels import *

import pandas as pd


class Solver():
    def __init__(self, 
                 type_:str=None, 
                #  time_object:TimeClass=None, 
                #  state_variable_dictionary:StateVariableDictionary=None,
                #  all_sv_data:DataFrame=None
                model:OdeModel=None
                 ) -> None:
        self.type = None
        self.model = model
        #####
        self._pvk = StateVariableDictionary()
        self._svk = StateVariableDictionary()
        #####
        self._asd = model.all_sv_data
        #####
        self._pvk_len = 0
        self._svk_len = 0
        self._vd  = model._state_variable_dict
        self._to  = model.time_object
        self._type = type_
        
        self.setup()
        
    def setup(self)->None:
        """
        Method for detecting which are the principal variables and which are the secondary ones.
        """
        for key, component in self._vd.items():
            if component.dudt_func is not None:
                print(f" -- Variable {bold_text(key)} added to the principal variable key list.")
                print(f'    - name of update function: {bold_text(component._ode_sys_mapping["dudt_name"])}')
                self._pvk[key] = component
            elif component.u_func is not None:
                print(f" -- Variable {bold_text(key)} added to the secondary variable key list.")
                print(f'    - name of update function: {bold_text(component._ode_sys_mapping["u_name"])}')
                self._svk[key] = component
            else:
                continue
        self._pvk_len = len(self._pvk)
        self._svk_len = len(self._svk)
        return
            
    @property
    def pvk(self) -> StateVariableDictionary:
        return self._pvk
    
    @property
    def svk(self) -> StateVariableDictionary:
        return self._svk
    
    @property
    def vd(self) -> StateVariableDictionary:
        return self._vd
    
    def get_principal_values(self, tind:int) -> dict[str,float]:
        return {name : sv.u[tind] for name, sv in self._pvk.items()}
    
    def get_secondary_values(self, tind:int) -> dict[str,float]:
        return {name : sv.u[tind] for name, sv in self._svk.items()}
        
    def update_generator(self):
        
        def s_u_update(ti:int) -> dict[str, float]:
            # return self.svk.apply(lambda sv : sv.u_func(t=self._to._sym_t_norm[ti],
            #                                              ))) )
            
            input_value_dict = self.vd.get_sv_values(ti)
            return {sv.name : 
                    sv.u_func(**dict(zip(sv.inputs.keys(), input_value_dict[sv.inputs.values()]))) 
                    for sv in self.svk.values()}
                
        
        def pv_dfdt_generated(ti:int, y:list[float]) -> pd.Series:
            """
            Function for computing dfdt for the principal state variables of the simulations.

            Args:
            -----
                ti (int): time index of dfdt
                y (list[float]): current prinicipal state variable values

            Returns:
            --------
                dict[str, float]: (state variable name), (dfdt value)
            """
            input_value_dict = {key : 0.0 for key in self.vd.keys()}
            s_u = s_u_update(ti=ti)
            
            for key, values in s_u.items():
                input_value_dict[key] = values
                
            for i, key in enumerate(self.pvk) :
                input_value_dict[key] = y[i]
                
            return pd.Series({sv.name:
                    sv.dudt_func(
                        t=self._to._sym_t_norm[ti],
                        **dict(zip(sv.inputs.keys(), input_value_dict[sv.inputs.values()]))
                        )
                    for sv in self.pvk.values()})
            
        return pv_dfdt_generated
            
            
        

            
        
        
        
        
    
    
    
    
    