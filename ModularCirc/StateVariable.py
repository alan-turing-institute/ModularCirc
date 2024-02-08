from .Time import TimeClass
from pandera.typing import Series, DataFrame
import numpy as np
import pandas as pd

class StateVariable():
    def __init__(self, name:str, timeobj:TimeClass) -> None:
        self._name = name
        self._to   = timeobj
        self._u    = pd.Series(np.zeros((timeobj.n_t,)), name=name, dtype='float64')
        self._cv   = 0.0
        
        self._ode_sys_mapping = pd.Series({
            'dudt_func' : None,
            'u_func'    : None,
            'inputs'    : None, # inputs for u_func and dudt_func
            'dudt_name' : None, # str
            'u_name'    : None, # str
            'i_func'    : None, # initialization function
            'i_inputs'  : None, # initialization function inputs
        })
        
    def __repr__(self) -> str:
        return f" > variable name: {self._name}"
        
    def set_dudt_func(self, function, function_name:str)->None:
        self._ode_sys_mapping['dudt_func'] = function
        self._ode_sys_mapping['dudt_name'] = function_name
        return
        
    def set_u_func(self, function, function_name:str)->None:
        self._ode_sys_mapping['u_func'] = function
        self._ode_sys_mapping['u_name'] = function_name
        return
        
    def set_inputs(self, inputs:Series[str]):
        self._ode_sys_mapping['inputs'] = inputs
        return
        
    def set_name(self, name)->None:
        self._name = name
        return
        
    def set_i_func(self, function, function_name:str)->None:
        self._ode_sys_mapping['i_func'] = function
        self._ode_sys_mapping['i_name'] = function_name
        
    def set_i_inputs(self, inputs:Series[str])->None:
        self._ode_sys_mapping['i_inputs'] = inputs
        
    @property
    def name(self):
        return self._name
    
    @property
    def dudt_func(self):
        return self._ode_sys_mapping['dudt_func']
    
    @property
    def dudt_name(self) -> str:
        return self._ode_sys_mapping['dudt_name']
    
    @property
    def inputs(self) -> Series[str]:
        return self._ode_sys_mapping['inputs']
    
    @property
    def u_func(self):
        return self._ode_sys_mapping['u_func']
    
    @property
    def u_name(self) -> str:
        return self._ode_sys_mapping['u_name']
    
    @property
    def u(self) -> list[float]:
        return self._u
    
    @property
    def i_func(self):
        return self._ode_sys_mapping['i_func']
    
    @property
    def i_name(self):
        return self._ode_sys_mapping['i_name']
    
    @property
    def i_inputs(self):
        return self._ode_sys_mapping['i_inputs']
    
            
         