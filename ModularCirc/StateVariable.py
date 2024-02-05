from .Time import TimeClass
import numpy as np

class StateVariable():
    def __init__(self, name:str, timeobj:TimeClass) -> None:
        self._name = name
        self._to   = timeobj
        self._u    = np.zeros((timeobj.n_t,))
        self._cv   = 0.0
        
        self._ode_sys_mapping = {
            'dudt_func' : None,
            'u_func'    : None,
            'inputs'    : {}
        }
        
    def __repr__(self) -> str:
        return f" > variable name: {self._name}"
        
    def set_dudt_func(self, function)->None:
        self._ode_sys_mapping['dudt_func'] = function
        
    def set_u_func(self, function)->None:
        self._ode_sys_mapping['u_func'] = function
        
    def set_inputs(self, inputs:list[str]):
        self._ode_sys_mapping['inputs'] = inputs
        
    def set_name(self, name)->None:
        self._name = name
        
    @property
    def name(self):
        return self._name
    
    @property
    def dudt(self):
        return self._ode_sys_mapping['dudt_func']
    
    @property
    def inputs(self):
        return self._ode_sys_mapping['inputs']