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
            'inputs'    : None, # series
            'dudt_name' : None,
            'u_name'    : None,
        })
        
    def __repr__(self) -> str:
        return f" > variable name: {self._name}"
        
    def set_dudt_func(self, function, function_name:str)->None:
        self._ode_sys_mapping['dudt_func'] = function
        self._ode_sys_mapping['dudt_name'] = function_name
        
    def set_u_func(self, function, function_name:str)->None:
        self._ode_sys_mapping['u_func'] = function
        self._ode_sys_mapping['u_name'] = function_name
        
    def set_inputs(self, inputs:Series[str]):
        self._ode_sys_mapping['inputs'] = inputs
        
    def set_name(self, name)->None:
        self._name = name
        
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
    
class StateVariableDictionary:
    def __init__(self, dict_:dict[str,StateVariable]=None) -> None:
        if dict_ is None:
            self._data = pd.Series()
        else:
            for sv in dict_.values():
                if not isinstance(sv, StateVariable): raise Exception(' This dictionary can only contain StateVariable instance values.')
            self._data = pd.Series(dict_)
        return
    
    def __getitem__(self, key:str)->StateVariable:
        return self._data[key]
    
    def __setitem__(self, key:str, value:StateVariable) -> None:
        if not isinstance(value, StateVariable): raise Exception(' This dictionary can only contain StateVariable instance values.')
        self._data[key] = value
        
    def __len__(self) -> None:
        return len(self._data)
    
    def items(self) -> list[tuple[str,StateVariable]]:
        return self._data.items()
    
    def keys(self) -> list[str]:
        return self._data.keys()
    
    @property
    def values(self) -> list[StateVariable]:
        return self._data.values
    
    @property
    def index(self) -> list:
        return self._data.index
    
    def get_sv_values(self, tind:int) -> dict[str,float]:
        return self._data.apply(lambda sv : sv.u[tind])
    
    def get_sv_dudt_func(self) -> pd.Series:
        return self._data.apply(lambda sv : sv.dudt_func)
    
    def get_sv_u_func(self) -> pd.Series:
        return self._data.apply(lambda sv : sv.u_func)
    
    def get_sv_inputs(self) -> Series[StateVariable]:
        return self._data.apply(lambda sv : sv.inputs)  
    
    # def get_sv_local_global_input_mapping(self, input_:Series[float]) -> Series[tuple[str, str]]:
    #     return self._data.apply(lambda sv : zip(sv.inputs.keys(), input_.loc[sv.inputs.values()]))
    
    def apply(self, *args, **kwargs):
        return self._data.apply(*args, **kwargs)
    
            
         