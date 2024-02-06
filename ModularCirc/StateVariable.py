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
            'inputs'    : {},
            'dudt_name' : None,
            'u_name'    : None,
        }
        
    def __repr__(self) -> str:
        return f" > variable name: {self._name}"
        
    def set_dudt_func(self, function, function_name:str)->None:
        self._ode_sys_mapping['dudt_func'] = function
        self._ode_sys_mapping['dudt_name'] = function_name
        
    def set_u_func(self, function, function_name:str)->None:
        self._ode_sys_mapping['u_func'] = function
        self._ode_sys_mapping['u_name'] = function_name
        
    def set_inputs(self, inputs:dict[str,str]):
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
    def dudt_name(self):
        return self._ode_sys_mapping['dudt_name']
    
    @property
    def inputs(self):
        return self._ode_sys_mapping['inputs']
    
    @property
    def u_func(self):
        return self._ode_sys_mapping['u_func']
    
    @property
    def u_name(self):
        return self._ode_sys_mapping['u_name']
    
    @property
    def u(self):
        return self._u
    
class StateVariableDictionary:
    def __init__(self, dict_:dict[str,StateVariable]=None) -> None:
        if dict_ is None:
            self._data = dict()
        else:
            for sv in dict_.values():
                if not isinstance(sv, StateVariable): raise Exception(' This dictionary can only contain StateVariable instance values.')
            self._data = dict_
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
    
    def values(self) -> list[StateVariable]:
        return self._data.values()
    
    def get_sv_values(self, tind:int):
        return {name : sv.u[tind] for name, sv in self.items()}
    
    
    
def get_dfdt(input_dict:StateVariableDictionary, output_dict:StateVariableDictionary):
    input_keys  = input_dict.keys()
    output_keys = output_dict.keys()
    func = [lambda t, **kwarg : sv for sv in output_dict.values()]
    
            
         