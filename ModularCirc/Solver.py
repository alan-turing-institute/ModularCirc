from .Time import TimeClass
from .StateVariable import StateVariable, StateVariableDictionary
from .HelperRoutines import bold_text


class Solver():
    def __init__(self, 
                 type_:str=None, 
                 time_object:TimeClass=None, 
                 state_variable_dictionary:dict[str, StateVariable]=None
                 ) -> None:
        self.type = None
        self._pvk = StateVariableDictionary()
        self._svk = StateVariableDictionary()
        self._pvk_len = 0
        self._svk_len = 0
        self._vd  = state_variable_dictionary
        self._to  = time_object
        self._type = type_
        
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
        
    def dfdt_generator(self, dict_:dict[str,StateVariable]):
        func_dict = dict()
        for name, variable in dict_.items():
            func = lambda t, y : variable.dudt_func()
            
        
        
        
        
    
    
    
    
    