from ..Time import TEMPLATE_TIME_SETUP_DICT, TimeClass
from ..Components.ComponentBase import ComponentBase
from ..StateVariable import StateVariable

import pandas as pd

class OdeModel():
    def __init__(self, time_setup_dict) -> None:
        self.time_object = TimeClass(time_setup_dict=time_setup_dict)
        self._state_variable_dict = pd.Series()
        self.all_sv_data = pd.DataFrame(index=self.time_object.time.index, dtype='float64')
        self.commponents = dict()
        self.name = 'Template'
        
    
    def connect_modules(self, 
                        module1:ComponentBase, 
                        module2:ComponentBase,
                        pvariable:StateVariable = None,
                        qvariable:StateVariable = None,
                        plabel:str=None, 
                        qlabel:str=None
                        ) ->None:   
        """
        Method for connecting Component modules.
        
        Inputs
        ------
        module1 (Component) : upstream Component
        module2 (Component) : downstream Component
        plabel (str) : new name for the shared pressure state variable
        qlabel (str) : new name for the shared flow state variable
        """
        if qvariable is None:
            if module1._Q_o._ode_sys_mapping['u_func'] is not None or module1._Q_o._ode_sys_mapping['dudt_func'] is not None:
                module2._Q_i = module1._Q_o
            elif module2._Q_i._ode_sys_mapping['u_func'] is not None or module2._Q_i._ode_sys_mapping['dudt_func'] is not None:
                module1._Q_o = module2._Q_i
            else:
                raise Exception(f'Definition of flow between modules {module1._name} and {module2._name} is ambiguous.')
        else:
            module2._Q_i = qvariable
            module1._Q_o = qvariable
            
        if pvariable is None:
            if module1._P_o._ode_sys_mapping['u_func'] is not None or module1._P_o._ode_sys_mapping['dudt_func'] is not None:
                module2._P_i = module1._P_o
            elif module2._P_i._ode_sys_mapping['u_func'] is not None or module2._P_i._ode_sys_mapping['dudt_func'] is not None:
                module1._P_o = module2._P_i
            else:
                raise Exception(f'Definition of pressure between modules {module1._name} and {module2._name} is ambiguous')
        else:
            module2._P_i = pvariable
            module1._P_o = pvariable
             
        if plabel is not None and plabel not in self._state_variable_dict.keys():
            module1._P_o.set_name(plabel)
            self._state_variable_dict[plabel] = module1._P_o
            self.all_sv_data[plabel] = module1.P_o
        if qlabel is not None and qlabel not in self._state_variable_dict.keys():
            module1._Q_o.set_name(qlabel)
            self._state_variable_dict[qlabel] = module1._Q_o
            self.all_sv_data[qlabel] = module1.Q_o
        return
        
            
    @property
    def state_variable_dict(self):
        return self._state_variable_dict
    
    def __str__(self) -> str:
        out = f'Model {self.name} \n\n'
        for component in self.commponents.values():
            out += (str(component) + '\n')
        out += '\n'
        out += 'Main State Variable Dictionary \n'
        for key in self._state_variable_dict.keys():
            out += f" - {key} \n"
        return out
    
    def set_v_sv(self, comp_key:str):
        v_key = 'v_' + comp_key
        self._state_variable_dict[v_key] = self.commponents[comp_key]._V
        self._state_variable_dict[v_key].set_name(v_key)
        self.all_sv_data[v_key] = self.commponents[comp_key].V     ##### test
        # self.commponents[comp_key]._V._u = self.all_sv_data[v_key]