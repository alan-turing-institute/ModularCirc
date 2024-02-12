from .Time import TimeClass
from .StateVariable import StateVariable
from .Models.OdeModel import OdeModel
from .HelperRoutines import bold_text
from pandera.typing import DataFrame, Series
from .Models.OdeModel import OdeModel

import pandas as pd
import numpy as np

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
        self._type = type_
        self._eps = 0.001
        
        self._initialize_by_function = pd.Series()
        self.setup()
        
    def setup(self)->None:
        """
        Method for detecting which are the principal variables and which are the secondary ones.
        """
        for key, component in self._vd.items():
            if component.i_func is not None:
                self._initialize_by_function[key] = component
            if component.dudt_func is not None:
                print(f" -- Variable {bold_text(key)} added to the principal variable key list.")
                print(f'    - name of update function: {bold_text(component._ode_sys_mapping["dudt_name"])}')
                self._psv[key] = component
            elif component.u_func is not None:
                print(f" -- Variable {bold_text(key)} added to the secondary variable key list.")
                print(f'    - name of update function: {bold_text(component._ode_sys_mapping["u_name"])}')
                self._ssv[key] = component
            else:
                continue        
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
        
        def initialize_by_function_rountine(y:Series[float]) -> Series[float]:
            return self._initialize_by_function.apply(
                lambda sv : sv.i_func(**sv.i_inputs.apply(lambda key : y[key])))
        
        def s_u_update(t:float, y:Series[float]) -> Series[float]:
            return self._ssv.apply(
                 lambda sv : sv.u_func(t=t, **sv.inputs.apply(lambda key : y[key])))
        
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
            y_secondary = s_u_update(t=t, y=y)
            y_new = pd.concat([y, y_secondary])
            return self._psv.apply(
                lambda sv : sv.dudt_func(t=t, **sv.inputs.apply(lambda key : y_new[key])))
            
        def pv_dfdt_Jacobian_function(t:float, y:Series[float]) ->DataFrame[float]:
            # create a perturbation matrix
            pm = self._eps * pd.DataFrame(index =y.index, columns=y.index, data=np.eye(N=y.index.size))
            # apply perturbation on y to for each entry to generate the Jacobian matrix
            return pm.apply(lambda col: (pv_dfdt_function(t, y+col) - pv_dfdt_function(t, y-col)) / 2.0 / self._eps) 
           
        self.initialize_by_function_rountine = initialize_by_function_rountine    
        self.pv_dfdt_global = pv_dfdt_function
        self.s_u_update     = s_u_update
        if self.use_back_component : self.pv_J_djdt_global = pv_dfdt_Jacobian_function
        
        
    def define_advancing_method(self):
        pass    
    
    def solve(self):
        self._asd.loc[0, self._initialize_by_function.index] = \
            self.initialize_by_function_rountine(y=self._asd.loc[0])
        for ind, trow in self._to.time.iterrows():
            if ind == 0 : continue
            ht = trow['cycle_t']
            yp = self._asd.loc[ind-1, self._psv.index]
            dydp = self.pv_dfdt_global(ht, y=yp)
            self._asd.loc[ind, self._psv.index] = yp + self.dt * dydp 
            self._asd.loc[ind, self._ssv.index] = self.s_u_update(ind, self._asd.loc[ind])        
        

            
        
        
        
        
    
    
    
    
    