from ..Time import TimeClass
from ..StateVariable import StateVariable
import numpy as np

class ComponentBase():
    def __init__(self,
                 name,
                 time_object:TimeClass,
                 v:float = None,
                 p:float = None,
                 ) -> None:
        self._name     = name
        self._to      = time_object

        self._P_i = StateVariable(name=name+'_P_i', timeobj=time_object)
        self._Q_i = StateVariable(name=name+'_Q_i', timeobj=time_object)
        self._P_o = StateVariable(name=name+'_P_o', timeobj=time_object)
        self._Q_o = StateVariable(name=name+'_Q_o', timeobj=time_object)
        self._V   = StateVariable(name=name+'_V' , timeobj=time_object)

        if v is not None:
            self.v0 = v
            self.V.loc[0] = v
        else:
            self.v0 = None
        self.p0 = p
        return

    def __repr__(self) -> str:
        var = (f" Component {self._name}" + '\n' +
               f" - P_i " + str(self._P_i) + '\n' +
               f" - Q_i " + str(self._Q_i) + '\n' +
               f" - P_o " + str(self._P_o) + '\n' +
               f" - Q_o " + str(self._Q_o) + '\n'
               )
        return var

    @property
    def P_i(self):
        return self._P_i._u

    @property
    def P_o(self):
        return self._P_o._u

    @property
    def Q_i(self):
        return self._Q_i._u

    @property
    def Q_o(self):
        return self._Q_o._u

    @property
    def V(self):
        return self._V._u

    @property 
    def P(self):
        """Standard pressure property - can be overridden if needed."""
        return self._P_i._u

    def _is_none_or_nan(self, value):
        """Helper to check if value is None or NaN."""
        return value is None or (isinstance(value, (int, float)) and np.isnan(value))
    
    def _validate_initial_conditions(self):
        """Validate that required initial conditions are provided."""
        if hasattr(self, 'v0') and hasattr(self, 'p0'):
            has_v0 = not self._is_none_or_nan(self.v0)
            has_p0 = not self._is_none_or_nan(self.p0)
            if not has_v0 and not has_p0:
                raise ValueError(f"Component {self._name}: Solver needs at least the initial volume or pressure to be defined!")

    def make_unique_io_state_variable(self, q_flag:bool=False, p_flag:bool=True) -> None:
        if q_flag:
            self._Q_o = self._Q_i
            self._Q_o.set_name(self._name + '_Q')
        if p_flag:
            self._P_o = self._P_i
            self._P_o.set_name(self._name + '_P')
        return

    def setup(self) -> None:
        raise Exception("This is a template class only.")

    def __del__(self):
        # print('Blah')
        if hasattr(self, '_P_i'):
            del self._P_i
        if hasattr(self, '_P_o'):
            del self._P_o
        if hasattr(self, '_Q_i'):
            del self._Q_i
        if hasattr(self, '_Q_o'):
            del self._Q_o
        if hasattr(self, '_V'):
            del self._V
