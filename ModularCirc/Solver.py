from .Time import TimeClass

class Solver():
    def __init__(self, type:str=None, time_object:TimeClass=None) -> None:
        self.type = None
        self._pvk = []
        self._svk = []
        
    def setup(self, state_variable_dictionary:dict)->None:
        print('Blah')
        for key, component in state_variable_dictionary.items():
            if component.dudt_func is not None:
                print(f" -- Variable {key} added to the principal variable key list.")
                self._pvk.append(key)
            elif component.u_func is not None:
                print(f" -- Variable {key} added to the secondary variable key list.")
                self._svk.append(key)
            else:
                continue
        return
            
    @property
    def pvk(self):
        return self._pvk
    
    @ property
    def svk(self):
        return self._svk