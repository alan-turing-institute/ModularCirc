import pandas as pd
from ..HelperRoutines import bold_text

class ParametersObject():
    def __init__(self, name='ParametersObject') -> None:
        self.components = {}
        self._name = name
        
        
    def __repr__(self) -> str:
        out = self._name + ' parameters set: \n'
        for comp in self.components: 
            out += f" * Component - {bold_text(str(comp))}" + '\n'
            for key, item in self.components[comp].items():
                if isinstance(item, float):
                    out += (f"  - {bold_text(str(key)):<20} : {item:.3e}") + '\n'
                else:
                    out += (f"  - {bold_text(str(key)):<20} : {item}") + '\n'
            out += '\n'
        return out
    
    def __getitem__(self, key):
        return self.components[key]
        
    def _set_comp(self, key:str, set, **kwargs):
        if key not in set:
            raise Exception('Wrong key!')
        for k, val in kwargs.items():
            if val is None: continue
            assert k in self[key].index
            self[key].loc[k] = val