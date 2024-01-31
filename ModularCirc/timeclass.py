import numpy as np

from dataclasses import dataclass

@dataclass
class TimeParameters():
    ncycles : int 
    tcycle  : float 
    dt      : float 

class TimeSeries():
        
    def __init__(self, ncycles:int =5, tcycle:float=1.0, dt=0.01) -> None:
        self.parameters = TimeParameters(ncycles=ncycles, tcycle=tcycle, dt=dt)
        self.series = np.arange(
            0.0,                                                # initial time point
            self.parameters.ncycles*self.parameters.tcycle,     # no of cycles X length of cycle
            self.parameters.dt                                  # timestep size
            )
        
    def __repr__(self) -> str:
        return 'TimeSeries Object \n' + self.parameters.__repr__() + '\n' + self.series.__repr__()
