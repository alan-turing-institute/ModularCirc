import numpy as np
import pandas as pd

TEMPLATE_TIME_SETUP_DICT = {
    'name'    :  'generic',
    'ncycles' :  5,
    'tcycle'  :  1.0,
    'dt'      :  0.1,
    'export_min' : 1
 }

class TimeClass():
    def __init__(self, time_setup_dict) -> None:
        self._time_setup_dict = time_setup_dict
        self._initialize_time_array()
        self.cti = 0 # current time step index

    @property
    def ncycles(self):
        if 'ncycles' in self._time_setup_dict.keys():
            return self._time_setup_dict['ncycles']
        else:
            return None

    @property
    def tcycle(self):
        if 'tcycle' in self._time_setup_dict.keys():
            return self._time_setup_dict['tcycle']
        else:
            return None

    @property
    def dt(self):
        if 'dt' in self._time_setup_dict.keys():
            return self._time_setup_dict['dt']
        else:
            return None

    @property
    def export_min(self):
        if 'export_min' in self._time_setup_dict.keys():
            return self._time_setup_dict['export_min']
        else:
            return None

    def _initialize_time_array(self):
        # discretization of on heart beat, used as template (pure numpy)
        self._one_cycle_t = np.linspace(
            start=0.0,
            stop=self.tcycle,
            num=int(self.tcycle / self.dt) + 1,
            dtype=np.float64
        )

        # Pre-calculate array sizes for efficiency
        n_per_cycle = len(self._one_cycle_t) - 1  # exclude last point to avoid duplication
        total_points = self.ncycles * n_per_cycle + 1  # +1 for final point
        
        # Pre-allocate arrays (much faster than list comprehensions)
        self._sym_t = np.empty(total_points, dtype=np.float64)
        self._cycle_t = np.empty(total_points, dtype=np.float64)
        
        # Vectorized array filling
        for cycle in range(self.ncycles):
            start_idx = cycle * n_per_cycle
            end_idx = start_idx + n_per_cycle
            self._sym_t[start_idx:end_idx] = self._one_cycle_t[:-1] + cycle * self.tcycle
            self._cycle_t[start_idx:end_idx] = self._one_cycle_t[:-1]
        
        # Add final point
        self._sym_t[-1] = self._one_cycle_t[-1] + (self.ncycles - 1) * self.tcycle
        self._cycle_t[-1] = self._one_cycle_t[-1]
        
        # Convert to pandas Series for compatibility with existing code
        self._sym_t = pd.Series(self._sym_t)
        self._cycle_t = pd.Series(self._cycle_t)

        self.time = pd.DataFrame({'cycle_t': self._cycle_t, 'sym_t': self._sym_t})

        # the total number of time steps including initial time step
        self.n_t = len(self._sym_t)

        # the number of time steps in a cycle
        self.n_c = len(self._one_cycle_t)
        return

    def new_time_step(self):
        self.cti += 1
