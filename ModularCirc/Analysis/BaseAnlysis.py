from ..Time import TimeClass
from ..StateVariable import StateVariable
from ..Models.OdeModel import OdeModel
from ..HelperRoutines import bold_text
from pandera.typing import DataFrame, Series
from ..Models.OdeModel import OdeModel

import numpy as np
import matplotlib.pyplot as plt

class ValveData():
    def __init__(self, name) -> None:
        self._name = name 
        
    def set_opening_closing(self, open, closed):
        self._open = open
        self._closed = closed
        
    def __repr__(self) -> str:
        return f"Valve {self._name}: \n" + f" - opening ind: {self._open} \n" + f" - closing ind: {self._closed} \n" 
    
    @property
    def open(self):
        return self._open
    
    @property
    def closed(self):
        return self._closed
    
    
class VentricleData():
    def __init__(self, name:str, volume_unit:str='ml') -> None:
        self._name = name
        self._vu   = volume_unit
        
    def set_volumes(self, edv, esv):
        self._edv = edv
        self._esv = esv
        
    def __repr__(self) -> str:
        return (f"Ventricle {self._name}: \n" + 
                f" - EDV: {self._edv:.2e} {self._vu}\n" + 
                f" - ESV: {self._esv:.2e} {self._vu}" )
    
    @property
    def edv(self):
        return self._edv
    
    @property
    def esv(self):
        return self._esv
    

class BaseAnalysis():
    def __init__(self, model:OdeModel=None) -> None:
        self.model = model
        
        self.valves= dict()
        self.ventricles = dict()
        
        self.tind  = np.arange(start=self.model.time_object.n_t-self.model.time_object.n_c,
                               stop =self.model.time_object.n_t)
        self.tsym  = self.model.time_object._one_cycle_t.values
        
    def plot_t_v(self, component:str, ax=None, time_units:str='s', volume_units:str='mL'):
        if ax is None:
            _, ax = plt.subplots(figsize=(5,5))
        ax.plot(self.tsym, self.model.commponents[component].V.values[self.tind],linewidth=4,)
        ax.set_title(component.upper() + ': Volume trace')
        ax.set_xlabel(f'Time (${time_units}$)')
        ax.set_ylabel(f'Volume (${volume_units}$)')
        ax.set_xlim(self.tsym[0], self.tsym[-1])
        return ax
    
    def plot_t_p(self, component:str, ax=None, time_units:str='s', pressure_units:str='mmHg'):
        if ax is None:
            _, ax = plt.subplots(figsize=(5,5))
        ax.plot(self.tsym, self.model.commponents[component].P.values[self.tind],linewidth=4,)
        ax.set_title(component.upper() + ': Pressure trace')
        ax.set_xlabel(f'Time (${time_units}$)')
        ax.set_ylabel(f'Volume (${pressure_units}$)')
        ax.set_xlim(self.tsym[0], self.tsym[-1])
        return ax
    
    def plot_p_v_loop(self, component, ax=None, volume_units:str='mL', pressure_units:str="mmHg"):
        if ax is None:
            _, ax = plt.subplots(figsize=(5,5))
        ax.plot(
            self.model.commponents[component].V.values[self.tind],
            self.model.commponents[component].P.values[self.tind],
            linewidth=4,
        )
        ax.set_title(component.upper() + ': PV loop')
        ax.set_xlabel(f'Volume (${volume_units}$)')
        ax.set_ylabel(f'Pressure (${pressure_units}$)')
        return ax

    def plot_fluxes(self, component:str, ax=None, time_units:str='s', volume_units:str='mL'):
        if ax is None:
            _, ax = plt.subplots(figsize=(5,5))
        ax.plot(
            self.tsym,
            self.model.commponents[component].Q_i.values[self.tind] - 
            self.model.commponents[component].Q_o.values[self.tind],
            linestyle='-',
            linewidth=4,
            alpha=0.6,
            label=f'{component} $dV/dt$'
        )     
        ax.plot(
            self.tsym,
            self.model.commponents[component].Q_i.values[self.tind],
            linestyle=':',
            linewidth=4,
            label=f'{component} $Q_i$'
        )  
        ax.plot(
            self.tsym,
            self.model.commponents[component].Q_o.values[self.tind],
            linestyle=':',
            linewidth=4,
            label=f'{component} $Q_o$'
        )      
        ax.set_title(f"{component.upper()}: Fluxes")   
        ax.set_xlabel(f'Time (${time_units}$)')
        ax.set_ylabel(f'Flux (${volume_units}\cdot {time_units}$)')   
        ax.set_xlim(self.tsym[0], self.tsym[-1])
        ax.legend()    
        
    def compute_opening_closing_valve(self, component:str, shift:float=0.0):
        valve = self.model.commponents[component]
        nshift= int(shift/ self.model.time_object.dt)
        self.valves[component] = ValveData(component)
        
        if not hasattr(valve, 'PHI'):
            pi    = valve.P_i.values[self.tind]
            po    = valve.P_o.values[self.tind]
            is_open = pi > po
        else:
            phi = valve.PHI.values[self.tind]
            min_phi = np.min(phi)
            is_open = (phi - min_phi) > 1.0e-2
        is_open_shifted = np.roll(is_open, -nshift)
        
        ind = np.arange(len(is_open))[is_open_shifted]
        self.valves[component].set_opening_closing(open = ind[0],
                                                   closed= ind[-1])
        
        
    def compute_ventricle_volume_limits(self, component:str, vic:int, voc:int):
        ventricle = self.model.commponents[component]
        volume    = ventricle.V.values[self.tind]
        
        self.ventricles[component] = VentricleData(component)
        self.ventricles[component].set_volumes(edv=volume[vic], esv=volume[voc])
                
    
    def compute_cardiac_output(self, component:str):
        valve = self.model.commponents[component]
        dt    = self.model.time_object.dt
        T     = self.model.time_object.tcycle / 60.0
        
        q     = valve.Q_i.values[self.tind]
        self.CO = q[:-1].sum() * dt / T
        
        return self.CO
            
        
            
        