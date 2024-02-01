import numpy as np
import sympy as sp
# from .timeclass import TimeSeries


def resistor_model_flow(p_in:float, p_out:float, r:float) -> float:
    """
    Resistor model.

    Args:
        p_in (float): input pressure
        p_out (float): ouput pressure
        r (float): resistor constant

    Returns:
        float: q (flow rate through resistive unit)
    """
    return (p_in - p_out) / r

def resistor_model_dp(q_in:float, r:float) -> float:
    return q_in * r

def grounded_capacitor_model_pressure(v:float, v_ref:float, c:float) -> float:
    """
    Capacitor model with constant capacitance. 

    Args:
    ----
        v (float): current volume
        v_ref (float): reference volume for which chamber pressure is zero
        c (float): capacitance constant

    Returns:
    --------
        float: pressure at input node
    """
    return (v - v_ref) / c

def chamber_volume_rate_change(q_in:float, q_out:float) -> float:
    """
    Volume change rate in chamber

    Args:
        q_in (float): _description_
        q_out (float): _description_

    Returns:
        float: _description_
    """
    return q_in - q_out

def relu_max(val:float) -> float: 
    return np.max([val, 0.0])

def softplus(val:float, alpha:float) -> float:
    if alpha * val <= 20.0:
        return 1/ alpha * np.log(1 + np.exp(alpha * val))
    else:
        return val
    
def get_softplus_max(alpha:float):
    """
    Method for generating softmax lambda function based on predefined alpha values
    
    Args:
    ----
        alpha (float): softplus alpha value

    Returns:
    -------
        function: softplus function with fixed alpha
    """
    return lambda val : softplus(val=val, alpha=alpha)

def non_ideal_diode_flow(p_in:float, p_out:float, r:float, max_func=relu_max) -> float:
    """
    Nonideal diode model with the option to choose the re

    Args:
    -----
        p_in (float): input pressure
        p_out (float): output pressure
        r (float): valve constant resistance
        max_func (function): function that dictates when valve oppens

    Returns:
        float: q (flow rate through valve)
    """
    return max_func([p_in - p_out, 0.0]) / r


def leaky_diode_flow(p_in:float, p_out:float, r_o:float, r_r:float) -> float:
    """
    Leaky diode model that outputs the flow rate through a leaky diode

    Args:
        p_in (float): input pressure 
        p_out (float): output pressure
        r_o (float): outflow resistance
        r_r (float): regurgitant flow resistance

    Returns:
        float: q flow rate through diode
    """
    if p_in > p_out:
        return (p_in - p_out) / r_o
    else:
        return (p_in - p_out) / r_r
    
def activation_function_1(t:float, t_max:float, t_tr:float, tau:float) -> float:
    """
    Activation function that dictates the transition between the passive and active behaviors.
    Based on the definition used in Naghavi et al (2024).

    Args:
        t (float):     current time within the cardiac cycle
        t_max (float): time to peak tension
        t_tr (float):  transition time
        tau (float):   the relaxation time constant

    Returns:
        float: activation function value
    """
    if t <= t_tr:
        return 0.5 * (1.0 - np.cos(np.pi * t / t_max))
    else:
        return 0.5 * np.exp(-(t - t_tr)/tau)
    
def chamber_pressure_function(v:float, t:float, activation_function, active_law, passive_law) ->float:
    """
    Generic function returning the chamber pressure at a given time for a given imput

    Args:
        v (float): current volume
        t (float): current time
        activation_function (procedure): activation function
        active_law (procedure): active p-v relation
        passive_law (procedure): passive p-v relation

    Returns:
        float: pressure
    """
    a = activation_function(t)
    return a * active_law(v,t) + (1 - a) * passive_law(v,t)

def chamber_linear_elastic_law(v:float, E:float, v_ref:float, *args, **kwargs) -> float:
    """
    Linear elastance model

    Args:
        v (float): volume
        E (float): Elastance
        v_ref (float): reference volume
        
    Returns:
        float: chamber pressure
    """
    return E * (v - v_ref)

def chamber_exponential_law(v:float, E:float, k:float, v_ref:float, *args, **kwargs) -> float:
    """
    Exponential chamber law

    Args:
        v (float): volume
        E (float): elastance constant
        k (float): exponential factor
        v_ref (float): reference volume

    Returns:
        float: chamber pressure
    """
    return E * np.exp(k * (v - v_ref) - 1)

TEMPLATE_TIME_SETUP_DICT = {
    'name'    :  'generic',
    'ncycles' :  5,
    'tcycle'  :  1.0,
    'dt'      :  0.1
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
    
    def _initialize_time_array(self):
        # discretization of on heart beat, used as template
        self._cycle_t = np.linspace(
            start= 0.0,
            stop = self.tcycle,
            num  = int(self.tcycle / self.dt)+1,
            dtype= np.float64
            )
        
        # discretization of the entire simulation duration
        self._sym_t = np.array(
            [t+cycle*self.tcycle for cycle in range(self.ncycles) for t in self._cycle_t[:-1]]
        )
        
        # array of the current time within the heart cycle
        self._sym_t_norm = np.array(
            [t for _ in range(self.ncycles) for t in self._cycle_t[:-1]]
        )
        
        # the total number of time steps including initial time step
        self.n_t = len(self._sym_t)
        
    def new_time_step(self):
        self.cti += 1
        

class Chamber():
    def __init__(self, 
                 time_object:TimeClass, 
                #  main_var:str
                 ) -> None:
        self._state_var = dict() 
        self._to      = time_object
        self.main_var = main_var
        
        self._state_var['P_i'] = np.zeros((time_object.n_t,))
        self._state_var['P_o'] = np.zeros((time_object.n_t,))
        self._state_var['Q_i'] = np.zeros((time_object.n_t,))
        self._state_var['Q_o'] = np.zeros((time_object.n_t,))
        
    @property
    def P_i(self):
        return self._state_var['P_i']
    
    @property
    def P_o(self):
        return self._state_var['P_o']
    
    @property
    def Q_i(self):
        return self._state_var['Q_i']
    
    @property
    def Q_o(self):
        return self._state_var['Q_o']
    
    def comp_dvdt(self, ind):
        return self.Q_i[ind] - self.Q_o[ind]
    
    def comp_v(self, q_i, q_o, v_prev):
        return v_prev  + self._to.dt * self.comp_dvdt(q_i, q_o)      
        
        
class Rc_component(Chamber):
    def __init__(self, 
                 time_object:TimeClass, 
                #  main_var:str, 
                 r:float, 
                 c:float, 
                 v_ref:float
                 ) -> None:
        # super().__init__(time_object, main_var)
        super().__init__(time_object)
        self._state_var['V']   = np.zeros((time_object.n_t,))
        self.R = r
        self.C = c
        self.V_ref = v_ref
        
    @property
    def V(self):
        return self._state_var['V']
        
    def comp_dpdt(self, Q_in:float, Q_out:float):
        return 1.0 /  self.C * (Q_in - Q_out)
    
    def comp_p(self, V):
        return (V - self.V_ref) / self.C
    
    
class Valve_leaky_diode(Chamber):
    def __init__(self, 
                 time_object: TimeClass,
                 r:float, 
                 max_func
                 ) -> None:
        super().__init__(time_object)
        self.R = r
        self._state_var['Q_o'] = self.Q_i
        self.max_func = max_func
        
    def comp_q(self, p_i, p_o):
        return self.max_func(p_i - p_o) / self.R
    
class HC_constant_elastance(Chamber):
    def __init__(self, 
                 time_object: TimeClass,
                 E_pas: float, 
                 E_act: float,
                 activation_function_template = activation_function_1,
                 *args, **kwargs
                 ) -> None:
        super().__init__(time_object)
        self._af = lambda t : activation_function_template(t, *args, **kwargs)
        self.E_pas = E_pas
        self.E_act = E_act
        self.eps = 1.0e-3
        
        self._state_var['V']   = np.zeros((time_object.n_t,))
        
    def comp_E(self, t:float) -> float:
        return self._af(t) * self.E_act + (1.0 - self._af(t)) * self.E_pas
    
    def comp_dEdt(self, t:float) -> float:
        return (self.comp_E(t + self.eps) - self.comp_E(t - self.eps)) / 2.0 / self.eps
    
    def comp_dvdt(self, q_in:float, q_out:float) -> float:
        return q_in - q_out
    
    def comp_dpdt(self, t:float, q_in:float, q_out:float) -> float:
        dvdt = self.comp_dvdt(q_in=q_in, q_out=q_out)
        dEdt = self.comp_dEdt(t)
        
        
        
        
class OdeModel():
    def __init__(self, time_setup_dict) -> None:
        self.time_object = TimeClass(time_setup_dict=time_setup_dict)
    
    def connect_modules(self, module1:Chamber, module2:Chamber) ->None:
        module2._state_var['Q_i'] = module1.Q_o
        module2._state_var['P_o'] = module1.P_o
    
class NaghaviModel(OdeModel):
    def __init__(self, time_setup_dict) -> None:
        super().__init__(time_setup_dict)
        
        # define a set of main variables
        self._main_variable_keys = {
            'p_lv', # pressure in the left ventricle
            'p_la', # pressure in the left atrium
            'p_ao', # pressure in the aorta
            'p_ar', # pressure in the arteries
            'p_lv', # pressure in the vena cava
        }
        
        self.av = Valve_leaky_diode(time_object=self.time_object,
                                    r = 1.0,
                                    max_func=relu_max
                                    )
        
        self.ao = Rc_component(time_object=self.time_object, 
                                  r = 1.0,
                                  c = 1.0, 
                                  v_ref = 1.0,
                                  )
        
        self.art = Rc_component(time_object=self.time_object,
                                r = 1.0,
                                c = 1.0,
                                v_ref= 1.0,
                                )
        
        self.ven = Rc_component(time_object=self.time_object,
                                r = 1.0,
                                c = 1.0,
                                v_ref= 1.0,
                                )
        
        self.connect_modules(self.ao,  self.art)
        self.connect_modules(self.art, self.ven)
        
        